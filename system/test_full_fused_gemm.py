import torch
import time
import numpy as np
from tqdm import tqdm

from custom_quantization_cuda import (
    pack_int4_kernel_fp16_rowmajor_cuda,
    fused_quantized_gemm_packedW_cuda,
    fused_quantized_gemm_packedW_instrumented_cuda
)
import vim_GEMM

def int_quantizer_BWD(x, quantBits):
    """
    Quantizes the tensor to integer values based on quantBits.
    """
    mx = x.abs().max()
    scale = mx / (2**(quantBits-1)-1)
    x = torch.clamp(x, -mx, mx) / (scale+1e-8)
    x_int = torch.round(x)
    return x_int

def main():
    # Use CUDA if available.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------------------------------------------
    # 1) Setup dimensions and tensors for our fused GEMM test.
    # ----------------------------------------------------------
    M, K, N = 8192, 4096, 4096
    
    """
    A=(b*m,k) W=(k,n)
    Weight is (M,K), activation is (K,N)
    M = 197, K = 768/1536 (or can be 384), N = 3072/80 * batch
    D can be 192, 384, 12, 16
    L as 197
    """

    # Generate per-row weight scales and per-column activation scales.
    weight_scale = torch.rand(M, device=device, dtype=torch.float32) * 1.5 + 0.5
    act_scale = torch.rand(N, device=device, dtype=torch.float32) * 1.5 + 0.5

    # Generate random integer matrices in the range [-20, 20].
    weight_int = torch.randint(-20, 21, (M, K), device=device, dtype=torch.int32)
    act_int = torch.randint(-20, 21, (K, N), device=device, dtype=torch.int32)

    # Convert to FP16 using the corresponding scales.
    weight_fp = (weight_int.float() * weight_scale.unsqueeze(1)).to(torch.float16)
    act_fp = (act_int.float() * act_scale.unsqueeze(0)).to(torch.float16)

    # ----------------------------------------------------------
    # 2) Extract outliers from the activation matrix.
    # ----------------------------------------------------------
    threshold = act_fp.float().mean() + 1.5 * act_fp.float().std()
    mask = act_fp.float() > threshold  # Boolean mask of outliers

    # Create a copy of act_fp with outliers zeroed out.
    act_fp_inlier = act_fp.clone()
    act_fp_inlier[mask] = 0

    # Build sparse COO tensor for outlier indices.
    outlier_indices = mask.nonzero(as_tuple=False).t()  # shape: [2, nnz]
    
    nnz = outlier_indices.shape[1]
    total_elements = act_fp.numel()
    nnz_percentage = (nnz / total_elements) * 100
    print("nnz:", nnz)
    print("nnz percentage: {:.4f}%".format(nnz_percentage))
    
    # Extract outlier values (FP16) and create dummy scaling factors.
    outlier_values = act_fp[mask].clone().half().contiguous().view(-1).to(device)
    act_outlier_scale = torch.ones(outlier_values.shape, device=device, dtype=torch.float32)
    n_lva_o = 16  # Dummy integer parameter for outlier quantization.

    # ----------------------------------------------------------
    # 3) Pack the weight for the fused GEMM kernel.
    # ----------------------------------------------------------
    pad_align = 16
    paddedK = ((K + pad_align - 1) // pad_align) * pad_align
    packed_weight_int8 = torch.empty((M, paddedK), dtype=torch.int8, device=device)

    pack_int4_kernel_fp16_rowmajor_cuda(
        weight_fp,       # FP16 weight tensor [M, K]
        weight_scale,    # Per-row scale [M] in FP32
        M,
        K,
        paddedK,
        packed_weight_int8
    )

    # Prepare output tensor for the fused kernel.
    output_packedW = torch.zeros((M, N), device=device, dtype=torch.float32)

    # ----------------------------------------------------------
    # 4) Time standard FP16 matmul vs. our fused kernel.
    # ----------------------------------------------------------
    warmup_iters = 5
    test_iters = 200

    # Warm-up iterations.
    for _ in range(warmup_iters):
        _ = weight_fp @ act_fp
        fused_quantized_gemm_packedW_cuda(
            packed_weight_int8,
            weight_scale,
            M, K, N,
            act_fp_inlier,
            act_scale,
            outlier_values,
            outlier_indices,
            act_outlier_scale,
            n_lva_o,
            output_packedW
        )
    if device == 'cuda':
        torch.cuda.synchronize()

    # Standard FP16 matmul timing.
    matmul_start = time.time()
    for _ in range(test_iters):
        _ = weight_fp @ act_fp
    if device == 'cuda':
        torch.cuda.synchronize()
    matmul_end = time.time()
    avg_matmul_time = (matmul_end - matmul_start) / test_iters

    # Fused kernel timing.
    fused_start = time.time()
    for _ in range(test_iters):
        fused_quantized_gemm_packedW_cuda(
            packed_weight_int8,
            weight_scale,
            M, K, N,
            act_fp_inlier,
            act_scale,
            outlier_values,
            outlier_indices,
            act_outlier_scale,
            n_lva_o,
            output_packedW
        )
    if device == 'cuda':
        torch.cuda.synchronize()
    fused_end = time.time()
    avg_fused_time = (fused_end - fused_start) / test_iters

    # run one instrumentation step
    fused_quantized_gemm_packedW_instrumented_cuda(
            packed_weight_int8,
            weight_scale,
            M, K, N,
            act_fp_inlier,
            act_scale,
            outlier_values,
            outlier_indices,
            act_outlier_scale,
            n_lva_o,
            output_packedW
        )
    

    print(f"\n--- Our Fused GEMM Performance Results for {M}x{K}x{N} ---")
    print(f"Average FP16 matmul time  : {avg_matmul_time:.6f} s")
    print(f"Average fused kernel time : {avg_fused_time:.6f} s")
    print(f"Speedup (FP16 / fused)    : {avg_matmul_time / avg_fused_time:.3f}")

    # ----------------------------------------------------------
    # 5) Time FP16 GEMM vs. vim_GEMM kernel (runtime comparison only).
    # ----------------------------------------------------------
    # Setup dimensions for the vim_GEMM test (with batch dimension).
    b = 1
    m, k, n = 8192, 4096, 4096
    testTurn = 200
    quantBits = 4  # Change to 8 if needed.

    # Create tensors for the FP16 GEMM test.
    A = torch.tensor(np.random.normal(1., 0.1, b * m * k), dtype=torch.float16, device=device).reshape(b, m, k)
    B = torch.tensor(np.random.normal(1., 0.1, n * k), dtype=torch.float16, device=device).reshape(n, k)
    smooth_scale = torch.tensor(np.random.normal(1., 0.1, k), dtype=torch.float16, device=device)

    # For quantBits == 4, adjust B and smooth_scale.
    if quantBits == 4:
        B_int4 = torch.tensor(np.random.normal(1., 0.1, n * (k // 2)), dtype=torch.float16, device=device).reshape(n, k // 2)
        smooth_scale = torch.tensor(np.random.normal(1., 0.1, k // 2), dtype=torch.float16, device=device)
    a_s = torch.tensor(np.random.normal(1., 0.1, m), dtype=torch.float16, device=device).reshape(m, 1)
    w_s = torch.tensor(np.random.normal(1., 0.1, n), dtype=torch.float16, device=device).reshape(n, 1)

    # Quantize B.
    B_q = int_quantizer_BWD(B_int4, quantBits) if quantBits == 4 else int_quantizer_BWD(B, quantBits)

    FP_time = []
    vim_kernel_time = []

    for _ in tqdm(range(testTurn), desc="vim_GEMM test"):
        # Standard FP16 GEMM timing.
        start = time.time()
        _ = A @ B.mT
        torch.cuda.synchronize()
        FP_time.append(time.time() - start)
        
        # vim_GEMM kernel timing.
        B_q_int8 = B_q.type(torch.int8).contiguous()
        A_contig = A.contiguous()
        start = time.time()
        _ = vim_GEMM.vim_GEMM(
            A_contig,
            B_q_int8,
            smooth_scale,
            a_s,
            w_s,
            16,         # Dummy integer parameter (e.g., block size)
            quantBits
        )
        torch.cuda.synchronize()
        vim_kernel_time.append(time.time() - start)

    avg_FP_time = sum(FP_time) / len(FP_time)
    avg_vim_time = sum(vim_kernel_time) / len(vim_kernel_time)

    print("\n--- vim_GEMM Kernel Performance Results ---")
    print(f"Average FP16 GEMM time       : {avg_FP_time:.6f} s")
    print(f"Average vim_GEMM kernel time : {avg_vim_time:.6f} s")
    print(f"Speedup (FP16 / vim_GEMM)    : {avg_FP_time / avg_vim_time:.3f}")

    # ----------------------------------------------------------
    # 6) Combined Runtime Comparison Summary.
    # ----------------------------------------------------------
    print("\n--- Combined GEMM Comparison Summary ---")
    print(f"Our fused GEMM average time   : {avg_fused_time:.6f} s")
    print(f"vim_GEMM kernel average time  : {avg_vim_time:.6f} s")
    if avg_fused_time < avg_vim_time:
        speedup_factor = avg_vim_time / avg_fused_time
        print(f"Our fused GEMM kernel is faster than vim_GEMM by a factor of {speedup_factor:.3f}")
    else:
        speedup_factor = avg_fused_time / avg_vim_time
        print(f"vim_GEMM kernel is faster than our fused GEMM kernel by a factor of {speedup_factor:.3f}")
    
if __name__ == '__main__':
    main()
