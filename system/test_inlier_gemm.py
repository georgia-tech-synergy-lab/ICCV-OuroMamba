import torch
import time

from custom_quantization_cuda import (
    pack_int4_kernel_fp16_rowmajor_cuda,
    fused_quantized_gemm_packedW_cuda
)

# -------------------------------
# Dummy quantizer for outlier extraction test.
# Replace this with your actual quantizer class.
# -------------------------------
class DummyQuantizer:
    def extract_outliers(self, x, n_refresh=10, static_o_list=None, static_s=None, qtype='uni'):
        """
        Dummy implementation that mimics outlier extraction:
          - Computes a threshold as mean + std of x.
          - Zeros out the elements above the threshold in the inlier matrix.
          - Creates a sparse COO tensor for the outliers.
          - Returns static_s (reshaped), a dummy s_outlier, the inlier matrix, and the sparse tensor.
        """
        b, l, d = x.shape
        # Compute threshold (simple criterion)
        threshold = x.mean() + x.std()
        mask = x > threshold
        
        # Create inlier matrix (copy and zero out outlier positions)
        inlier = x.clone()
        inlier[mask] = 0
        
        # Create sparse COO tensor for outliers.
        # For simplicity, we build it using nonzero indices and corresponding values.
        outlier_indices = mask.nonzero(as_tuple=False).t()  # shape: [ndim, num_outliers]
        outlier_values = x[mask]
        outlier_coo = torch.sparse_coo_tensor(outlier_indices, outlier_values, size=x.size())
        
        # Create dummy scaling factors for outliers: here, simply ones.
        s_outlier = torch.ones(b * l, 1, device=x.device)
        
        # Reshape static_s to [b*l, 1] if provided.
        static_s_out = static_s.view(b * l, 1) if static_s is not None else None
        
        return static_s_out, s_outlier, inlier.view(b * l, d), outlier_coo

def main():
    # Use CUDA if available
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # ----------------------------------------------------------
    # 1) SET DIMENSIONS (LARGE) for GEMM tests
    # ----------------------------------------------------------
    M, K, N = 8192, 4096, 4096

    # Create per-row weight scales (for M rows) and per-column activation scales (for N columns)
    weight_scale = torch.rand(M, device=device, dtype=torch.float32) * 1.5 + 0.5
    act_scale = torch.rand(N, device=device, dtype=torch.float32) * 1.5 + 0.5

    # Generate random integer matrices in [-20, 20].
    weight_int = torch.randint(-20, 21, (M, K), device=device, dtype=torch.int32)
    act_int = torch.randint(-20, 21, (K, N), device=device, dtype=torch.int32)

    # Convert these to FP16 by multiplying each row/col by its scale.
    weight_fp = (weight_int.float() * weight_scale.unsqueeze(1)).to(torch.float16)
    act_fp = (act_int.float() * act_scale.unsqueeze(0)).to(torch.float16)

    # We'll create an output for the new fused kernel
    output_packedW = torch.zeros((M, N), device=device, dtype=torch.float32)

    # ----------------------------------------------------------
    # 2) PACK THE WEIGHT
    # ----------------------------------------------------------
    pad_align = 16
    paddedK = ((K + pad_align - 1) // pad_align) * pad_align  # ensure padded dimension is multiple of 16
    packed_weight_int8 = torch.empty((M, paddedK), dtype=torch.int8, device=device)

    pack_int4_kernel_fp16_rowmajor_cuda(
        weight_fp,       # [M, K] in FP16
        weight_scale,    # per-row scale [M] in FP32
        M,
        K,
        paddedK,
        packed_weight_int8
    )

    # ----------------------------------------------------------
    # 3) TIMING: STANDARD FP16 MATMUL VS FUSED KERNEL
    # ----------------------------------------------------------
    warmup_iters = 5
    test_iters = 100

    for _ in range(warmup_iters):
        _ = weight_fp @ act_fp
        fused_quantized_gemm_packedW_cuda(
            packed_weight_int8,
            weight_scale,
            M, K, N,
            act_fp,
            act_scale,
            output_packedW
        )
    if device == 'cuda':
        torch.cuda.synchronize()

    # Measure standard FP16 matmul time
    matmul_start = time.time()
    for _ in range(test_iters):
        out_matmul = weight_fp @ act_fp
    if device == 'cuda':
        torch.cuda.synchronize()
    matmul_end = time.time()
    avg_matmul_time = (matmul_end - matmul_start) / test_iters

    # Measure fused kernel time
    fused_start = time.time()
    for _ in range(test_iters):
        fused_quantized_gemm_packedW_cuda(
            packed_weight_int8,
            weight_scale,
            M, K, N,
            act_fp,
            act_scale,
            output_packedW
        )
    if device == 'cuda':
        torch.cuda.synchronize()
    fused_end = time.time()
    avg_fused_time = (fused_end - fused_start) / test_iters

    # Report GEMM performance results
    print(f"\n--- Performance Results for {M}x{K}x{N} ---")
    print(f"Average FP16 matmul time  : {avg_matmul_time:.6f} s")
    print(f"Average Fused kernel time : {avg_fused_time:.6f} s")
    print(f"Speedup (matmul / fused)  : {avg_matmul_time / avg_fused_time:.3f}")

    # ----------------------------------------------------------
    # 4) (Optional) CPU REFERENCE & CORRECTNESS CHECK
    # ----------------------------------------------------------
    weight_q = weight_int.float().clamp(-7, 7)
    act_q = act_int.float().clamp(-7, 7)
    ref = (weight_q @ act_q) * (weight_scale.unsqueeze(1) * act_scale.unsqueeze(0))
    diff = torch.abs(output_packedW - ref)
    print("\n--- Correctness Check ---")
    print("Max absolute difference :", diff.max().item())
    print("Mean absolute difference:", diff.mean().item())

    # ----------------------------------------------------------
    # 5) PERFORMANCE TEST: extract_outliers for [b, l, d] = [64, 128, 8192]
    # ----------------------------------------------------------
    # Dimensions for the outlier extraction test
    b, l, d = 64, 128, 4096
    # Create a random input tensor (using float32 for this example)
    x_for_extract = torch.randn(b, l, d, device=device, dtype=torch.float32)
    n_refresh = 10

    # Dummy static scaling factors (if needed, matching expected shape [b, l, 1])
    static_s = torch.rand(b, l, 1, device=device, dtype=torch.float32)

    # Create an instance of the dummy quantizer (replace with your quantizer if available)
    quantizer = DummyQuantizer()

    # Warm-up iterations to stabilize performance
    for _ in range(5):
        _ = quantizer.extract_outliers(x_for_extract, n_refresh=n_refresh, static_o_list=None, static_s=static_s, qtype='uni')
    if device == 'cuda':
        torch.cuda.synchronize()

    # Time the extract_outliers function over several iterations
    test_iters = 100
    start = time.time()
    for _ in range(test_iters):
        _ = quantizer.extract_outliers(x_for_extract, n_refresh=n_refresh, static_o_list=None, static_s=static_s, qtype='uni')
    if device == 'cuda':
        torch.cuda.synchronize()
    end = time.time()
    avg_extract_time = (end - start) / test_iters

    print(f"\n--- Performance Results for extract_outliers with shape [{b}, {l}, {d}] ---")
    print(f"Average extract_outliers time: {avg_extract_time:.6f} s")

if __name__ == '__main__':
    main()
