import torch
import custom_quantization_cuda as int4_gemm  # Import the custom CUDA extension

def test_quantized_gemm():
    torch.manual_seed(0)
    # Dimensions (K must be even for packing)
    M, K, N = 256, 256, 256

    # Define quantization scales for weight and activation
    weight_scale = 0.1  # example scale factor for weight
    act_scale = 0.2     # example scale factor for activation

    # For packing, compute padded width (physical stride) aligning to 16.
    pad_align = 16
    padded_virtualK = ((K + pad_align - 1) // pad_align) * pad_align

    # --------------------------------------------------------------
    # 1) Prepare the weight and activation matrices in FP16.
    #    Note: We generate integer values in [-8, 7] then cast to FP16.
    # --------------------------------------------------------------
    weight_fp16 = torch.randint(-8, 8, (M, K), dtype=torch.int8, device='cuda').to(torch.float16)
    act_fp = torch.randint(-8, 8, (K, N), dtype=torch.int8, device='cuda').to(torch.float16)

    # --------------------------------------------------------------
    # 2) Pack the weight only once.
    #    The kernel expects:
    #      - input weight (FP16) of shape [M, K]
    #      - scale factor (weight_scale)
    #      - dimensions M, K, and padded_virtualK (physical stride)
    #      - output tensor of shape [M, padded_virtualK] (int8)
    # --------------------------------------------------------------
    packed_weight = torch.empty((M, padded_virtualK), dtype=torch.int8, device='cuda')
    int4_gemm.pack_int4_kernel_fp16_rowmajor(
        weight_fp16, weight_scale, M, K, padded_virtualK, packed_weight
    )

    # --------------------------------------------------------------
    # 3) Prepare dummy tensors for outlier-related arguments.
    #    (They are unused in this test so we just pass placeholders.)
    # --------------------------------------------------------------
    act_outlier = torch.zeros((1,), dtype=torch.float16, device='cuda')         # dummy tensor
    act_outlier_indices = torch.zeros((2, 1), dtype=torch.int64, device='cuda') # dummy tensor
    act_outlier_scale = torch.tensor([0.0], dtype=torch.float16, device='cuda')   # dummy tensor
    n_lva_o = 0  # dummy integer parameter

    # --------------------------------------------------------------
    # 4) Run the fused quantized GEMM CUDA kernel.
    #    This kernel first packs the activation internally and then performs
    #    the int4 GEMM, scaling the accumulation by (weight_scale * act_scale).
    #    The output is returned as a FP16 tensor.
    # --------------------------------------------------------------
    C_cuda = int4_gemm.fused_quantized_gemm_packedW_cuda(
        packed_weight, weight_scale,
        M, K, N,
        act_fp, act_scale,
        act_outlier, act_outlier_indices, act_outlier_scale,
        n_lva_o
    )

    # --------------------------------------------------------------
    # 5) Simulate the quantized GEMM on the CPU.
    #    - Quantize both weight and activation matrices:
    #         q = round( value / scale )  then clip to [-7, 7]
    #    - Compute the reference int32 GEMM accumulation.
    #    - Finally, dequantize by multiplying by (weight_scale * act_scale) and round.
    # --------------------------------------------------------------
    weight_q = (weight_fp16 / weight_scale).round().clamp(-7, 7)
    act_q = (act_fp / act_scale).round().clamp(-7, 7)
    # Perform GEMM in int32
    C_ref_int32 = torch.matmul(weight_q.to(torch.int32).to(torch.float), act_q.to(torch.int32).to(torch.float ))
    # Dequantize the result
    C_ref = (C_ref_int32 * weight_scale * act_scale).round().to(torch.float16)

    # --------------------------------------------------------------
    # 6) Compare the results.
    # --------------------------------------------------------------
    print("Simulated reference GEMM result (sum over all elements):", C_ref.float().sum().item())
    print("Fused quantized GEMM result (sum over all elements):", C_cuda.float().sum().item())
    
    # Allow a tolerance of 1 unit in the accumulated int32 result after dequantization.
    if torch.allclose(C_cuda, C_ref, atol=1, rtol=1e-3):
        print("Test (quantized GEMM) PASSED!")
    else:
        diff = (C_cuda - C_ref).abs().max().item()
        print("Test (quantized GEMM) FAILED! Max abs diff: {}".format(diff))

if __name__ == "__main__":
    print("Running fused quantized GEMM correctness test...")
    test_quantized_gemm()
