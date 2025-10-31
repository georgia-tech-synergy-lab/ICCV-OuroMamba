import torch
import time
import custom_quantization_cuda as cuda_extension

def roundf_like(x: torch.Tensor) -> torch.Tensor:
    """
    Mimics C's roundf behavior: round half away from zero.
    For x >= 0, returns floor(x + 0.5); for x < 0, returns -floor(-x + 0.5).
    """
    return torch.where(x >= 0, torch.floor(x + 0.5), -torch.floor(-x + 0.5))

def simulate_quantized_gemm(weight, activation, scale_weight, scale_activation):
    """
    Simulate the quantized GEMM using roundf-like rounding:
      - Quantize weight: roundf_like(weight/scale_weight), clamp to [-7, 7].
      - Quantize activation: roundf_like(activation/scale_activation), clamp to [-127, 127].
      - Perform integer matrix multiplication (convert to float to mimic int arithmetic).
    """
    # Quantize weights.
    w_q = roundf_like(weight / scale_weight).clamp(-7, 7).to(torch.int32)
    # Quantize activations.
    a_q = roundf_like(activation / scale_activation).clamp(-127, 127).to(torch.int32)
    # Perform GEMM. We cast to float to do the matmul, which simulates integer accumulation.
    return torch.matmul(w_q.float(), a_q.float())

def test_correctness_int4_int8_gemm():
    """
    Tests correctness by comparing:
      CUTLASS int4×int8 GEMM (via your CUDA extension)
      vs.
      Simulated quantized GEMM (Python).
    """
    device = torch.device("cuda")
    torch.manual_seed(0)

    # Define GEMM dimensions
    M, K, N = 128, 64, 32

    # Create random FP16 weight and activation matrices
    weight_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    activation_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)

    # Quantization scale factors
    scale_weight = 0.1
    scale_activation = 0.05

    # 1) Run the CUTLASS INT4×INT8 GEMM from the extension
    padded_virtualK = (K + 1) // 2
    weight_packed = torch.empty((M, padded_virtualK), dtype=torch.int8, device=device)
    cuda_extension.pack_int4_kernel_fp16_rowmajor(
        weight_fp16, scale_weight, M, K, padded_virtualK, weight_packed
    )
    quantized_result = cuda_extension.int4_int8_gemm(
        weight_packed, activation_fp16, M, N, K, scale_activation
    )

    # 2) Simulate quantized GEMM in Python
    sim_int32 = simulate_quantized_gemm(weight_fp16, activation_fp16, scale_weight, scale_activation)

    # 3) Compare and log statistics
    print("==== Correctness Test: Quantized GEMM ====")
    print("---- Matrix Summaries ----")
    print("Simulated GEMM (int32): Mean: {:.4f}, Min: {:.4f}, Max: {:.4f}".format(
        sim_int32.float().mean().item(), sim_int32.float().min().item(), sim_int32.float().max().item()))
    print("CUTLASS int4×int8 GEMM (int32): Mean: {:.4f}, Min: {:.4f}, Max: {:.4f}".format(
        quantized_result.float().mean().item(), quantized_result.float().min().item(), quantized_result.float().max().item()))

    # Compute absolute differences
    diff_sim_quant = (sim_int32.float() - quantized_result.float()).abs()
    mean_abs_diff = diff_sim_quant.mean().item()
    max_abs_diff = diff_sim_quant.max().item()

    # Compute relative differences (all elements)
    denom = quantized_result.float().abs() + 1e-8
    rel_diff_all = (diff_sim_quant / denom).mean().item()

    # Also compute a threshold-based relative difference to avoid huge blow-ups near zero
    threshold = 1.0
    mask = quantized_result.float().abs() > threshold
    if mask.any():
        rel_diff_thresholded = (diff_sim_quant[mask] / denom[mask]).mean().item()
    else:
        rel_diff_thresholded = float('nan')

    print("\n---- Difference Metrics (Simulated vs CUTLASS) ----")
    print(f"Mean absolute difference: {mean_abs_diff:.4f}")
    print(f"Maximum absolute difference: {max_abs_diff:.4f}")
    print(f"Mean relative difference (all elements): {rel_diff_all:.4f}")
    print(f"Mean relative difference (|CUTLASS| > {threshold}): {rel_diff_thresholded:.4f}")

    # Sum of elements as an overall sanity check
    print("\n---- Sum of Elements ----")
    print(f"Simulated GEMM sum: {sim_int32.float().sum().item():.4f}")
    print(f"CUTLASS GEMM sum: {quantized_result.float().sum().item():.4f}")


def benchmark_fp16_gemm(M, K, N, iters=100):
    """
    Times a standard FP16 GEMM using torch.matmul.
    """
    device = torch.device('cuda')
    weight_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    activation_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    
    # Warm-up
    for _ in range(10):
        _ = torch.matmul(weight_fp16, activation_fp16)
    torch.cuda.synchronize()
    
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        C_fp16 = torch.matmul(weight_fp16, activation_fp16)
    end.record()
    torch.cuda.synchronize()
    
    avg_time = start.elapsed_time(end) / iters  # milliseconds
    return avg_time, C_fp16

def benchmark_int4_int8_gemm(M, K, N, iters=100, scale_weight=0.1, scale_activation=0.05):
    """
    Times the INT4×INT8 GEMM from the CUDA extension.
    """
    device = torch.device('cuda')
    weight_fp16 = torch.randn(M, K, dtype=torch.float16, device=device)
    activation_fp16 = torch.randn(K, N, dtype=torch.float16, device=device)
    
    # Pack the weight matrix
    padded_virtualK = (K + 1) // 2
    weight_packed = torch.empty((M, padded_virtualK), dtype=torch.int8, device=device)
    cuda_extension.pack_int4_kernel_fp16_rowmajor(
        weight_fp16, scale_weight, M, K, padded_virtualK, weight_packed
    )
    
    # Warm-up
    for _ in range(10):
        _ = cuda_extension.int4_int8_gemm(weight_packed, activation_fp16, M, N, K, scale_activation)
    torch.cuda.synchronize()
    
    # Timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    for _ in range(iters):
        C_int4 = cuda_extension.int4_int8_gemm(weight_packed, activation_fp16, M, N, K, scale_activation)
    end.record()
    torch.cuda.synchronize()
    
    avg_time = start.elapsed_time(end) / iters  # milliseconds
    return avg_time, C_int4


if __name__ == "__main__":
    # 1) Correctness test
    test_correctness_int4_int8_gemm()
    
    # 2) Performance benchmark
    print("\n==== Performance Benchmark ====")
    M, K, N = 8192, 8192, 8192
    iters = 50

    print("\nBenchmarking FP16 GEMM...")
    fp16_time, _ = benchmark_fp16_gemm(M, K, N, iters)
    print(f"FP16 GEMM average time per iteration (ms): {fp16_time:.4f}")
    
    print("\nBenchmarking INT4×INT8 GEMM...")
    int4_time, _ = benchmark_int4_int8_gemm(M, K, N, iters)
    print(f"INT4×INT8 GEMM average time per iteration (ms): {int4_time:.4f}")
    
    if int4_time > 0:
        speedup = fp16_time / int4_time
    else:
        speedup = float('inf')
    print(f"\nSpeedup of INT4×INT8 GEMM over FP16 GEMM: {speedup:.2f}x")
