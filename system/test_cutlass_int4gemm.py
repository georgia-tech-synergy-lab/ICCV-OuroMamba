import torch
import time
import int4_gemm  # Import the built extension

def test_int4_gemm():
    torch.manual_seed(0)
    # Dimensions (virtual_K must be even)
    M, virtual_K, N = 256, 256, 256

    # Generate random fp16 matrices. We generate integer values in [-8,7] and cast to fp16.
    A_fp16 = torch.randint(-8, 8, (M, virtual_K), dtype=torch.int8, device='cuda').to(torch.float16)
    B_fp16 = torch.randint(-8, 8, (virtual_K, N), dtype=torch.int8, device='cuda').to(torch.float16)
    
    # ================================================================
    # 1) Test baseline GEMM with alpha=1.0
    # ================================================================
    C_cuda = int4_gemm.int4_gemm(A_fp16, B_fp16, virtual_K, alpha=1.0)
    
    # Create reference result:
    # First, simulate the rounding and clipping on the CPU.
    A_ref = A_fp16.round().clamp(-8, 7).to(torch.float16)
    B_ref = B_fp16.round().clamp(-8, 7).to(torch.float16)
    C_ref = torch.matmul(A_ref, B_ref)  # in fp16
    C_ref = C_ref.to(torch.int32)       # accumulate in int32 reference

    # Compare sums and check exact element match:
    print("Reference GEMM result (sum over all elements):", C_ref.float().sum().item())
    print("CUTLASS int4 GEMM result (sum over all elements):", C_cuda.sum().item())
    if torch.equal(C_cuda, C_ref):
        print("Test (alpha=1.0) PASSED!")
    else:
        print("Test (alpha=1.0) FAILED!")
    
    # ================================================================
    # 2) Test scaled GEMM (alpha != 1.0)
    # ================================================================
    alpha_test = 2.5  # arbitrary scale factor
    C_cuda_scaled = int4_gemm.int4_gemm(A_fp16, B_fp16, virtual_K, alpha=alpha_test)
    
    # Reference: scale the existing reference
    # Because we want alpha * (A*B), we do:
    #   alpha * C_ref_no_scale
    # but recall C_ref_no_scale is int32, so do the scaling in float16 or float32.
    # We must be consistent with the extension’s final type (which is int32).
    # The extension uses half alpha but accumulates in int32, so the final result is also int32.
    # We'll replicate that in Python:
    C_ref_scaled = (C_ref.float() * alpha_test).round()  # emulate half-precision scaling if needed
    C_ref_scaled = C_ref_scaled.to(torch.int32)
    
    print("\nTesting alpha scaling = {}...".format(alpha_test))
    print("Reference scaled result (sum):", C_ref_scaled.float().sum().item())
    print("CUTLASS int4 GEMM scaled result (sum):", C_cuda_scaled.sum().item())
    
    if torch.allclose(C_cuda_scaled, C_ref_scaled, atol=1, rtol=1e-2): # max error could be 1
        print("Test (alpha={}) PASSED!".format(alpha_test))
    else:
        # If there's a mismatch, let's measure max error:
        diff = (C_cuda_scaled - C_ref_scaled).abs().max().item()
        print("Test (alpha={}) FAILED! Max abs diff: {}".format(alpha_test, diff))

def benchmark_fp16_gemm(M, virtual_K, N, iters=100, alpha=1.0):
    # Generate random fp16 matrices (simulate rounding/clipping as in the int4 kernel)
    A_fp16 = torch.randint(-8, 8, (M, virtual_K), dtype=torch.int8, device='cuda').to(torch.float16)
    B_fp16 = torch.randint(-8, 8, (virtual_K, N), dtype=torch.int8, device='cuda').to(torch.float16)
    
    # Use CUDA events for accurate timing
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    C = None
    for _ in range(iters):
        # We clamp first, then matmul
        A_ref = A_fp16.round().clamp(-8, 7).to(torch.float16)
        B_ref = B_fp16.round().clamp(-8, 7).to(torch.float16)
        C = torch.matmul(A_ref, B_ref) * alpha
    end.record()
    torch.cuda.synchronize()
    
    # Average time in milliseconds
    elapsed_fp16 = start.elapsed_time(end) / iters
    # Cast the output to float32 before summing to avoid overflow issues.
    return elapsed_fp16, C.float().sum().item()

def benchmark_int4_gemm(M, virtual_K, N, iters=100, alpha=1.0):
    # Generate random fp16 matrices
    A_fp16 = torch.randint(-8, 8, (M, virtual_K), dtype=torch.int8, device='cuda').to(torch.float16)
    B_fp16 = torch.randint(-8, 8, (virtual_K, N), dtype=torch.int8, device='cuda').to(torch.float16)
    
    # Warm-up iterations
    for _ in range(10):
        _ = int4_gemm.int4_gemm(A_fp16, B_fp16, virtual_K, alpha=alpha)
    torch.cuda.synchronize()
    
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    
    start.record()
    C = None
    for _ in range(iters):
        C = int4_gemm.int4_gemm(A_fp16, B_fp16, virtual_K, alpha=alpha)
    end.record()
    torch.cuda.synchronize()
    
    elapsed_int4 = start.elapsed_time(end) / iters
    return elapsed_int4, C.sum().item()

if __name__ == "__main__":
    print("Running int4 GEMM correctness tests (with alpha scaling)...")
    test_int4_gemm()
    
    print("\nBenchmarking performance...\n")
    
    # Dimensions (note: virtual_K must be even)
    M, virtual_K, N = 8192, 8192, 8192
    iters = 50
    alpha = 3.14
    # Benchmark the FP16 approach
    fp16_time, fp16_sum = benchmark_fp16_gemm(M, virtual_K, N, iters, alpha=alpha)
    
    # Benchmark the int4 approach (no scaling, alpha=1.0)
    int4_time, int4_sum = benchmark_int4_gemm(M, virtual_K, N, iters, alpha=alpha)
    
    print("FP16 GEMM  (alpha={alpha}):")
    print("  Average time per iteration (ms):", fp16_time)
    print("  Result sum:", fp16_sum)
    print()
    print(f"INT4 GEMM (alpha={alpha}):")
    print("  Average time per iteration (ms):", int4_time)
    print("  Result sum:", int4_sum)
    print()
    
    speedup = fp16_time / int4_time if int4_time > 0 else float('inf')
    print("Speedup of INT4 GEMM over FP16 GEMM: {:.2f}x".format(speedup))
