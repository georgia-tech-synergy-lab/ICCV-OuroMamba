import torch
import time

def format_flops(flops):
    if flops >= 1e12:
        return f"{flops/1e12:.2f} TFLOPS"
    elif flops >= 1e9:
        return f"{flops/1e9:.2f} GFLOPS"
    elif flops >= 1e6:
        return f"{flops/1e6:.2f} MFLOPS"
    else:
        return f"{flops:.2f} FLOPS"

def gemm_test(dtype, size, iterations=100, warmup=10):
    # Use GPU if available, otherwise CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Create random matrices A and B of the given size and type
    A = torch.randn(size, size, device=device, dtype=dtype)
    B = torch.randn(size, size, device=device, dtype=dtype)
    
    # Warm-up iterations (to stabilize any startup overhead)
    for _ in range(warmup):
        _ = torch.matmul(A, B)
        if device.type == "cuda":
            torch.cuda.synchronize()
    
    # Measure execution time over multiple iterations
    if device.type == "cuda":
        # Use CUDA events for accurate timing on GPU
        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)
        
        start_event.record()
        for _ in range(iterations):
            _ = torch.matmul(A, B)
        end_event.record()
        
        # Wait for all operations to finish
        torch.cuda.synchronize()
        total_time_ms = start_event.elapsed_time(end_event)  # in milliseconds
        avg_time = total_time_ms / iterations
    else:
        # Fallback timing for CPU
        start = time.time()
        for _ in range(iterations):
            _ = torch.matmul(A, B)
        end = time.time()
        total_time_ms = (end - start) * 1000  # Convert to milliseconds
        avg_time = total_time_ms / iterations

    return avg_time

if __name__ == "__main__":
    matrix_size = 10240  # Adjust matrix dimensions as needed
    iterations = 30       # Number of iterations for timing

    # Test GEMM performance for FP32
    fp32_time = gemm_test(torch.float32, matrix_size, iterations)
    # Throughput in FLOPS for FP32: 2 * N^3 operations per iteration
    fp32_flops = (2 * (matrix_size ** 3)) / (fp32_time / 1000)
    fp32_throughput = format_flops(fp32_flops)

    # Test GEMM performance for FP16
    fp16_time = gemm_test(torch.float16, matrix_size, iterations)
    fp16_flops = (2 * (matrix_size ** 3)) / (fp16_time / 1000)
    fp16_throughput = format_flops(fp16_flops)

    # Calculate the speedup factor: FP32 time divided by FP16 time
    speedup = fp32_time / fp16_time

    print(f"Average GEMM time for FP32: {fp32_time:.3f} ms, Throughput: {fp32_throughput}")
    print(f"Average GEMM time for FP16: {fp16_time:.3f} ms, Throughput: {fp16_throughput}")
    print(f"Speedup (FP32 / FP16): {speedup:.2f}x")
