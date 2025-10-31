import torch
import numpy as np
import custom_quantization_cuda as torch_extension
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

###########################################
# Parameters and matrix dimensions
###########################################
B = 64
M = 12288       # number of rows in weight (output channels)
K = 3072      # common dimension (channels) — logical number of FP16 channels
N = B * 788    # number of activation columns

weight_scale = 1   # weight quantization scale
act_scale = 1      # activation inlier scale

###########################################
# Generate random FP16 matrices
###########################################
weight_fp16 = torch.randn(M, K, device=device, dtype=torch.half)
act_fp16 = torch.randn(K, N, device=device, dtype=torch.half)

###########################################
# Convert act_fp16 to column-major order
###########################################
# act_fp16 is originally in row-major order.
# Convert to a NumPy array, enforce Fortran (column-major) order, then convert back to a torch tensor.
act_fp16_np = act_fp16.cpu().numpy()
act_fp16_np_col = np.asfortranarray(act_fp16_np)  # now column-major
act_fp16_col = torch.from_numpy(act_fp16_np_col).to(device)

###########################################
# Outlier configuration (none in this test)
###########################################
outlier_percentage = 0.2
n_outlier = max(4, (int(N * outlier_percentage) // 4) * 4)  # need to be divisible by 4
print(f"outlier percentage {outlier_percentage} | outlier channels: {n_outlier}")
outlier_indices = torch.sort(torch.randperm(N)[:n_outlier])[0].to(device)
outlier_scales = (act_scale * 10) * torch.ones(n_outlier, device=device, dtype=torch.float32)

###########################################
# Pack weight matrix using the provided kernel
###########################################
# Each FP16 weight is quantized to a 4-bit value.
# Two 4-bit values are packed into one int8.
pad_align = 16
logical_weight_int8 = K // 2  
padded_K = ((logical_weight_int8 + pad_align - 1) // pad_align) * pad_align

packed_weight = torch.empty((M, padded_K), device=device, dtype=torch.int8)

torch_extension.pack_int4_kernel_fp16_rowmajor(
    weight_fp16,      # FP16 weight matrix [M x K]
    weight_scale,     # quantization scale for weights
    M,                # number of rows
    K,                # number of FP16 columns (logical length)
    padded_K,         # padded physical width in int8 elements
    packed_weight     # output tensor (packed int4 stored as int8)
)

###########################################
# Run the custom GEMM with outlier support using column-major activations.
###########################################
output_cuda = torch.empty((M, N), device=device, dtype=torch.half)
torch_extension.custom_gemm_with_outliers(
    packed_weight,
    act_fp16_col,  # activation matrix in column-major order
    weight_scale,
    act_scale,
    outlier_indices,
    outlier_scales,
    output_cuda
)

# ###########################################
# # Compute reference simulation for quantized GEMM (using row-major activations)
# ###########################################
# # For the reference computation, we use the original act_fp16 (row-major).
# # Quantize weight.
# weight_int = torch.round(weight_fp16.float() / weight_scale).clamp(-7, 7)
# # Adjust negatives as in the CUDA pack (simulate unsigned 4-bit representation).
# weight_int_packed = weight_int.clone()
# weight_int_packed[weight_int < 0] += 16

# # Compute reference GEMM.
# y_ref = torch.zeros((M, N), device=device, dtype=torch.float32)
# inlier_mask = torch.ones(N, dtype=torch.bool, device=device)
# inlier_mask[outlier_indices] = False
# inlier_indices = torch.nonzero(inlier_mask).squeeze(1)

# if inlier_indices.numel() > 0:
#     act_inlier = torch.round(act_fp16.float()[:, inlier_indices] / act_scale).clamp(-7, 7)
#     gemm_inlier = weight_int @ act_inlier  # integer GEMM on raw (unpacked) int values
#     y_ref[:, inlier_indices] = gemm_inlier.float() * (weight_scale * act_scale)

# for i, col in enumerate(outlier_indices):
#     col_idx = int(col.item())
#     scale_out = outlier_scales[i].item()
#     act_outlier = torch.round(act_fp16.float()[:, col_idx] / scale_out).clamp(-127, 127)
#     gemm_outlier = weight_int @ act_outlier.unsqueeze(1)
#     y_ref[:, col_idx] = gemm_outlier.squeeze(1).float() * (weight_scale * scale_out)

# ###########################################
# # Compare results and print statistics
# ###########################################
# y_cuda_np = output_cuda.float().cpu().numpy()
# y_ref_np = y_ref.cpu().numpy()

# stats = {}
# stats["ref_min"] = float(y_ref_np.min())
# stats["ref_max"] = float(y_ref_np.max())
# stats["ref_sum"] = float(y_ref_np.sum())
# stats["cuda_min"] = float(y_cuda_np.min())
# stats["cuda_max"] = float(y_cuda_np.max())
# stats["cuda_sum"] = float(y_cuda_np.sum())

# diff = y_ref_np - y_cuda_np
# abs_diff = np.abs(diff)
# epsilon = 1e-6
# rel_err = abs_diff / (np.abs(y_ref_np) + epsilon)
# stats["mean_abs_error"] = float(abs_diff.mean())
# mask = np.abs(y_ref_np) > 1
# stats["mean_rel_error_large"] = float((rel_err[mask].mean() * 100) if np.any(mask) else 0.0)

# print("\nStatistics comparing quantized GEMM reference with CUDA kernel output:")
# for key, value in stats.items():
#     if "rel_error" in key:
#         print(f"{key}: {value:.6f}%")
#     else:
#         print(f"{key}: {value:.6f}")

###########################################
# Performance Test: Specialized CUDA kernel vs. Native FP16 GEMM
###########################################
if device.type == "cuda":
    num_warmup = 10
    num_iters = 100

    # --- Specialized CUDA kernel performance test ---
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        torch_extension.custom_gemm_with_outliers(
            packed_weight,
            act_fp16_col,
            weight_scale,
            act_scale,
            outlier_indices,
            outlier_scales,
            output_cuda
        )
    torch.cuda.synchronize()

    start_special = torch.cuda.Event(enable_timing=True)
    end_special = torch.cuda.Event(enable_timing=True)
    start_special.record()
    for _ in range(num_iters):
        torch_extension.custom_gemm_with_outliers(
            packed_weight,
            act_fp16_col,
            weight_scale,
            act_scale,
            outlier_indices,
            outlier_scales,
            output_cuda
        )
        torch.cuda.synchronize()
    end_special.record()
    torch.cuda.synchronize()
    specialized_time = start_special.elapsed_time(end_special)  # in ms
    avg_specialized_latency = specialized_time / num_iters

    # --- Native FP16 GEMM performance test ---
    torch.cuda.synchronize()
    for _ in range(num_warmup):
        _ = torch.matmul(weight_fp16, act_fp16)
    torch.cuda.synchronize()

    start_native = torch.cuda.Event(enable_timing=True)
    end_native = torch.cuda.Event(enable_timing=True)
    start_native.record()
    for _ in range(num_iters):
        _ = torch.matmul(weight_fp16, act_fp16)
        torch.cuda.synchronize()
    end_native.record()
    torch.cuda.synchronize()
    native_time = start_native.elapsed_time(end_native)  # in ms
    avg_native_latency = native_time / num_iters

    speedup = avg_native_latency / avg_specialized_latency

    print("\n=== Performance Test Results ===")
    print(f"Specialized CUDA kernel avg latency: {avg_specialized_latency:.3f} ms")
    print(f"Native FP16 GEMM avg latency: {avg_native_latency:.3f} ms")
    print(f"Speedup (native / specialized): {speedup:.3f}x")
else:
    print("CUDA device not available. Performance test requires CUDA.")
