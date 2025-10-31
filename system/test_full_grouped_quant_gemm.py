import torch
import time
import numpy as np

# Assume our CUDA extension is built and imported as my_cuda_ext.
import custom_quantization_cuda as my_cuda_ext

device = torch.device("cuda")

# --------------------------------------------------------------
# (1) Define matrix dimensions and scales
# --------------------------------------------------------------
M, K, N = 8192, 4096, 2048
#M, K, N = 512*197,8192,8192
weight_scale = 1
act_scale = 1

print(f"[LOG] Running test with M={M}, K={K}, N={N} on GPU.")

# --------------------------------------------------------------
# (2) Create random weight and activation in FP16
# --------------------------------------------------------------
# Generate random integers in [-7, 7] then convert to FP16.
weight_fp16 = torch.randint(-7, 8, (M, K), dtype=torch.int8, device=device).to(torch.float16)
act_fp = torch.randint(-7, 8, (K, N), dtype=torch.int8, device=device).to(torch.float16)

print(f"[LOG] weight_fp16 stats: min={weight_fp16.min().item()}, max={weight_fp16.max().item()}, mean={weight_fp16.float().mean().item()}")
print(f"[LOG] act_fp stats: min={act_fp.min().item()}, max={act_fp.max().item()}, mean={act_fp.float().mean().item()}")

total_elements = K * N


frac_total_outliers = 0.0001
frac_outlier_cols = 0.01
num_outlier_cols = int(N * frac_outlier_cols)
nnz_target = int(total_elements * frac_total_outliers)
# Choose a fixed number per outlier column
outliers_per_col = max(1, nnz_target // max(num_outlier_cols, 1))
outliers_per_col = min(outliers_per_col, K)

print(f"[LOG] Total elements: {total_elements}")
print(f"[LOG] Outlier columns: {num_outlier_cols} (of N={N})")
print(f"[LOG] Desired nnz: {nnz_target}, outliers_per_col: {outliers_per_col}")

if num_outlier_cols > 0:
    col_selection = torch.randperm(N, device=device)[:num_outlier_cols]
else:
    col_selection = torch.tensor([], device=device, dtype=torch.int64)

outlier_rows_list = []
outlier_cols_list = []
for c in col_selection:
    rows = torch.randperm(K, device=device)[:outliers_per_col]
    outlier_rows_list.append(rows)
    outlier_cols_list.append(
        torch.full((outliers_per_col,), c.item(), dtype=torch.int64, device=device)
    )

if len(outlier_rows_list) > 0:
    outlier_rows = torch.cat(outlier_rows_list, dim=0)
    outlier_cols = torch.cat(outlier_cols_list, dim=0)
else:
    outlier_rows = torch.tensor([], device=device, dtype=torch.int64)
    outlier_cols = torch.tensor([], device=device, dtype=torch.int64)


nnz = outlier_rows.numel()
print(f"[LOG] Actual nnz (number of outliers): {nnz}")
if nnz > 0:
    print(f"[LOG] Sample outlier rows: {outlier_rows[:min(10, nnz)].cpu().tolist()}")
    print(f"[LOG] Sample outlier cols: {outlier_cols[:min(10, nnz)].cpu().tolist()}")

# Outlier values and per-outlier scale
act_outlier_values = act_fp[outlier_rows, outlier_cols]
act_outlier_scale = 0.05 + 0.20 * torch.rand((nnz,), device=device, dtype=torch.float16)
if nnz > 0:
    print(f"[LOG] Sample outlier activation values: {act_outlier_values[:min(10, nnz)]}")
    print(f"[LOG] Sample outlier scales: {act_outlier_scale[:min(10, nnz)]}")

# Inlier activation: zero out outlier positions
act_fp_inlier = act_fp.clone()
if nnz > 0:
    act_fp_inlier[outlier_rows, outlier_cols] = 0.0

# --------------------------------------------------------------
# (4) Quantize activation outliers and convert COO to CSC
# --------------------------------------------------------------
# Prepare COO indices: a tensor of shape [2, nnz] where row 0: row indices, row 1: col indices.
if nnz > 0:
    outlier_indices = torch.stack([outlier_rows, outlier_cols], dim=0)
    # Allocate CSC outputs:
    csc_values = torch.empty((nnz,), dtype=torch.int8, device=device)
    csc_row_indices = torch.empty((nnz,), dtype=torch.int32, device=device)
    csc_scale = torch.empty((nnz,), dtype=torch.float16, device=device)
    csc_col_ptr = torch.empty((N + 1,), dtype=torch.int32, device=device)
    # Choose quantization range parameter, e.g.:
    n_lva_o = 254
    my_cuda_ext.quantize_and_convert_outliers_to_csc(
        act_outlier_values,   # act_outlier: FP16 [nnz]
        act_outlier_scale,    # act_outlier_scale: FP16 [nnz]
        outlier_indices,      # COO indices [2, nnz]
        nnz,                  # nnz
        K,                    # num_rows = K (activation rows)
        N,                    # num_cols = N (activation columns)
        n_lva_o,
        csc_values,
        csc_row_indices,
        csc_scale,
        csc_col_ptr
    )
else:
    # Create dummy CSC tensors if no outliers.
    csc_values = torch.empty((0,), dtype=torch.int8, device=device)
    csc_row_indices = torch.empty((0,), dtype=torch.int32, device=device)
    csc_scale = torch.empty((0,), dtype=torch.float16, device=device)
    csc_col_ptr = torch.empty((N + 1,), dtype=torch.int32, device=device)

# --------------------------------------------------------------
# (5) Quantize weight and convert to CSR
# --------------------------------------------------------------
# For weight quantization, we pack the FP16 weight into int4 format.
pad_align = 16
# padded_virtualK is computed similarly as in fused function.
padded_virtualK = ((K + pad_align - 1) // pad_align) * pad_align

# Allocate tensor for the packed weight; shape [M, padded_virtualK] as int8.
packed_weight_int8 = torch.empty((M, padded_virtualK), dtype=torch.int8, device=device)
my_cuda_ext.pack_int4_kernel_fp16_rowmajor(
    weight_fp16,  # FP16 weight [M, K]
    weight_scale, # scale
    M,
    K,
    padded_virtualK,
    packed_weight_int8
)

# Next, convert the dense quantized weight (packed_weight_int8) to CSR.
# For the conversion we need to supply "active_k" indices.
# Here we assume that only the first K logical columns are active.
active_k = torch.arange(K, device=device, dtype=torch.int32)
# Allocate CSR buffers:
csr_values = torch.empty((M * K,), dtype=torch.int8, device=device)
csr_col_indices = torch.empty((M * K,), dtype=torch.int32, device=device)
csr_row_ptr = torch.empty((M + 1,), dtype=torch.int32, device=device)
my_cuda_ext.convert_dense_quantized_weight_to_csr(
    packed_weight_int8,  # dense_quantized_weight: [M, padded_virtualK]
    active_k,            # active_k: shape [K]
    csr_values,
    csr_col_indices,
    csr_row_ptr
)

# --------------------------------------------------------------
# (6) Run fused GEMM (dense int4 GEMM plus outlier spGEMM)
# --------------------------------------------------------------
fused_output = my_cuda_ext.fused_quantized_gemm_packedW_cuda(
    packed_weight_int8,  
    weight_scale,        
    M, K, N,
    act_fp_inlier,   # inlier activation matrix (with outlier positions zeroed)
    act_scale,
    # Weight CSR parameters:
    csr_values,
    csr_col_indices,
    csr_row_ptr,
    # Activation outlier CSC parameters:
    csc_values,
    csc_row_indices,
    csc_scale,
    csc_col_ptr
)

# --------------------------------------------------------------
# (7) Compute reference result using simple quantize-dequantize (Q+D) and matmul.
# --------------------------------------------------------------
# Quantize-dequantize the weight.
weight_q = (weight_fp16 / weight_scale).round().clamp(-7, 7)
weight_fp16_qd = (weight_q * weight_scale).to(torch.float16)

# Quantize-dequantize the inlier activation.
act_inlier_q = (act_fp_inlier / act_scale).round().clamp(-7, 7)
act_inlier_deq = (act_inlier_q * act_scale).to(torch.float16)
act_fp16_qd = act_inlier_deq.clone()

# Now process outlier positions.
if nnz > 0:
    q_outliers = (act_outlier_values / act_outlier_scale).round().clamp(-127, 127)
    outlier_deq = q_outliers * act_outlier_scale
    act_fp16_qd[outlier_rows, outlier_cols] = outlier_deq

# Reference GEMM.
C_ref_qd = torch.matmul(weight_fp16_qd, act_fp16_qd)
print(f"[LOG] Reference Q+D output stats: min={C_ref_qd.min().item()}, max={C_ref_qd.max().item()}, mean={C_ref_qd.float().mean().item()}")

# --------------------------------------------------------------
# (8) Compare results and report error metrics
# --------------------------------------------------------------
# Convert fused output and reference to float32 for comparison.
C_fused = fused_output.to(torch.float32)
C_ref = C_ref_qd.to(torch.float32)
print(f"[LOG] CUDA Q+D output stats: min={C_fused.min().item()}, max={C_fused.max().item()}, mean={C_fused.float().mean().item()}")

# Compute absolute error and relative error (avoid division by zero).
abs_error = torch.abs(C_fused - C_ref)
eps = 1e-6
rel_error = abs_error / (torch.abs(C_ref) + eps)

mean_abs_error = abs_error.mean().item()
max_abs_error = abs_error.max().item()
mean_rel_error = rel_error.mean().item()
max_rel_error = rel_error.max().item()

print(f"[DEBUG] Mean absolute error: {mean_abs_error}")
print(f"[DEBUG] Mean relative error: {mean_rel_error}")

# Check if errors are within acceptable threshold.
rel_threshold = 0.1
if mean_rel_error > rel_threshold:
    print("[ERROR] Fused GEMM output does not match the reference!")
else:
    print("[LOG] Fused GEMM output matches the reference within tolerance.")

# --------------------------------------------------------------
# (9) Performance Benchmarking
# --------------------------------------------------------------
num_iters = 100

# Warm up fused GEMM.
for _ in range(10):
    _ = my_cuda_ext.fused_quantized_gemm_packedW_cuda(
        packed_weight_int8,
        weight_scale,
        M, K, N,
        act_fp_inlier,
        act_scale,
        csr_values,
        csr_col_indices,
        csr_row_ptr,
        csc_values,
        csc_row_indices,
        csc_scale,
        csc_col_ptr
    )
torch.cuda.synchronize()

# Benchmark fused GEMM.
start_time = time.time()
for _ in range(num_iters):
    _ = my_cuda_ext.fused_quantized_gemm_packedW_cuda(
        packed_weight_int8,
        weight_scale,
        M, K, N,
        act_fp_inlier,
        act_scale,
        csr_values,
        csr_col_indices,
        csr_row_ptr,
        csc_values,
        csc_row_indices,
        csc_scale,
        csc_col_ptr
    )
torch.cuda.synchronize()
fused_time = (time.time() - start_time) / num_iters
print(f"[PERF] Fused GEMM average time per iteration: {fused_time * 1000:.3f} ms")

# Benchmark standard FP16 GEMM using torch.matmul.
# Note: We use the original FP16 weight and activation for this comparison.
for _ in range(10):
    _ = torch.matmul(weight_fp16, act_fp)
torch.cuda.synchronize()

start_time = time.time()
for _ in range(num_iters):
    _ = torch.matmul(weight_fp16, act_fp)
torch.cuda.synchronize()
fp16_time = (time.time() - start_time) / num_iters
print(f"[PERF] Standard FP16 GEMM average time per iteration: {fp16_time * 1000:.3f} ms")
