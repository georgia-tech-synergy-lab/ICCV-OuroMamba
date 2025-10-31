import time
import torch
import numpy as np

# Adjust the following import to your extension module name.
import custom_quantization_cuda as my_extension

def quantize_values(vals, scales):
    """
    CPU-side quantization matching our kernel for outliers: 
      q = round(val / scale) clamped to int8.
    """
    q = torch.round(vals / scales)
    q.clamp_(min=-127, max=127)
    return q.to(torch.int8)

def quantize_weight_values(vals, scale):
    # Mimic roundf: round half away from zero.
    ratio = vals / scale
    q = torch.where(ratio >= 0, torch.floor(ratio + 0.5), torch.ceil(ratio - 0.5))
    q.clamp_(min=-7, max=7)
    return q.to(torch.int8)


def coo_to_sorted_triplets(coo_tensor):
    """
    Given a sparse COO tensor, return sorted (col, row, value) triplets.
    We sort primarily by column then row.
    """
    # COO indices: shape [2, nnz]
    indices = coo_tensor._indices()  # first row: row, second row: col
    values  = coo_tensor._values()
    # For sorting by (col, row), build key = (col << 16) + row (enough for our sizes)
    cols = indices[1].to(torch.int64)
    rows = indices[0].to(torch.int64)
    keys = (cols << 16) + rows
    idx_sorted = torch.argsort(keys)
    return rows[idx_sorted].cpu().numpy(), cols[idx_sorted].cpu().numpy(), values[idx_sorted].cpu().numpy()

def csc_triplets_from_native(sparse_csc):
    """
    Extract sorted (col, row, value) triplets from a native CSC tensor.
    A native CSC tensor has attributes:
       csc_col: tensor of shape [ncols+1]
       row_indices: tensor of shape [nnz]
       values: tensor of shape [nnz]
    We create the COO indices (row, col) from that.
    """
    csc_col = sparse_csc.ccol_indices().cpu()
    row_indices = sparse_csc.row_indices().cpu()
    values = sparse_csc.values().cpu()
    cols_list = []
    rows_list = []
    vals_list = []
    ncols = csc_col.numel() - 1
    for col in range(ncols):
        start = csc_col[col].item()
        end   = csc_col[col+1].item()
        if end > start:
            num = end - start
            rows_list.append(row_indices[start:end])
            cols_list.append(torch.full((num,), col, dtype=torch.int64))
            vals_list.append(values[start:end])
    if rows_list:
        rows = torch.cat(rows_list, dim=0)
        cols = torch.cat(cols_list, dim=0)
        vals = torch.cat(vals_list, dim=0)
    else:
        rows = torch.tensor([], dtype=torch.int64)
        cols = torch.tensor([], dtype=torch.int64)
        vals = torch.tensor([], dtype=values.dtype)
    # Now sort by (col, row)
    keys = (cols << 16) + rows
    idx_sorted = torch.argsort(keys)
    return rows[idx_sorted].cpu().numpy(), cols[idx_sorted].cpu().numpy(), vals[idx_sorted].cpu().numpy()

def test_quantize_and_convert_outliers_to_csc():
    device = "cuda"
    M = N = 1024
    total_elements = M * N

    # Settings: 5% columns get outliers, and overall ~10% outliers.
    frac_outlier_cols = 0.05    # ~5% of columns will have outliers
    frac_total_outliers = 0.10  # ~10% of total elements are outliers

    num_outlier_cols = int(N * frac_outlier_cols)
    nnz_target = int(total_elements * frac_total_outliers)
    outliers_per_col = nnz_target // num_outlier_cols

    # Ensure unique row indices by limiting outliers_per_col to at most M.
    actual_outliers_per_col = min(outliers_per_col, M)

    print(f"Matrix size: {M}x{N}")
    print(f"Total elements: {total_elements:,}")
    print(f"Target nnz (outliers): {nnz_target:,}")
    print(f"Outlier columns: {num_outlier_cols} (~{frac_outlier_cols*100:.1f}%)")
    print(f"Outliers per selected column (unique): {actual_outliers_per_col}")

    # Choose a subset of columns to contain outliers.
    col_selection = torch.randperm(N)[:num_outlier_cols]
    outlier_rows_list = []
    outlier_cols_list = []
    for c in col_selection:
        # Sample unique row indices without replacement.
        rows = torch.randperm(M)[:actual_outliers_per_col]
        outlier_rows_list.append(rows)
        outlier_cols_list.append(torch.full((actual_outliers_per_col,), c.item(), dtype=torch.int32))

    outlier_rows = torch.cat(outlier_rows_list, dim=0).to(device)
    outlier_cols = torch.cat(outlier_cols_list, dim=0).to(device)
    nnz = outlier_rows.numel()
    print(f"Actual nnz (unique): {nnz:,}")

    # Create random outlier FP16 values and per-outlier scales.
    outlier_vals = (torch.randn(nnz, device=device, dtype=torch.float16) * 10.0)
    outlier_scales = torch.rand(nnz, device=device, dtype=torch.float16) + 0.01

    # Build outlier indices in COO format: shape [2, nnz] (row, col)
    outlier_indices = torch.stack([outlier_rows.long(), outlier_cols.long()], dim=0)

    # -------------------------------
    # Create native CSC from sparse COO via PyTorch
    # -------------------------------
    # Quantize the outlier values on CPU using the same logic.
    outlier_vals_cpu = outlier_vals.float().cpu()
    outlier_scales_cpu = outlier_scales.float().cpu()
    quantized_cpu = quantize_values(outlier_vals_cpu, outlier_scales_cpu)
    # Create a COO tensor (values are quantized int8)
    coo_indices = outlier_indices.cpu()
    sparse_coo = torch.sparse_coo_tensor(coo_indices, quantized_cpu, size=(M, N), dtype=torch.int8).to("cuda")
    # Use native conversion to CSC.
    t0 = time.time()
    native_csc = sparse_coo.to_sparse_csc()
    torch.cuda.synchronize()
    t1 = time.time()
    native_time = t1 - t0

    native_rows, native_cols, native_vals = csc_triplets_from_native(native_csc)

    # -------------------------------
    # Call our custom CUDA extension (updated to also permute scales)
    # -------------------------------
    # Pre-allocate output tensors (on GPU)
    csc_values_gpu = torch.empty(nnz, dtype=torch.int8, device=device)
    csc_row_indices_gpu = torch.empty(nnz, dtype=torch.int32, device=device)
    csc_scale_gpu = torch.empty(nnz, dtype=torch.float16, device=device)  # new output tensor for scales
    csc_col_ptr_gpu = torch.empty(N + 1, dtype=torch.int32, device=device)

    # Warmup call
    my_extension.quantize_and_convert_outliers_to_csc(
        outlier_vals,
        outlier_scales,
        outlier_indices,
        nnz,
        M,
        N,
        256,
        csc_values_gpu,
        csc_row_indices_gpu,
        csc_scale_gpu,      
        csc_col_ptr_gpu
    )
    torch.cuda.synchronize()

    # Time the custom conversion over several iterations.
    num_iters = 10
    t_start = time.time()
    for _ in range(num_iters):
        my_extension.quantize_and_convert_outliers_to_csc(
            outlier_vals,
            outlier_scales,
            outlier_indices,
            nnz,
            M,
            N,
            256,
            csc_values_gpu,
            csc_row_indices_gpu,
            csc_scale_gpu,
            csc_col_ptr_gpu
        )
    torch.cuda.synchronize()
    t_end = time.time()
    custom_avg_time = (t_end - t_start) / num_iters

    # Extract CSC triplets from our custom conversion.
    csc_col_ptr_cpu = csc_col_ptr_gpu.cpu()
    csc_vals_cpu = csc_values_gpu.cpu()
    csc_rows_cpu = csc_row_indices_gpu.cpu()
    csc_scales_cpu = csc_scale_gpu.cpu()

    custom_rows_list = []
    custom_cols_list = []
    custom_vals_list = []
    custom_scales_list = []
    for col in range(N):
        start = csc_col_ptr_cpu[col].item()
        end = csc_col_ptr_cpu[col+1].item()
        if end > start:
            num = end - start
            custom_rows_list.append(csc_rows_cpu[start:end])
            custom_cols_list.append(torch.full((num,), col, dtype=torch.int32))
            custom_vals_list.append(csc_vals_cpu[start:end])
            custom_scales_list.append(csc_scales_cpu[start:end])
    if custom_rows_list:
        custom_rows = torch.cat(custom_rows_list, dim=0)
        custom_cols = torch.cat(custom_cols_list, dim=0)
        custom_vals = torch.cat(custom_vals_list, dim=0)
        custom_scales = torch.cat(custom_scales_list, dim=0)
    else:
        custom_rows = torch.tensor([], dtype=torch.int32)
        custom_cols = torch.tensor([], dtype=torch.int32)
        custom_vals = torch.tensor([], dtype=torch.int8)
        custom_scales = torch.tensor([], dtype=torch.float16)

    # Now sort both native and custom triplets by (col, row) for comparison.
    def sort_triplets(rows, cols, vals):
        rows = rows.to(torch.int64)
        cols = cols.to(torch.int64)
        keys = (cols << 16) + rows
        idx = torch.argsort(keys)
        return rows[idx].numpy(), cols[idx].numpy(), vals[idx].numpy()

    native_sorted = sort_triplets(torch.tensor(native_rows), torch.tensor(native_cols), torch.tensor(native_vals))
    custom_sorted = sort_triplets(custom_rows, custom_cols, custom_vals)

    # Compare quantized values, rows, and columns.
    same_shape = (native_sorted[0].shape == custom_sorted[0].shape)
    same_rows = np.array_equal(native_sorted[0], custom_sorted[0])
    same_cols = np.array_equal(native_sorted[1], custom_sorted[1])
    same_vals = np.array_equal(native_sorted[2], custom_sorted[2])
    
    if same_shape and same_rows and same_cols and same_vals:
        print("SUCCESS: Custom CSC conversion (quantized values) matches PyTorch's native CSC conversion.")
    else:
        print("ERROR: Mismatch between custom and native CSC conversion for quantized values.")
        print(f"Rows equal: {same_rows}, Cols equal: {same_cols}, Vals equal: {same_vals}")
    
    # -------------------------------
    # Additional correctness: check dequantization
    # -------------------------------
    # Dequantize the custom CSC values by multiplying by the permuted scales.
    custom_dequant = custom_vals.float() * custom_scales.float()
    # Now, compute the CPU dequantized values from the original outlier tensors.
    quantized_cpu = quantize_values(outlier_vals_cpu, outlier_scales_cpu)
    cpu_dequant = quantized_cpu.float() * outlier_scales_cpu
    # Create a COO tensor for the CPU dequantized values.
    cpu_coo = torch.sparse_coo_tensor(coo_indices, cpu_dequant, size=(M, N), dtype=torch.float32)
    _, _, cpu_sorted_dequant = coo_to_sorted_triplets(cpu_coo)

    # Sort custom dequantized triplets similarly.
    _, _, custom_sorted_dequant = sort_triplets(custom_rows, custom_cols, custom_dequant)

    same_dequant = np.allclose(cpu_sorted_dequant, custom_sorted_dequant, atol=1e-3)
    if same_dequant:
        print("SUCCESS: Dequantization using permuted scales is correct.")
    else:
        print("ERROR: Dequantized values do not match the expected results.")
        diff = np.abs(np.array(cpu_sorted_dequant) - np.array(custom_sorted_dequant))
        print("Mean difference: {:.4f}, Max difference: {:.4f}".format(diff.mean(), diff.max()))

    print(f"\nNative PyTorch CSC conversion average time: {native_time*1000:.3f} ms")
    print(f"Custom CSC conversion average time: {custom_avg_time*1000:.3f} ms")
    print(f"Speedup: {native_time / custom_avg_time:.2f}x (native / custom)")

def test_convert_dense_quantized_weight_to_csr():
    device = "cuda"
    M = 128
    # Use an even number of logical columns so that each row packs neatly.
    logical_cols = 1024  # logical FP16 width
    padded_virtualK = logical_cols // 2  # physical width in int8 elements
    print(f"Weight matrix size: {M} x {logical_cols}")

    # Create random FP16 weight matrix.
    # Choose a distribution such that quantized values fall into [-7, 7]. For example, use a moderate scale.
    scale = 0.5  # scalar scale for weight quantization
    weight_fp16 = torch.randn(M, logical_cols, device=device, dtype=torch.float16) * 3.0

    # Pack the FP16 weight matrix into int8 packed representation (each int8 packs two int4 values).
    dense_quantized_weight = torch.empty((M, padded_virtualK), dtype=torch.int8, device=device)
    my_extension.pack_int4_kernel_fp16_rowmajor(
        weight_fp16, scale, M, logical_cols, padded_virtualK, dense_quantized_weight
    )
    torch.cuda.synchronize()

    # Choose a set of active logical column indices (in the range [0, logical_cols)).
    # For this test, we choose a random subset and sort them.
    num_active = 256
    active_k = torch.sort(torch.randperm(logical_cols)[:num_active])[0].to(torch.int32).to(device)

    # Pre-allocate output tensors for CSR conversion.
    total_nnz = M * num_active
    csr_values = torch.empty(total_nnz, dtype=torch.int8, device=device)
    csr_col_indices = torch.empty(total_nnz, dtype=torch.int32, device=device)
    csr_row_ptr = torch.empty(M + 1, dtype=torch.int32, device=device)

    # Call the weight-to-CSR conversion kernel.
    my_extension.convert_dense_quantized_weight_to_csr(
        dense_quantized_weight,
        active_k,
        csr_values,
        csr_col_indices,
        csr_row_ptr
    )
    torch.cuda.synchronize()

    # Compute the expected quantized weight matrix on CPU.
    weight_cpu = weight_fp16.cpu().float()
    expected_q = quantize_weight_values(weight_cpu, scale)  # shape: [M, logical_cols]

    # Build expected CSR arrays by selecting the active columns for each row.
    expected_vals_list = []
    expected_cols_list = []
    expected_row_ptr = [0]
    for m in range(M):
        # active_k is in the range [0, logical_cols) which is the same as the logical index.
        row_vals = expected_q[m, active_k.cpu()]
        expected_vals_list.append(row_vals)
        expected_cols_list.append(active_k.cpu())
        expected_row_ptr.append(expected_row_ptr[-1] + len(active_k))
    expected_vals = torch.cat(expected_vals_list, dim=0).numpy()
    expected_cols = torch.cat(expected_cols_list, dim=0).numpy()
    expected_row_ptr = np.array(expected_row_ptr, dtype=np.int32)

    # Retrieve CSR results from GPU.
    csr_vals_cpu = csr_values.cpu().numpy()
    csr_cols_cpu = csr_col_indices.cpu().numpy()
    csr_row_ptr_cpu = csr_row_ptr.cpu().numpy()

    same_vals = np.array_equal(expected_vals, csr_vals_cpu)
    same_cols = np.array_equal(expected_cols, csr_cols_cpu)
    same_row_ptr = np.array_equal(expected_row_ptr, csr_row_ptr_cpu)
    if same_vals and same_cols and same_row_ptr:
        print("SUCCESS: Weight to CSR conversion matches expected quantized values.")
    else:
        print("ERROR: Mismatch in weight to CSR conversion.")
        if not same_vals:
            print("Mismatch in quantized values.")
        if not same_cols:
            print("Mismatch in column indices.")
        if not same_row_ptr:
            print("Mismatch in row pointer.")

if __name__ == "__main__":
    print("Running outlier CSC conversion test...")
    test_quantize_and_convert_outliers_to_csc()
    print("\nRunning weight to CSR conversion test...")
    test_convert_dense_quantized_weight_to_csr()
