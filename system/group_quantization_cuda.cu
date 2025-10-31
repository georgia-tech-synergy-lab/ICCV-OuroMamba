#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <vector>
#include <stdio.h>
#include <inttypes.h>   // for PRIuPTR

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/half.h"
// paralle prefix scan for COO -> CSC transformation
#include <thrust/device_vector.h>
#include <thrust/scan.h>
#include <thrust/functional.h>
#include <thrust/sequence.h>
#include <thrust/copy.h>
#include <thrust/remove.h>
// ------------------------------------------------------
// Quantize outliers & build per-column histogram
// ------------------------------------------------------
// Each thread does:
//   1) quantize = round( outlier_value / outlier_scale ), clamp to int8 range
//   2) store the quantized result in a temp array
//   3) atomicAdd( &col_counts[col], 1 )
//
// col_counts[] will be turned into csc_col_ptr by an exclusive-scan later.
__global__ void quantize_outlier_build_histogram_kernel(
    const half* __restrict__ act_outlier,        // [nnz] FP16
    const half* __restrict__ act_outlier_scale,  // [nnz] FP16
    const long* __restrict__ outlier_indices,    // [2*nnz] int64 (COO: row,col)
    int nnz,
    int8_t* __restrict__ temp_quantized,         // [nnz] int8, intermediate
    int*    __restrict__ col_counts,             // [num_cols], output histogram
    int     num_cols,
    int     n_lva_o                               // Quantization range parameter
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    // Compute dynamic quantization range from n_lva_o
    int o_qmax = n_lva_o / 2 - 1;
    int o_qmin = -o_qmax;

    // 1) Quantize
    float val_fp   = __half2float(act_outlier[i]);
    float scale_fp = __half2float(act_outlier_scale[i]);
    float scaled   = val_fp / (scale_fp);
    int q = __float2int_rn(scaled);

    // Clamp to computed range instead of hardcoded [-127, 127]
    q = max(o_qmin, min(q, o_qmax));
    temp_quantized[i] = static_cast<int8_t>(q);

    // 2) Figure out which column
    long col64 = outlier_indices[nnz + i];  
    int col    = static_cast<int>(col64);

    // 3) Bump the histogram for that column
    if (col < num_cols) {
      atomicAdd(&col_counts[col], 1);
    }
}

// ------------------------------------------------------
// Fill the CSC and outlier scale arrays
// ------------------------------------------------------
// We assume col_counts[] has been turned into csc_col_ptr[] by an exclusive scan,
// but we STILL need a second usage of col_counts[] as scratch counters so each
// column knows how many we have placed so far.  So we keep a copy of csc_col_ptr[]
// in "col_ptr" and re-use col_counts[] as counters.
//
// Steps per thread i:
//   col = outlier_indices(1, i)
//   offset = atomicAdd(&col_counts[col], 1) + col_ptr[col];
//   csc_values[offset]      = temp_quantized[i]
//   csc_row_indices[offset] = row of i
//   ccsc_scale[offset] = scale of i
//
// csc_col_ptr is length (num_cols + 1).
__global__ void fill_csc_arrays_kernel(
    const int8_t* __restrict__ temp_quantized, // [nnz]
    const long*   __restrict__ outlier_indices,  // [2*nnz]
    int nnz,
    int* __restrict__ col_counts,              // [num_cols] used as counters; zeroed before launch
    const int* __restrict__ csc_col_ptr,         // [num_cols+1], prefix sums
    int8_t* __restrict__ csc_values,             // [nnz]
    int* __restrict__ csc_row_indices,           // [nnz]
    half* __restrict__ csc_scale,                // [nnz] output scale array in CSC order
    const half* __restrict__ act_outlier_scale,  // [nnz] original scale values (COO order)
    int num_cols
)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nnz) return;

    // Determine the column using the second row of outlier_indices.
    long col64 = outlier_indices[nnz + i];
    int col = static_cast<int>(col64);

    // Determine the row using the first row of outlier_indices.
    long row64 = outlier_indices[i];
    int row = static_cast<int>(row64);

    // Compute the destination offset within this column.
    int my_offset_in_col = atomicAdd(&col_counts[col], 1);
    int offset = csc_col_ptr[col] + my_offset_in_col;

    // Write quantized value and row index as before.
    csc_values[offset]      = temp_quantized[i];
    csc_row_indices[offset] = row;
    
    // Also permute the scale value into CSC order.
    csc_scale[offset] = act_outlier_scale[i];
}

// ------------------------------------------------------
// quantize_and_convert_outliers_to_csc_host
// ------------------------------------------------------
// Inputs:
//   act_outlier:        FP16 [nnz]
//   act_outlier_scale:  FP16 [nnz]
//   outlier_indices:    int64 [2, nnz] (COO format: row,col)
//   nnz:                number of outliers
//   num_rows, num_cols: dimensions of the full matrix
//
// Outputs:
//   csc_values:      int8 [nnz]
//   csc_row_indices: int  [nnz]
//   csc_col_ptr:     int  [num_cols+1]
//
// This function does:
//   (1) Launch kernel to quantize outliers and build a per-col histogram
//   (2) Exclusive scan of col_counts => csc_col_ptr
//   (3) Launch kernel to fill the CSC arrays
// ------------------------------------------------------
void quantize_and_convert_outliers_to_csc_host(
    torch::Tensor act_outlier,       // [nnz], FP16
    torch::Tensor act_outlier_scale, // [nnz], FP16
    torch::Tensor outlier_indices,   // [2, nnz], int64
    int nnz,
    int num_rows,
    int num_cols,
    int n_lva_o,
    // Outputs
    torch::Tensor csc_values,        // [nnz],  int8
    torch::Tensor csc_row_indices,   // [nnz],  int
    torch::Tensor csc_scale,         // [nnz],  FP16, permuted scales in CSC order
    torch::Tensor csc_col_ptr       // [num_cols+1], int
) {
    // Safety checks
    TORCH_CHECK(act_outlier.dim() == 1, "act_outlier must be 1D");
    TORCH_CHECK(act_outlier_scale.dim() == 1, "act_outlier_scale must be 1D");
    TORCH_CHECK(outlier_indices.dim() == 2, "outlier_indices must be 2D of shape [2, nnz]");
    TORCH_CHECK(csc_values.size(0) == nnz, "csc_values must have length = nnz");
    TORCH_CHECK(csc_row_indices.size(0) == nnz, "csc_row_indices must have length = nnz");
    TORCH_CHECK(csc_col_ptr.size(0) == (num_cols + 1), "csc_col_ptr must have length = num_cols+1");

    // Create an integer buffer "col_counts" on device to hold histogram counts per column
    auto opts_int = torch::dtype(torch::kInt32).device(act_outlier.device());
    auto col_counts = torch::zeros({num_cols}, opts_int);

    // Create a temp buffer to hold the quantized outlier values
    auto temp_quantized = torch::empty({nnz}, torch::dtype(torch::kInt8).device(act_outlier.device()));

    // 1) Kernel: quantize & build histogram
    {
      int block_size = 256;
      int grid_size = (nnz + block_size - 1) / block_size;
      quantize_outlier_build_histogram_kernel<<<grid_size, block_size>>>(
          reinterpret_cast<const half*>(act_outlier.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(act_outlier_scale.data_ptr<at::Half>()),
          reinterpret_cast<const long*>(outlier_indices.data_ptr<int64_t>()),
          nnz,
          temp_quantized.data_ptr<int8_t>(),
          col_counts.data_ptr<int>(),
          num_cols,
          n_lva_o
      );
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("quantize_outlier_build_histogram_kernel failed: %s\n",
                 cudaGetErrorString(err));
      }
    }

    // 2) Exclusive scan (prefix sum) on col_counts => csc_col_ptr
    {
        thrust::device_ptr<int> col_counts_ptr((int*)col_counts.data_ptr<int>());
        thrust::device_ptr<int> csc_col_ptr_ptr((int*)csc_col_ptr.data_ptr<int>());
    
        // Copy col_counts -> csc_col_ptr for the scan
        //   We'll scan the first num_cols elements and store result in csc_col_ptr.
        //   Then we want csc_col_ptr[num_cols] to be the total NNZ.
    
        //  A) copy col_counts into the first num_cols of csc_col_ptr
        //  Must use raw pointer to pass into cudaMemcpy
        int*       csc_col_ptr_raw  = thrust::raw_pointer_cast(csc_col_ptr_ptr);
        const int* col_counts_raw   = thrust::raw_pointer_cast(col_counts_ptr);
        cudaMemcpy(
            csc_col_ptr_raw,
            col_counts_raw,
            num_cols * sizeof(int),
            cudaMemcpyDeviceToDevice
        );
    
        //  B) Perform exclusive scan of those num_cols elements
        thrust::exclusive_scan(
            csc_col_ptr_ptr,               // start
            csc_col_ptr_ptr + num_cols,    // end
            csc_col_ptr_ptr                // destination
        );
    
        //  C) Read the last two device values back to host
        //     to compute total_nnz_in_outliers = csc_col_ptr[num_cols - 1] + col_counts[num_cols - 1].
        int last_col_prefix     = 0;
        int last_col_hist_count = 0;
    
        // read csc_col_ptr[num_cols - 1]
        cudaMemcpy(
            &last_col_prefix,
            csc_col_ptr_raw + (num_cols - 1),
            sizeof(int),
            cudaMemcpyDeviceToHost
        );
    
        // read col_counts[num_cols - 1]
        cudaMemcpy(
            &last_col_hist_count,
            col_counts_raw + (num_cols - 1),
            sizeof(int),
            cudaMemcpyDeviceToHost
        );
    
        // total nnz in outliers across all columns:
        int total_nnz_in_outliers = last_col_prefix + last_col_hist_count;
    
        //  D) write total_nnz_in_outliers into csc_col_ptr[num_cols]
        cudaMemcpy(
            csc_col_ptr_raw + num_cols,
            &total_nnz_in_outliers,
            sizeof(int),
            cudaMemcpyHostToDevice
        );
    }
  

    // 3) Kernel: fill the CSC arrays
    //    We'll re-use col_counts[] as a set of counters with atomicAdd.
    //    So first zero them out again:
    col_counts.zero_();

    {
      int block_size = 256;
      int grid_size = (nnz + block_size - 1) / block_size;
      fill_csc_arrays_kernel<<<grid_size, block_size>>>(
          temp_quantized.data_ptr<int8_t>(),
          reinterpret_cast<const long*>(outlier_indices.data_ptr<int64_t>()),
          nnz,
          col_counts.data_ptr<int>(),
          csc_col_ptr.data_ptr<int>(),
          csc_values.data_ptr<int8_t>(),
          csc_row_indices.data_ptr<int>(),
          reinterpret_cast<half*>(csc_scale.data_ptr<at::Half>()),
          reinterpret_cast<const half*>(act_outlier_scale.data_ptr<at::Half>()),
          num_cols
      );
      cudaError_t err = cudaGetLastError();
      if (err != cudaSuccess) {
          printf("fill_csc_arrays_kernel failed: %s\n", cudaGetErrorString(err));
      }
    }
}

__global__ void dense_to_csr_kernel(
    const int8_t* __restrict__ dense_weight,  // [M x padded_K] physical layout (each int8 packs 2 int4)
    const int* __restrict__ active_k,           // [num_active] logical column indices (0 .. padded_K*2-1)
    int M,
    int padded_K,       // number of int8 elements per row (physical width)
    int num_active,     // number of active logical columns per row
    int8_t* __restrict__ csr_values,           // output: size M * num_active (each value is one unpacked int4 stored as int8)
    int* __restrict__ csr_col_indices,         // output: size M * num_active (stores logical column indices)
    int* __restrict__ csr_row_ptr              // output: size M+1
)
{
    int m = blockIdx.x * blockDim.x + threadIdx.x;
    if (m >= M) return;

    // Each row m starts at this position in the CSR arrays.
    int row_start = m * num_active;
    csr_row_ptr[m] = row_start;

    for (int j = 0; j < num_active; j++) {
        // active_k[j] is the logical column index.
        int logical_col = active_k[j];          // ranges from 0 to (padded_K*2 - 1)
        int physical_col = logical_col / 2;       // each int8 holds two int4 values
        int shift = (logical_col & 1) * 4;          // lower nibble if even, upper nibble if odd

        int8_t packed_val = dense_weight[m * padded_K + physical_col];
        int8_t unpacked = (packed_val >> shift) & 0xF;  // extract the 4-bit value

        // Convert from unsigned (0..15) to signed int4 (-8..7)
        if (unpacked >= 8) {
            unpacked -= 16;
        }

        csr_values[row_start + j] = unpacked;
        csr_col_indices[row_start + j] = logical_col;
    }

    if (m == M - 1) {
        csr_row_ptr[M] = M * num_active;
    }
}


void convert_dense_quantized_weight_to_csr_host(
    torch::Tensor dense_quantized_weight,  // int8, shape [M, padded_K] (each int8 packs 2 int4 values)
    torch::Tensor active_k,                // int32, shape [num_active] with logical column indices (0 .. padded_K*2-1)
    torch::Tensor csr_values,              // pre-allocated tensor, shape [M * num_active]
    torch::Tensor csr_col_indices,         // pre-allocated tensor, shape [M * num_active]
    torch::Tensor csr_row_ptr              // pre-allocated tensor, shape [M + 1]
) {
    TORCH_CHECK(dense_quantized_weight.dim() == 2, "dense_quantized_weight must be 2D");
    TORCH_CHECK(active_k.dim() == 1, "active_k must be 1D");

    int M = dense_quantized_weight.size(0);
    int padded_K = dense_quantized_weight.size(1);
    int num_active = active_k.size(0);
    int total_nnz = M * num_active;

    TORCH_CHECK(csr_values.numel() == total_nnz, "csr_values must have ", total_nnz, " elements");
    TORCH_CHECK(csr_col_indices.numel() == total_nnz, "csr_col_indices must have ", total_nnz, " elements");
    TORCH_CHECK(csr_row_ptr.numel() == (M + 1), "csr_row_ptr must have ", (M + 1), " elements");

    int threads = 256;
    int blocks  = (M + threads - 1) / threads;
    dense_to_csr_kernel<<<blocks, threads>>>(
        dense_quantized_weight.data_ptr<int8_t>(),
        active_k.data_ptr<int>(),
        M,
        padded_K,
        num_active,
        csr_values.data_ptr<int8_t>(),
        csr_col_indices.data_ptr<int>(),
        csr_row_ptr.data_ptr<int>()
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("dense_to_csr_kernel failed: %s\n", cudaGetErrorString(err));
    }
}

//--------------------------------------------------------------
// pack kernel for FP16 matrices in row-major order.
// Now the output buffer is assumed to have a physical row width equal to padded_virtualK,
// where padded_virtualK is at least as large as the virtual width (in FP16 elements).
// The kernel writes packed int8 values (each packing 2 FP16 values) into the first (cols/2) positions of each row.
__global__ void pack_int4_kernel_fp16_rowmajor(
    const __half* __restrict__ input,   // FP16 input [rows, cols]
    const float scale,    // FP32 scale per row
    int total_elements,                 // rows * cols (logical FP16 count)
    int cols,                           // virtual width (number of FP16 values per row)
    int physical_stride,                // padded row length (in int8 elements), at least >= cols/2
    int8_t* __restrict__ output)        // output: packed int8 (each holds two 4-bit values)
{
    int total_pairs = total_elements / 2;
    int valid_pairs_per_row = cols / 2;

    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index0 = pair_idx * 2;
    if (pair_idx < total_pairs) {
        int row = index0 / cols;
        int pair_in_row = (index0 % cols) / 2;
        if (pair_in_row < valid_pairs_per_row) {
            int value0 = 0, value1 = 0;
            if (index0 < total_elements) {
                float x = __half2float(input[index0]);
                int q = roundf(x / scale);
                q = max(-7, min(q, 7));
                if (q < 0) q += 16;
                value0 = q & 0xF;
            }
            int index1 = index0 + 1;
            if (index1 < total_elements) {
                float x = __half2float(input[index1]);
                int q = roundf(x / scale);
                q = max(-7, min(q, 7));
                if (q < 0) q += 16;
                value1 = q & 0xF;
            }
            int8_t packed = (int8_t)((value1 << 4) | value0);
            int physical_index = row * physical_stride + pair_in_row;
            output[physical_index] = packed;
        }
    }
}

//--------------------------------------------------------------
// Host launcher for pack_int4_kernel_fp16_rowmajor.
void pack_int4_kernel_fp16_rowmajor_host(
    torch::Tensor input_fp16,   // FP16 [rows, cols]
    float scale,        // FP32, scalar
    int rows,
    int cols,
    int padded_virtualK,        // padded physical stride (in int8 elements)
    torch::Tensor output_int8   // int8 [rows, padded_virtualK]
) {
    TORCH_CHECK(input_fp16.dim() == 2, "input_fp16 must be 2D");
    TORCH_CHECK(output_int8.dim() == 2,  "output_int8 must be 2D");

    int total_elements = rows * cols;
    int threads = 256;
    int weight_pairs = total_elements / 2;
    int blocks = (weight_pairs + threads - 1) / threads;

    pack_int4_kernel_fp16_rowmajor<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(input_fp16.data_ptr<at::Half>()),
        scale,
        total_elements,
        cols,
        padded_virtualK,
        output_int8.data_ptr<int8_t>()
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[pack_int4_kernel_fp16_rowmajor_host] CUDA error: %s\n", cudaGetErrorString(err));
    }
}

//--------------------------------------------------------------
// packing kernel for FP16 activation matrices in column-major order.
__global__ void pack_int4_kernel_B_fp16(
    const __half* __restrict__ input,    // FP16 activation [K, N]
    const float act_rscale,  // FP32 per-column reciprocal scale [N]
    int8_t* __restrict__ output,           // shape [N, padded_virtualK]
    int virtual_K,                         // number of FP16 elements in a column (must be even)
    int N,
    int physical_stride)                   // padded col length (in int8 elements)
{
    int logical_packedK = virtual_K / 2;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int pair_in_col = blockIdx.y * blockDim.y + threadIdx.y;
    if (col < N && pair_in_col < logical_packedK) {
        int idx0 = (pair_in_col * 2) * N + col;
        int idx1 = idx0 + N;

        int a = __float2int_rn(__half2float(input[idx0]) * act_rscale);
        int b = __float2int_rn(__half2float(input[idx1]) * act_rscale);

        a = max(-7, min(a, 7));
        b = max(-7, min(b, 7));

        if (a < 0) a += 16;
        if (b < 0) b += 16;

        int8_t packed = static_cast<int8_t>((b << 4) | (a & 0x0F));
        int physical_index = col * physical_stride + pair_in_col;
        output[physical_index] = packed;
    }
}

//--------------------------------------------------------------
// CUTLASS int4 GEMM using packed inputs.
// "virtual_K" is the original number of int4 (unpacked) values per row.
cudaError_t CutlassInt4Gemm(
    int M,
    int N,
    int virtual_K,
    const cutlass::int4b_t *A,
    int lda,
    const cutlass::int4b_t *B,
    int ldb,
    int32_t *C,
    int ldc,
    cutlass::half_t alpha,   // scale the accumulated value
    cutlass::half_t beta     // scale the existing C value
) {
  using ElementAccumulator = int32_t;       // accumulators in int32
  using ElementComputeEpilogue = cutlass::half_t;  // epilogue scaling in half

  using ElementInputA = cutlass::int4b_t;
  using ElementInputB = cutlass::int4b_t;
  using ElementOutput = int32_t;

  using LayoutInputA = cutlass::layout::RowMajor;
  using LayoutInputB = cutlass::layout::ColumnMajor;
  using LayoutOutput = cutlass::layout::RowMajor;

  using MMAOp = cutlass::arch::OpClassTensorOp;
  using SmArch = cutlass::arch::Sm80;

  using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 256, 256>;
  using ShapeMMAWarp        = cutlass::gemm::GemmShape<64, 64, 256>;
  using ShapeMMAOp          = cutlass::gemm::GemmShape<16, 8, 64>;

  using SwizzleThreadBlock =
      cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

  // 4 = the epilogue vector length processed in one iteration
  using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
      ElementOutput, 4, ElementAccumulator, ElementComputeEpilogue>;

  constexpr int NumStages = 3;

  using Gemm = cutlass::gemm::device::Gemm<
      ElementInputA,
      LayoutInputA,
      ElementInputB,
      LayoutInputB,
      ElementOutput,
      LayoutOutput,
      ElementAccumulator,
      MMAOp,
      SmArch,
      ShapeMMAThreadBlock,
      ShapeMMAWarp,
      ShapeMMAOp,
      EpilogueOp,
      SwizzleThreadBlock,
      NumStages
  >;
  cutlass::gemm::GemmCoord problem_size(M, N, virtual_K);

  // alpha, beta are already half
  int split_k_slices = 1;

  typename Gemm::Arguments arguments{
      problem_size,
      {A, lda},
      {B, ldb},
      {C, ldc},
      {C, ldc},
      {alpha, beta},
      split_k_slices
  };

  Gemm gemm_op;
  cutlass::Status status = gemm_op(arguments);
  if (status != cutlass::Status::kSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

// -----------------------------------------------------------------
// Modified spGEMM kernel (CSR @ CSC) for outlier accumulation
// -----------------------------------------------------------------
// For each output element (m, active_idx), we use active_idx to look up
// the actual output column (n = active_k[active_idx]). Then for each
// nonzero in CSC column n, we use the relative index (i - start)
// to directly index into the CSR weights for row m.
__global__ void spgemm_csr_csc_kernel(
    int M,
    int N,  // full output width
    const int* __restrict__ csr_row_ptr,       // [M+1]
    const int8_t* __restrict__ csr_values,       // [M * num_active]
    const int* __restrict__ active_k,            // sorted active set; length = num_active
    int num_active,
    const int* __restrict__ csc_col_ptr,         // [N+1]
    const int* __restrict__ csc_row_indices,       // [nnz]
    const int8_t* __restrict__ csc_values,         // [nnz]
    const half* __restrict__ csc_scale,          // [nnz]
    half weight_scale,                           // scalar (FP16)
    half* __restrict__ output                    // [M x N] in row-major order
)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;              // output row
    int active_idx = blockIdx.x * blockDim.x + threadIdx.x;       // index into active set

    if (m < M && active_idx < num_active) {
        // Look up the actual output column from the active set.
        int n = active_k[active_idx];

        int start = csc_col_ptr[n];
        int end   = csc_col_ptr[n+1];
        half sum = __float2half(0.0f);
        for (int i = start; i < end; i++) {
            // Directly compute the corresponding weight index since the ordering is identical.
            int j = i - start;  // relative index for this column, guaranteed to match
            // For row m, the corresponding weight is stored at:
            // csr_values[ csr_row_ptr[m] + j ]
            int row_start = csr_row_ptr[m];
            int idx = row_start + j;
            int8_t q_w = csr_values[idx];
            int8_t q_a = csc_values[i];
            float dequant_weight = (float)q_w * __half2float(weight_scale);
            float dequant_act    = (float)q_a * __half2float(csc_scale[i]);
            float contrib = dequant_weight * dequant_act;
            sum = __hadd(sum, __float2half(contrib));
        }
        // Write the accumulated contribution into the output at (m, n)
        int out_idx = m * N + n;
        output[out_idx] = __hadd(output[out_idx], sum);
    }
}


__global__ void add_inlier_results_kernel(
    const int32_t* __restrict__ inliers,  // [M*N] int32
    int M,
    int N,
    half w_scale,
    half a_scale,
    half* __restrict__ output             // [M*N] FP16
)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx < total) {
        float val = static_cast<float>(inliers[idx]);
        // Multiply by (weight_scale * act_scale)
        val *= (__half2float(w_scale) * __half2float(a_scale));
        // Accumulate into output
        float old_val = __half2float(output[idx]);
        output[idx] = __float2half(old_val + val);
    }
}


//--------------------------------------------------------------
// Fused GEMM with outlier accumulation (parallel version)
torch::Tensor fused_quantized_gemm_packedW_cuda(
    torch::Tensor packed_weight_int8,  
    float weight_scale,        
    int M, int K, int N,
    torch::Tensor act_fp,             
    float act_scale,
    // Weight CSR parameters:
    torch::Tensor csr_values,
    torch::Tensor csr_col_indices,
    torch::Tensor csr_row_ptr,
    // Activation outlier CSC parameters:
    torch::Tensor csc_values,
    torch::Tensor csc_row_indices,
    torch::Tensor csc_scale,
    torch::Tensor csc_col_ptr
)
{
    cudaError_t err;
    int pad_align = 16;
    int padded_virtualK = ((K + pad_align - 1) / pad_align) * pad_align;

    // Create CUDA events (for timing if desired)
    cudaEvent_t start, end;
    cudaEvent_t pack_start, pack_end;
    cudaEvent_t gemm_start, gemm_end;
    cudaEvent_t spgemm_start, spgemm_end;
    cudaEvent_t add_start, add_end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&pack_start);
    cudaEventCreate(&pack_end);
    cudaEventCreate(&gemm_start);
    cudaEventCreate(&gemm_end);
    cudaEventCreate(&spgemm_start);
    cudaEventCreate(&spgemm_end);
    cudaEventCreate(&add_start);
    cudaEventCreate(&add_end);

    // Create separate streams for inlier GEMM+conversion and spGEMM
    cudaStream_t stream_inlier, stream_spgemm;
    cudaStreamCreate(&stream_inlier);
    cudaStreamCreate(&stream_spgemm);

    cudaEventRecord(start);

    // -------------------------------------------------------------
    // (1) Pack inlier activations (launch on default or one stream)
    // -------------------------------------------------------------
    cudaEventRecord(pack_start);
    auto quantized_act = torch::empty({N, padded_virtualK}, torch::dtype(torch::kInt8).device(act_fp.device()));
    float act_rscale = 1.0f / act_scale;

    dim3 blockDim_(16, 16);
    int logical_packedK = K / 2;  
    dim3 gridDim_((N + blockDim_.x - 1) / blockDim_.x,
                  (logical_packedK + blockDim_.y - 1) / blockDim_.y);

    pack_int4_kernel_B_fp16<<<gridDim_, blockDim_>>>(
        reinterpret_cast<const __half*>(act_fp.data_ptr<at::Half>()),
        act_rscale,
        quantized_act.data_ptr<int8_t>(),
        K,
        N,
        padded_virtualK
    );
    cudaEventRecord(pack_end);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in pack_int4_kernel_B_fp16 (activation): %s\n", cudaGetErrorString(err));
    }

    // -------------------------------------------------------------
    // (2) Allocate intermediate and output tensors
    // -------------------------------------------------------------
    // For dense GEMM accumulation (int32)
    auto C_int32 = torch::zeros({M, N}, torch::dtype(torch::kInt32).device(act_fp.device()));
    // Final fp16 output; note both kernels will write into this
    auto output = torch::empty({M, N}, torch::dtype(torch::kFloat16).device(act_fp.device()));

    // -------------------------------------------------------------
    // (3) Launch dense int4 GEMM (inliers) in stream_inlier
    // -------------------------------------------------------------
    cudaEventRecord(gemm_start, stream_inlier);
    int lda = padded_virtualK * 2;  
    int ldb = padded_virtualK * 2;  
    int ldc = N;

    cutlass::half_t alpha_half(weight_scale * act_scale);
    cutlass::half_t beta_half(0);

    err = CutlassInt4Gemm(
        M, N, K,
        reinterpret_cast<const cutlass::int4b_t*>(packed_weight_int8.data_ptr<int8_t>()),
        lda,
        reinterpret_cast<const cutlass::int4b_t*>(quantized_act.data_ptr<int8_t>()),
        ldb,
        C_int32.data_ptr<int32_t>(),
        ldc,
        alpha_half,
        beta_half
    );
    cudaEventRecord(gemm_end, stream_inlier);
    if (err != cudaSuccess) {
        printf("CUTLASS int4 GEMM failed (packedW version): %s\n", cudaGetErrorString(err));
        return torch::Tensor();
    }

    // -------------------------------------------------------------
    // (4) Launch spGEMM (sparse outlier accumulation) in stream_spgemm
    // -------------------------------------------------------------
    if (csc_values.numel() > 0) {
        cudaEventRecord(spgemm_start, stream_spgemm);

        // Get number of active elements from csr_row_ptr
        int h_num_active = 0;
        cudaMemcpy(&h_num_active, csr_row_ptr.data_ptr<int>() + 1, sizeof(int), cudaMemcpyDeviceToHost);
        int num_active = h_num_active;

        // Launch kernel over M rows and num_active columns (active set)
        dim3 blockDim(16, 16);
        dim3 gridDim((num_active + blockDim.x - 1) / blockDim.x,
                     (M + blockDim.y - 1) / blockDim.y);
        spgemm_csr_csc_kernel<<<gridDim, blockDim, 0, stream_spgemm>>>(
            M, N,  // full output width for proper indexing
            csr_row_ptr.data_ptr<int>(),
            csr_values.data_ptr<int8_t>(),
            csr_col_indices.data_ptr<int>(),  // active set from row 0
            num_active,
            csc_col_ptr.data_ptr<int>(),
            csc_row_indices.data_ptr<int>(),
            csc_values.data_ptr<int8_t>(),
            reinterpret_cast<const half*>(csc_scale.data_ptr<at::Half>()),
            __float2half(weight_scale),
            reinterpret_cast<half*>(output.data_ptr<at::Half>())
        );
        cudaEventRecord(spgemm_end, stream_spgemm);
    }


    // -------------------------------------------------------------
    // (5) Synchronize streamsand combine results
    // -------------------------------------------------------------
    cudaStreamSynchronize(stream_inlier);
    cudaStreamSynchronize(stream_spgemm);

    cudaEventRecord(add_start);
    // Add the inlier GEMM int32 accumulators into the final FP16 output:
    {
        // block / grid setup
        int total_elems = M * N;
        int block_size = 256;
        int grid_size = (total_elems + block_size - 1) / block_size;

        add_inlier_results_kernel<<<grid_size, block_size>>>(
            C_int32.data_ptr<int32_t>(),      // inlier GEMM result (accumulated in int32)
            M,
            N,
            __float2half(weight_scale),       // same scales used in spGEMM
            __float2half(act_scale),
            reinterpret_cast<half*>(output.data_ptr<at::Half>())
        );
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("add_inlier_results_kernel failed: %s\n", cudaGetErrorString(err));
        }
    }
    cudaEventRecord(add_end);


    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // -------------------------------------------------------------
    // (6) Print timing information (if desired)
    // -------------------------------------------------------------
    float total_time = 0.0f;
    float pack_time = 0.0f, gemm_time = 0.0f, add_time = 0.0f;
    cudaEventElapsedTime(&total_time, start, end);
    cudaEventElapsedTime(&pack_time, pack_start, pack_end);
    cudaEventElapsedTime(&gemm_time, gemm_start, gemm_end);
    cudaEventElapsedTime(&add_time, add_start, add_end);

    printf("Fused Quantized GEMM (packedW) timing (ms):\n");
    printf("  Total time:           %f\n", total_time);
    printf("  Activation packing:   %f (%.2f%%)\n", pack_time, (pack_time / total_time) * 100.0f);
    printf("  Dense GEMM:           %f (%.2f%%)\n", gemm_time, (gemm_time / total_time) * 100.0f);
    printf("  Add Results:          %f (%.2f%%)\n", add_time, (add_time / total_time) * 100.0f);
    if(csc_values.numel() > 0)
    {
        float spgemm_time = 0.0f;
        cudaEventElapsedTime(&spgemm_time, spgemm_start, spgemm_end);
        printf("  spGEMM:               %f (%.2f%%)\n", spgemm_time, (spgemm_time / total_time) * 100.0f);
    }

    // -------------------------------------------------------------
    // Clean up resources
    // -------------------------------------------------------------
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaEventDestroy(pack_start);
    cudaEventDestroy(pack_end);
    cudaEventDestroy(gemm_start);
    cudaEventDestroy(gemm_end);
    cudaEventDestroy(spgemm_start);
    cudaEventDestroy(spgemm_end);
    cudaEventDestroy(add_start);
    cudaEventDestroy(add_end);
    cudaStreamDestroy(stream_inlier);
    cudaStreamDestroy(stream_spgemm);

    return output;
}



//--------------------------------------------------------------
// PyBind11 binding code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_int4_kernel_fp16_rowmajor", &pack_int4_kernel_fp16_rowmajor_host,
          "Pack FP16 matrix in row-major order into int4 format");
    m.def("fused_quantized_gemm_packedW_cuda", &fused_quantized_gemm_packedW_cuda,
          "Fused quantized GEMM with packed weight and activation that returns a FP16 tensor");
    m.def("quantize_and_convert_outliers_to_csc", &quantize_and_convert_outliers_to_csc_host,
            "Quantize outlier activations to int8 and convert to CSC format.");
    m.def("convert_dense_quantized_weight_to_csr", &convert_dense_quantized_weight_to_csr_host,
            "Convert the quantized weight matrix to CSR format.");
}