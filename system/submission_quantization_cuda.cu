#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <vector>
#include <stdio.h>
#include <inttypes.h>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/half.h"
#include "cutlass/gemm/device/gemm_universal.h"

//--------------------------------------------------------------
// CUTLASS GEMM routines
//--------------------------------------------------------------

cudaError_t CutlassInt4Int8Gemm(
    int M,
    int N,
    int virtual_K,
    const cutlass::int4b_t *A,
    int lda, // physical row stride of A (in int8 elements, each packing 2 int4s)
    const int8_t *B,
    int ldb, // for B, ldb = leading dimension (in int8 elements) in column-major layout
    int32_t *C,
    int ldc, // C is row-major
    cutlass::half_t alpha,   // scale the accumulated value
    cutlass::half_t beta     // scale the existing C value
) {
    using ElementA = cutlass::int4b_t;   // int4 type (weight)
    using ElementB = int8_t;             // int8 type (activation)
    using ElementOutput = int32_t;       // int32 output
    using ElementAccumulator = int32_t;
    using ElementComputeEpilogue = cutlass::half_t;  // epilogue scaling in half

    using GemmInt4Int8 = cutlass::gemm::device::GemmUniversal<
        ElementA,
        cutlass::layout::RowMajor,       // A is row-major
        ElementB,
        cutlass::layout::ColumnMajor,    // B is column-major (packed)
        ElementOutput,
        cutlass::layout::RowMajor,       // C is row-major
        ElementAccumulator,
        cutlass::arch::OpClassTensorOp,
        cutlass::arch::Sm80,
        cutlass::gemm::GemmShape<128,128,64>,
        cutlass::gemm::GemmShape<64,64,64>,
        cutlass::gemm::GemmShape<16,8,32>,
        cutlass::epilogue::thread::LinearCombination<
                ElementOutput,
                128 / cutlass::sizeof_bits<ElementOutput>::value,
                ElementAccumulator,
                ElementComputeEpilogue>,
        cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<>,
        4,    // Stages
        32,   // AlignmentA (in units of int4 elements)
        16,   // AlignmentB (in units of int8 elements)
        cutlass::arch::OpMultiplyAddMixedInputUpcast,
        cutlass::ComplexTransform::kNone,
        cutlass::ComplexTransform::kNone
    >;

    typename GemmInt4Int8::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,         // mode
        cutlass::gemm::GemmCoord{M, N, virtual_K},        // problem size
        1,                                               // split-k slices
        {alpha, beta},                                   // epilogue output op parameters
        reinterpret_cast<const void*>(A),                // pointer A
        reinterpret_cast<const void*>(B),                // pointer B
        reinterpret_cast<const void*>(C),                // pointer C
        reinterpret_cast<void*>(C),                      // pointer D (output)
        /* Strides: */
        static_cast<int64_t>(lda),
        static_cast<int64_t>(ldb),
        static_cast<int64_t>(ldc),
        static_cast<int64_t>(ldc),
        /* Layout-specific strides: */
        static_cast<int64_t>(lda),
        static_cast<int64_t>(ldb),
        static_cast<int64_t>(ldc),
        static_cast<int64_t>(ldc),
        /* Extra pointers for split-K reduction (unused): */
        nullptr, nullptr, nullptr
    );

    GemmInt4Int8 gemm_op;
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

//--------------------------------------------------------------

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
    cutlass::half_t alpha,   // scaling factor (here used as identity, i.e. 1.0)
    cutlass::half_t beta     // scaling factor (0.0)
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

//--------------------------------------------------------------
// Debug flag (set to 1 to enable device prints)
#define DEBUG_PRINT 0

//--------------------------------------------------------------
// pack_int4_kernel_fp16_rowmajor: Packs FP16 input into int4 stored in int8.
__global__ void pack_int4_kernel_fp16_rowmajor(
    const __half* __restrict__ input,   // FP16 input [rows, cols]
    const float scale,                  // FP32 scale per row
    int total_elements,                 // rows * cols
    int cols,                           // virtual width (number of FP16 values per row)
    int physical_stride,                // padded row length (in int8 elements)
    int8_t* __restrict__ output)        // output: packed int8
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
                int q_clamped = max(-7, min(q, 7));
                if (q_clamped < 0) q_clamped += 16;
                value0 = q_clamped & 0xF;
            }
            int index1 = index0 + 1;
            if (index1 < total_elements) {
                float x = __half2float(input[index1]);
                int q = roundf(x / scale);
                int q_clamped = max(-7, min(q, 7));
                if (q_clamped < 0) q_clamped += 16;
                value1 = q_clamped & 0xF;
            }
            int8_t packed = (int8_t)((value1 << 4) | value0);
            int physical_index = row * physical_stride + pair_in_row;
            output[physical_index] = packed;

            if (DEBUG_PRINT && row == 0 && pair_in_row < 4) {
                printf("[DEBUG pack_int4] row %d, pair_in_row %d, index0 %d, index1 %d, value0 %d, value1 %d, packed %d\n",
                    row, pair_in_row, index0, index1, value0, value1, packed);
            }
        }
    }
}

//--------------------------------------------------------------
// quantize_act_inliers_kernel: Quantizes activations for inlier columns.
// Assumes input matrix is [K x N] in column-major order.
__global__ void quantize_act_inliers_kernel(
    const __half* __restrict__ input,   // FP16 activation matrix [K x N]
    float scale,                        // activation scale for inliers
    int rows,                           // K
    int cols,                           // N
    int8_t* __restrict__ output)        // output buffer in column-major order
{
    int pair_per_col = (rows + 1) / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_pairs = cols * pair_per_col;
    if (idx < total_pairs) {
        int col = idx / pair_per_col;
        int pair_in_col = idx % pair_per_col;
        int row0 = pair_in_col * 2;
        int row1 = row0 + 1;
        int value0 = 0, value1 = 0;
        if (row0 < rows) {
            float x = __half2float(input[row0 + col * rows]);
            int q = roundf(x / scale);
            q = max(-7, min(q, 7));
            if (q < 0) q += 16;
            value0 = q & 0xF;
        }
        if (row1 < rows) {
            float x = __half2float(input[row1 + col * rows]);
            int q = roundf(x / scale);
            q = max(-7, min(q, 7));
            if (q < 0) q += 16;
            value1 = q & 0xF;
        }
        int8_t packed = (int8_t)((value1 << 4) | value0);
        output[col * pair_per_col + pair_in_col] = packed;
    }
}

//--------------------------------------------------------------
// quantize_act_outliers_kernel: Quantizes outlier activations into int8
// and zeroes out the original activation to avoid reprocessing.
__global__ void quantize_act_outliers_kernel(
    __half* __restrict__ input,    // FP16 [K x N] in column-major order
    int rows,                      // K
    int cols,                      // N
    const int64_t* __restrict__ outlier_cols,
    const float* __restrict__ outlier_scales,
    int n_outlier,
    int8_t* __restrict__ output,   // output buffer (column-major, one column per outlier)
    int outlier_ld)                // leading dimension for output (number of rows)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = rows * n_outlier;
    if (idx < total) {
        int outlier_idx = idx / rows;
        int row = idx % rows;
        int col = outlier_cols[outlier_idx];
        float scale = outlier_scales[outlier_idx];
        float x = __half2float(input[row + col * rows]);
        int q = (int)roundf(x / scale);
        q = max(-127, min(127, q));
        output[outlier_idx * outlier_ld + row] = (int8_t)q;
        // Zero out to prevent double counting in the inlier kernel.
        input[row + col * rows] = __float2half(0.0f);
    }
}

//--------------------------------------------------------------
// dequantize_inliers_kernel: Converts the int32 inlier GEMM output to FP16.
// Each element is scaled by (weight_scale * act_scale).
__global__ void dequantize_inliers_kernel(
    const int32_t* __restrict__ C_inliers,
    int M,
    int N,
    float scale, // weight_scale * act_scale
    __half* __restrict__ output_fp16)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * N;
    if (idx < total) {
        float val = (float) C_inliers[idx] * scale;
        output_fp16[idx] = __float2half(val);
    }
}

//--------------------------------------------------------------
// dequantize_outliers_kernel: Dequantizes outlier GEMM output and adds the contributions
// into the final FP16 output. Each outlier value is scaled by (outlier_scale / act_scale).
__global__ void dequantize_outliers_kernel(
    const int32_t* __restrict__ C_outliers,
    int M,
    int n_outlier,
    int outlier_ld,
    const int64_t* __restrict__ outlier_cols,
    const float* __restrict__ outlier_scales,
    float act_scale,
    __half* __restrict__ output_fp16,
    int N)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = M * n_outlier;
    if (idx < total) {
        int outlier_idx = idx / M;
        int m = idx % M;
        int final_col = outlier_cols[outlier_idx];
        int32_t out_val = C_outliers[m * outlier_ld + outlier_idx];
        float addend = (float) out_val * (outlier_scales[outlier_idx] / act_scale);
        int output_index = m * N + final_col;
        float current = __half2float(output_fp16[output_index]);
        output_fp16[output_index] = __float2half(current + addend);
    }
}

//--------------------------------------------------------------
// Host launcher for pack_int4_kernel_fp16_rowmajor.
void pack_int4_kernel_fp16_rowmajor_host(
    torch::Tensor input_fp16,   // FP16 [rows, cols]
    float scale,                // FP32 scalar
    int rows,
    int cols,
    int padded_virtualK,        // padded row length (in int8 elements)
    torch::Tensor output_int8   // int8 [rows, padded_virtualK]
) {
    TORCH_CHECK(input_fp16.dim() == 2, "input_fp16 must be 2D");
    TORCH_CHECK(output_int8.dim() == 2, "output_int8 must be 2D");

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
// custom_gemm_with_outliers:
// This host function quantizes activations into inlier and outlier parts,
// runs the GEMMs with α = 1.0, and then fuses results by launching two
// separate dequantization kernels that write directly to the final FP16 output.
void custom_gemm_with_outliers(
    torch::Tensor packed_weight_int4,   // int8 tensor of shape [M, padded_K]
    torch::Tensor act_fp,               // FP16 tensor [K, N] (column-major)
    float weight_scale,                 // weight scaling factor (FP32)
    float act_scale,                    // inlier activation scale (FP32)
    torch::Tensor act_outlier_indices,  // 1D int64 tensor with outlier column indices
    torch::Tensor act_outlier_scales,   // 1D float32 tensor with per-outlier scales
    torch::Tensor output_fp16           // FP16 tensor [M, N] (final output, row-major)
) {
    TORCH_CHECK(packed_weight_int4.dim() == 2, "packed_weight_int4 must be 2D");
    TORCH_CHECK(act_fp.dim() == 2, "act_fp must be 2D");
    TORCH_CHECK(output_fp16.dim() == 2, "output_fp16 must be 2D");
    TORCH_CHECK(act_outlier_indices.dim() == 1, "act_outlier_indices must be 1D");
    TORCH_CHECK(act_outlier_scales.dim() == 1, "act_outlier_scales must be 1D");

    int M = packed_weight_int4.size(0);
    int padded_K = packed_weight_int4.size(1);
    int K = act_fp.size(0);
    int N = act_fp.size(1);
    int n_outlier = act_outlier_indices.size(0);

    //--------------------------------------------------------------------------
    // 1) Set pad alignment and compute physical strides.
    //--------------------------------------------------------------------------
    int pad_align = 16;
    int pair_per_col = (K + 1) / 2;
    int inlier_physical_stride = ((pair_per_col + pad_align - 1) / pad_align) * pad_align;

    //--------------------------------------------------------------------------
    // 2) Allocate inlier activation buffer.
    //--------------------------------------------------------------------------
    auto quantized_act_inliers = torch::empty(
        {N * inlier_physical_stride},
        torch::dtype(torch::kInt8).device(act_fp.device())
    );

    //--------------------------------------------------------------------------
    // 3) Allocate outlier buffers only if needed.
    //--------------------------------------------------------------------------
    torch::Tensor quantized_act_outliers;
    torch::Tensor C_outliers;
    int K_padded = 0;
    int padded_n_outlier = 0;
    if (n_outlier > 0) {
        K_padded = ((K + pad_align - 1) / pad_align) * pad_align;
        quantized_act_outliers = torch::empty(
            {n_outlier * K_padded},
            torch::dtype(torch::kInt8).device(act_fp.device())
        );
        padded_n_outlier = ((n_outlier + 3) / 4) * 4;
        C_outliers = torch::empty(
            {M, padded_n_outlier},
            torch::dtype(torch::kInt32).device(act_fp.device())
        );
    }

    //--------------------------------------------------------------------------
    // 4) Allocate temporary GEMM accumulator buffer for inliers.
    //--------------------------------------------------------------------------
    auto C_inliers = torch::empty({M, N}, torch::dtype(torch::kInt32).device(act_fp.device()));

    //--------------------------------------------------------------------------
    // 5) Launch kernels to quantize the activations.
    //--------------------------------------------------------------------------
    __half* act_fp_ptr = reinterpret_cast<__half*>(act_fp.data_ptr<at::Half>());
    int8_t* quant_inliers_ptr = quantized_act_inliers.data_ptr<int8_t>();
    const int64_t* d_outlier_cols = act_outlier_indices.data_ptr<int64_t>();
    const float* d_outlier_scales = act_outlier_scales.data_ptr<float>();

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time_quant_inliers = 0.0f;
    float time_quant_outliers = 0.0f;
    float time_gemm_inliers   = 0.0f;
    float time_gemm_outliers  = 0.0f;
    float time_dequantize_inliers = 0.0f;
    float time_dequantize_outliers = 0.0f;

    // (a) Outlier activation quantization.
    if (n_outlier > 0) {
        int total_outlier_elements = n_outlier * K;
        int threads = 256;
        int blocks = (total_outlier_elements + threads - 1) / threads;
        cudaEventRecord(start, 0);
        quantize_act_outliers_kernel<<<blocks, threads>>>(
            act_fp_ptr,
            K,
            N,
            d_outlier_cols,
            d_outlier_scales,
            n_outlier,
            quantized_act_outliers.data_ptr<int8_t>(),
            K_padded
        );
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_quant_outliers, start, stop);
        cudaError_t err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("[quantize_act_outliers_kernel] CUDA error: %s\n", cudaGetErrorString(err));
            return;
        }
    }

    // (b) Inlier activation quantization.
    int total_pairs = N * pair_per_col;
    int threads = 256;
    int blocks = (total_pairs + threads - 1) / threads;
    cudaEventRecord(start, 0);
    quantize_act_inliers_kernel<<<blocks, threads>>>(
        act_fp_ptr,
        act_scale,
        K,
        N,
        quant_inliers_ptr
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_quant_inliers, start, stop);
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[quantize_act_inliers_kernel] CUDA error: %s\n", cudaGetErrorString(err));
        return;
    }

    //--------------------------------------------------------------------------
    // 6) Launch the GEMM operations.
    //--------------------------------------------------------------------------
    const cutlass::int4b_t* weight_ptr =
         reinterpret_cast<const cutlass::int4b_t*>(packed_weight_int4.data_ptr<int8_t>());
    const cutlass::int4b_t* act_inliers_ptr =
         reinterpret_cast<const cutlass::int4b_t*>(quant_inliers_ptr);

    // Use α = 1.0 for both GEMMs.
    cutlass::half_t gemm_alpha(1.0f);
    cutlass::half_t gemm_beta(0.0f);
    cudaEventRecord(start, 0);
    err = CutlassInt4Gemm(
        M,
        N,
        K,                      // Virtual K (logical number of int4 elements)
        weight_ptr,             // A: packed weight (int4 stored in int8)
        padded_K * 2,           // lda: padded_K int8 elements per row (each int8 packs 2 int4s)
        act_inliers_ptr,        // B: inlier activation buffer (packed int4 in int8)
        inlier_physical_stride * 2, // ldb: each column has inlier_physical_stride int8 elements
        C_inliers.data_ptr<int32_t>(),
        N,                      // ldc: C is row-major
        gemm_alpha,
        gemm_beta
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_gemm_inliers, start, stop);
    if (err != cudaSuccess) {
        printf("[CutlassInt4Gemm] GEMM error\n");
        return;
    }

    // Outlier GEMM (if any outliers)
    if (n_outlier > 0) {
        gemm_alpha = __float2half(1.0f);  // identity scaling
        cudaEventRecord(start, 0);
        err = CutlassInt4Int8Gemm(
            M,
            n_outlier,
            K,
            weight_ptr,
            padded_K * 2,
            quantized_act_outliers.data_ptr<int8_t>(),
            K_padded,
            C_outliers.data_ptr<int32_t>(),
            padded_n_outlier,
            gemm_alpha,
            gemm_beta
        );
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_gemm_outliers, start, stop);
        if (err != cudaSuccess) {
            printf("[CutlassInt4Int8Gemm] GEMM error\n");
            return;
        }
    }

    //--------------------------------------------------------------------------
    // 7) Dequantize and fuse GEMM results using separate kernels.
    // (a) Dequantize inlier GEMM output.
    threads = 256;
    blocks = (M * N + threads - 1) / threads;
    cudaEventRecord(start, 0);
    dequantize_inliers_kernel<<<blocks, threads>>>(
        C_inliers.data_ptr<int32_t>(),
        M,
        N,
        weight_scale * act_scale,
        reinterpret_cast<__half*>(output_fp16.data_ptr<at::Half>())
    );
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_dequantize_inliers, start, stop);

    // (b) Dequantize outlier GEMM output and add contributions.
    if (n_outlier > 0) {
        threads = 256;
        blocks = (M * n_outlier + threads - 1) / threads;
        cudaEventRecord(start, 0);
        dequantize_outliers_kernel<<<blocks, threads>>>(
            C_outliers.data_ptr<int32_t>(),
            M,
            n_outlier,
            padded_n_outlier,
            d_outlier_cols,
            d_outlier_scales,
            act_scale,
            reinterpret_cast<__half*>(output_fp16.data_ptr<at::Half>()),
            N
        );
        cudaEventRecord(stop, 0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&time_dequantize_outliers, start, stop);
    }

    float total_time = time_quant_inliers + time_quant_outliers +
                       time_gemm_inliers   + time_gemm_outliers   +
                       time_dequantize_inliers + time_dequantize_outliers;

    printf("Kernel runtime breakdown (percentages):\n");
    if (total_time > 0.0f) {
        printf("  Inlier activation quantization: %.2f%%\n", (time_quant_inliers / total_time) * 100.0f);
        if (n_outlier > 0) {
            printf("  Outlier activation quantization: %.2f%%\n", (time_quant_outliers / total_time) * 100.0f);
        }
        printf("  Inlier GEMM: %.2f%%\n", (time_gemm_inliers / total_time) * 100.0f);
        if (n_outlier > 0) {
            printf("  Outlier GEMM: %.2f%%\n", (time_gemm_outliers / total_time) * 100.0f);
        }
        printf("  Inlier dequantization: %.2f%%\n", (time_dequantize_inliers / total_time) * 100.0f);
        if (n_outlier > 0) {
            printf("  Outlier dequantization: %.2f%%\n", (time_dequantize_outliers / total_time) * 100.0f);
        }
    } else {
        printf("No kernel timings recorded.\n");
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
}

//--------------------------------------------------------------
// PyBind11 binding code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_int4_kernel_fp16_rowmajor", &pack_int4_kernel_fp16_rowmajor_host,
          "Pack FP16 matrix in row-major order into int4 format");
    m.def("custom_gemm_with_outliers", &custom_gemm_with_outliers,
          "Custom GEMM with outlier support: quantizes activations into inlier/outlier parts, runs GEMMs with identity scaling, "
          "and fuses results using separate dequantization kernels.");
}
