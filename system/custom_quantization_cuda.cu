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

//--------------------------------------------------------------
// Helper macro for logging pointer alignment.
#define LOG_PTR_ALIGN(ptr, type_size) \
    printf("  %s: address = 0x%" PRIxPTR ", alignment %% %d = %d\n", #ptr, (uintptr_t)(ptr), type_size, ((uintptr_t)(ptr)) % type_size);

//--------------------------------------------------------------
// pack kernel for FP16 matrices in row-major order.
// Now the output buffer is assumed to have a physical row width equal to padded_virtualK,
// where padded_virtualK is at least as large as the virtual width (in FP16 elements).
// The kernel writes packed int8 values (each packing 2 FP16 values) into the first (cols/2) positions of each row.
__global__ void pack_int4_kernel_fp16_rowmajor(
    const __half* __restrict__ input,   // FP16 input [rows, cols]
    const float* __restrict__ scale,    // FP32 scale per row [rows]
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
                float s = scale[row];
                float x = __half2float(input[index0]);
                int q = roundf(x / s);
                q = max(-7, min(q, 7));
                if (q < 0) q += 16;
                value0 = q & 0xF;
            }
            int index1 = index0 + 1;
            if (index1 < total_elements) {
                float s = scale[row];
                float x = __half2float(input[index1]);
                int q = roundf(x / s);
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
    torch::Tensor scale,        // FP32 [rows]
    int rows,
    int cols,
    int padded_virtualK,        // padded physical stride (in int8 elements)
    torch::Tensor output_int8   // int8 [rows, padded_virtualK]
) {
    TORCH_CHECK(input_fp16.dim() == 2, "input_fp16 must be 2D");
    TORCH_CHECK(scale.dim() == 1,      "scale must be 1D");
    TORCH_CHECK(output_int8.dim() == 2,  "output_int8 must be 2D");

    int total_elements = rows * cols;
    int threads = 256;
    int weight_pairs = total_elements / 2;
    int blocks = (weight_pairs + threads - 1) / threads;

    pack_int4_kernel_fp16_rowmajor<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(input_fp16.data_ptr<at::Half>()),
        scale.data_ptr<float>(),
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
    const float* __restrict__ act_rscale,  // FP32 per-column reciprocal scale [N]
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
        float r = act_rscale[col];

        int a = __float2int_rn(__half2float(input[idx0]) * r);
        int b = __float2int_rn(__half2float(input[idx1]) * r);

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
// Legacy scale kernel.
__global__ void scale_inlier_kernel(
    const int32_t* __restrict__ accum,
    const float* __restrict__ weight_scale,
    const float* __restrict__ act_scale,
    int M, int N,
    float* __restrict__ out)
{
    int m = blockIdx.y * blockDim.y + threadIdx.y;
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (m < M && n < N) {
        int32_t val = accum[m * N + n];
        float scale = weight_scale[m] * act_scale[n];
        out[m * N + n] = (float)val * scale;
    }
}

//--------------------------------------------------------------
// quantize outliers kernel.
// (Uses outlier_indices only for optional debug printing.)
__global__ void quantize_outliers_kernel(
    const __half* __restrict__ outlier_values,  // FP16 outlier values
    const int64_t* __restrict__ outlier_indices, // debug info (unused in computation)
    const float* __restrict__ s_outlier,      
    int nnz,
    int n_lva_o,
    int8_t* __restrict__ out_quantized)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        float scale = s_outlier[idx];
        float x = __half2float(outlier_values[idx]);
        float q = roundf(x / scale);
        int o_qmax = n_lva_o / 2 - 1;
        int o_qmin = -o_qmax;
        q = fminf(fmaxf(q, (float)o_qmin), (float)o_qmax);
        int quant = (int)q;
        out_quantized[idx] = (int8_t)quant;
        // if (idx < 4) {
        //     printf("quantize_outliers_kernel: idx=%d, x=%f, scale=%f, q=%f, quant=%d\n",
        //            idx, x, scale, q, quant);
        // }
    }
}

// New kernel to accumulate outlier contributions into the FP32 output.
__global__ void accumulate_outliers_kernel(
    const int8_t* __restrict__ quantized_outlier,
    const int64_t* __restrict__ row_indices,
    const int64_t* __restrict__ col_indices,
    const float* __restrict__ s_outlier,
    int nnz,
    int N,  // number of columns (used as stride in output)
    float* __restrict__ output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        int row = row_indices[idx];
        int col = col_indices[idx];
        float scale = s_outlier[idx];
        float dequant = ((int)quantized_outlier[idx]) * scale;
        // Atomically add the dequantized outlier contribution to the output.
        atomicAdd(&output[row * N + col], dequant);
    }
}


//--------------------------------------------------------------
// Legacy CUTLASS int4 GEMM.
cudaError_t CutlassInt4Gemm(
    const int M,
    const int N,
    const int virtual_K,
    const cutlass::int4b_t *A,
    int lda,
    const cutlass::int4b_t *B,
    int ldb,
    int32_t *C,
    int ldc)
{
    using ElementAccumulator = int32_t;
    using ElementComputeEpilogue = ElementAccumulator;
    using ElementInputA = cutlass::int4b_t;
    using ElementInputB = cutlass::int4b_t;
    using ElementOutput = int32_t;

    using LayoutInputA = cutlass::layout::RowMajor;
    using LayoutInputB = cutlass::layout::ColumnMajor;
    using LayoutOutput = cutlass::layout::RowMajor;

    using MMAOp = cutlass::arch::OpClassTensorOp;
    using SmArch = cutlass::arch::Sm80;

    using ShapeMMAThreadBlock = cutlass::gemm::GemmShape<128, 128, 128>;
    using ShapeMMAWarp = cutlass::gemm::GemmShape<64, 64, 128>;
    using ShapeMMAOp = cutlass::gemm::GemmShape<16, 8, 64>;

    using SwizzleThreadBlock = cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>;

    using EpilogueOp = cutlass::epilogue::thread::LinearCombination<
        ElementOutput, 4, ElementAccumulator, ElementComputeEpilogue>;

    constexpr int NumStages = 3;

    using Gemm = cutlass::gemm::device::Gemm<ElementInputA,
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
                                             NumStages>;

    cutlass::gemm::GemmCoord problem_size(M, N, virtual_K);

    ElementComputeEpilogue alpha(1);
    ElementComputeEpilogue beta(0);

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
// Legacy fused GEMM that packs both weight and activation.
void fused_quantized_gemm_cuda(
    torch::Tensor weight_fp,         // FP16 weight [M, K]
    torch::Tensor weight_scale,      // per-row weight scale [M]
    torch::Tensor act_fp,            // FP16 activation [K, N]
    torch::Tensor act_scale,         // per-column activation scale [N]
    torch::Tensor act_outlier,       // FP16 outlier values
    torch::Tensor act_outlier_indices, // indices (assumed shape [2, nnz])
    torch::Tensor act_outlier_scale,   // outlier scale (per outlier)
    int n_lva_o,                     // integer parameter for outlier quantization
    int M, int K, int N,
    torch::Tensor output)            // FP32 output [M, N]
{
    cudaError_t err;

    int pad_align = 16;
    int padded_virtualK = ((K + pad_align - 1) / pad_align) * pad_align; // in int8 elements

    // 1) Pack the weight.
    auto quantized_weight = torch::empty({M, padded_virtualK}, torch::dtype(torch::kInt8).device(weight_fp.device()));
    int total_weight_elements = M * K;
    int threads = 256;
    int weight_pairs = total_weight_elements / 2;
    int blocks = (weight_pairs + threads - 1) / threads;
    pack_int4_kernel_fp16_rowmajor<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(weight_fp.data_ptr<at::Half>()),
        weight_scale.data_ptr<float>(),
        total_weight_elements,
        K,                
        padded_virtualK,
        quantized_weight.data_ptr<int8_t>());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in pack_int4_kernel_fp16_rowmajor (weight): %s\n", cudaGetErrorString(err));
    }

    // 2) Pack the activation.
    auto quantized_act = torch::empty({N, padded_virtualK}, torch::dtype(torch::kInt8).device(act_fp.device()));
    auto act_rscale = torch::reciprocal(act_scale);
    dim3 blockDim(16, 16);
    int logical_packedK = K / 2;
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (logical_packedK + blockDim.y - 1) / blockDim.y);
    pack_int4_kernel_B_fp16<<<gridDim, blockDim>>>(
        reinterpret_cast<const __half*>(act_fp.data_ptr<at::Half>()),
        act_rscale.data_ptr<float>(),
        quantized_act.data_ptr<int8_t>(),
        K,
        N,
        padded_virtualK); 
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in pack_int4_kernel_B_fp16: %s\n", cudaGetErrorString(err));
    }

    // 3) Perform Int4 GEMM.
    int lda = padded_virtualK * 2;  
    int ldb = padded_virtualK * 2;  
    auto C_int32 = torch::zeros({M, N}, torch::dtype(torch::kInt32).device(weight_fp.device()));
    int ldc = N;
    err = CutlassInt4Gemm(
        M, N, K, 
        reinterpret_cast<const cutlass::int4b_t*>(quantized_weight.data_ptr<int8_t>()),
        lda,
        reinterpret_cast<const cutlass::int4b_t*>(quantized_act.data_ptr<int8_t>()),
        ldb,
        C_int32.data_ptr<int32_t>(), 
        ldc);
    if (err != cudaSuccess) {
        printf("CUTLASS int4 GEMM failed: %s\n", cudaGetErrorString(err));
        return;
    }

    // 4) Scale accumulations back to FP32.
    dim3 blockDim2(16, 16);
    dim3 gridDim2((N + blockDim2.x - 1) / blockDim2.x, (M + blockDim2.y - 1) / blockDim2.y);
    scale_inlier_kernel<<<gridDim2, blockDim2>>>(
        C_int32.data_ptr<int32_t>(),
        weight_scale.data_ptr<float>(),
        act_scale.data_ptr<float>(),
        M, N,
        output.data_ptr<float>());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in scale_inlier_kernel: %s\n", cudaGetErrorString(err));
    }

    // 5) Quantize and accumulate outliers.
    int nnz = act_outlier.numel();
    if (nnz > 0) {
        auto quantized_outlier = torch::empty_like(act_outlier, torch::dtype(torch::kInt8).device(act_outlier.device()));
        int threads_out = 256;
        int blocks_out = (nnz + threads_out - 1) / threads_out;
        quantize_outliers_kernel<<<blocks_out, threads_out>>>(
            reinterpret_cast<const __half*>(act_outlier.data_ptr<at::Half>()),
            act_outlier_indices.data_ptr<int64_t>(),
            act_outlier_scale.data_ptr<float>(),
            nnz,
            n_lva_o,
            quantized_outlier.data_ptr<int8_t>());
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in quantize_outliers_kernel (outliers): %s\n", cudaGetErrorString(err));
        }
        // Assuming act_outlier_indices is a [2, nnz] tensor:
        const int64_t* row_indices = act_outlier_indices.data_ptr<int64_t>();
        const int64_t* col_indices = row_indices + act_outlier_indices.size(1);
        accumulate_outliers_kernel<<<blocks_out, threads_out>>>(
            quantized_outlier.data_ptr<int8_t>(),
            row_indices,
            col_indices,
            act_outlier_scale.data_ptr<float>(),
            nnz,
            N,
            output.data_ptr<float>());
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in accumulate_outliers_kernel (outliers): %s\n", cudaGetErrorString(err));
        }
    }
    cudaDeviceSynchronize();
}

//--------------------------------------------------------------
// Legacy fused GEMM that takes an already packed weight.
void fused_quantized_gemm_packedW_cuda(
    torch::Tensor packed_weight_int8,  
    torch::Tensor weight_scale,        
    int M, int K, int N,
    torch::Tensor act_fp,             
    torch::Tensor act_scale,          
    torch::Tensor act_outlier,       // FP16 outlier values
    torch::Tensor act_outlier_indices, // indices (assumed shape [2, nnz])
    torch::Tensor act_outlier_scale,   // outlier scale
    int n_lva_o,                     // integer parameter for outlier quantization
    torch::Tensor output)
{
    cudaError_t err;

    int pad_align = 16;
    int padded_virtualK = ((K + pad_align - 1) / pad_align) * pad_align;

    // 1) Pack the activation (inliers).
    auto quantized_act = torch::empty({N, padded_virtualK}, torch::dtype(torch::kInt8).device(act_fp.device()));
    auto act_rscale = torch::reciprocal(act_scale);
    dim3 blockDim(16, 16);
    int logical_packedK = K / 2;
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (logical_packedK + blockDim.y - 1) / blockDim.y);
    pack_int4_kernel_B_fp16<<<gridDim, blockDim>>>(
        reinterpret_cast<const __half*>(act_fp.data_ptr<at::Half>()),
        act_rscale.data_ptr<float>(),
        quantized_act.data_ptr<int8_t>(),
        K,
        N,
        padded_virtualK);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in pack_int4_kernel_B_fp16 (activation): %s\n", cudaGetErrorString(err));
    }

    // 2) Perform Int4 GEMM.
    int lda = padded_virtualK * 2;  
    int ldb = padded_virtualK * 2;  
    auto C_int32 = torch::zeros({M, N}, torch::dtype(torch::kInt32).device(act_fp.device()));
    int ldc = N;
    err = CutlassInt4Gemm(
        M, N, K,
        reinterpret_cast<const cutlass::int4b_t*>(packed_weight_int8.data_ptr<int8_t>()),
        lda,
        reinterpret_cast<const cutlass::int4b_t*>(quantized_act.data_ptr<int8_t>()),
        ldb,
        C_int32.data_ptr<int32_t>(),
        ldc);
    if (err != cudaSuccess) {
        printf("CUTLASS int4 GEMM failed (packedW version): %s\n", cudaGetErrorString(err));
        return;
    }

    // 3) Scale accumulations back to FP32.
    dim3 blockDim2(16, 16);
    dim3 gridDim2((N + blockDim2.x - 1) / blockDim2.x, (M + blockDim2.y - 1) / blockDim2.y);
    scale_inlier_kernel<<<gridDim2, blockDim2>>>(
        C_int32.data_ptr<int32_t>(),
        weight_scale.data_ptr<float>(),
        act_scale.data_ptr<float>(),
        M, N,
        output.data_ptr<float>());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in scale_inlier_kernel (packedW version): %s\n", cudaGetErrorString(err));
    }

    // 4) Quantize and accumulate outliers.
    int nnz = act_outlier.numel();
    if (nnz > 0) {
        auto quantized_outlier = torch::empty_like(act_outlier, torch::dtype(torch::kInt8).device(act_outlier.device()));
        int threads_out = 256;
        int blocks_out = (nnz + threads_out - 1) / threads_out;
        quantize_outliers_kernel<<<blocks_out, threads_out>>>(
            reinterpret_cast<const __half*>(act_outlier.data_ptr<at::Half>()),
            act_outlier_indices.data_ptr<int64_t>(),
            act_outlier_scale.data_ptr<float>(),
            nnz,
            n_lva_o,
            quantized_outlier.data_ptr<int8_t>());
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in quantize_outliers_kernel (outliers, packedW): %s\n", cudaGetErrorString(err));
        }
        // Assuming act_outlier_indices is a [2, nnz] tensor:
        const int64_t* row_indices = act_outlier_indices.data_ptr<int64_t>();
        const int64_t* col_indices = row_indices + act_outlier_indices.size(1);
        accumulate_outliers_kernel<<<blocks_out, threads_out>>>(
            quantized_outlier.data_ptr<int8_t>(),
            row_indices,
            col_indices,
            act_outlier_scale.data_ptr<float>(),
            nnz,
            N,
            output.data_ptr<float>());
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in accumulate_outliers_kernel (outliers, packedW): %s\n", cudaGetErrorString(err));
        }
    }
    cudaDeviceSynchronize();
}

//--------------------------------------------------------------
// Fused GEMM where weight is already packed int4, activation is
// packed on-device, with outlier accumulation. Instrumented version.
void fused_quantized_gemm_packedW_instrumented_cuda(
    torch::Tensor packed_weight_int8,  
    torch::Tensor weight_scale,        
    int M, int K, int N,
    torch::Tensor act_fp,             
    torch::Tensor act_scale,          
    torch::Tensor act_outlier,       // FP16 outlier values
    torch::Tensor act_outlier_indices, // indices (assumed shape [2, nnz])
    torch::Tensor act_outlier_scale,   // outlier scale
    int n_lva_o,                     // integer parameter for outlier quantization
    torch::Tensor output)
{
    cudaError_t err;

    int pad_align = 16;
    int padded_virtualK = ((K + pad_align - 1) / pad_align) * pad_align;

    // Create CUDA events for instrumentation.
    cudaEvent_t start, end;
    cudaEvent_t pack_start, pack_end;
    cudaEvent_t gemm_start, gemm_end;
    cudaEvent_t scale_start, scale_end;
    cudaEvent_t outlier_start, outlier_end;
    cudaEventCreate(&start);
    cudaEventCreate(&end);
    cudaEventCreate(&pack_start);
    cudaEventCreate(&pack_end);
    cudaEventCreate(&gemm_start);
    cudaEventCreate(&gemm_end);
    cudaEventCreate(&scale_start);
    cudaEventCreate(&scale_end);
    cudaEventCreate(&outlier_start);
    cudaEventCreate(&outlier_end);

    cudaEventRecord(start);

    // 1) Pack the activation (inliers).
    cudaEventRecord(pack_start);
    auto quantized_act = torch::empty({N, padded_virtualK}, torch::dtype(torch::kInt8).device(act_fp.device()));
    auto act_rscale = torch::reciprocal(act_scale);
    dim3 blockDim(16, 16);
    int logical_packedK = K / 2;
    dim3 gridDim((N + blockDim.x - 1) / blockDim.x, (logical_packedK + blockDim.y - 1) / blockDim.y);
    pack_int4_kernel_B_fp16<<<gridDim, blockDim>>>(
        reinterpret_cast<const __half*>(act_fp.data_ptr<at::Half>()),
        act_rscale.data_ptr<float>(),
        quantized_act.data_ptr<int8_t>(),
        K,
        N,
        padded_virtualK);
    cudaEventRecord(pack_end);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in pack_int4_kernel_B_fp16 (activation): %s\n", cudaGetErrorString(err));
    }

    // 2) Perform Int4 GEMM.
    cudaEventRecord(gemm_start);
    int lda = padded_virtualK * 2;  
    int ldb = padded_virtualK * 2;  
    auto C_int32 = torch::zeros({M, N}, torch::dtype(torch::kInt32).device(act_fp.device()));
    int ldc = N;
    err = CutlassInt4Gemm(
        M, N, K,
        reinterpret_cast<const cutlass::int4b_t*>(packed_weight_int8.data_ptr<int8_t>()),
        lda,
        reinterpret_cast<const cutlass::int4b_t*>(quantized_act.data_ptr<int8_t>()),
        ldb,
        C_int32.data_ptr<int32_t>(),
        ldc);
    cudaEventRecord(gemm_end);
    if (err != cudaSuccess) {
        printf("CUTLASS int4 GEMM failed (packedW version): %s\n", cudaGetErrorString(err));
        return;
    }

    // 3) Scale accumulations back to FP32.
    cudaEventRecord(scale_start);
    dim3 blockDim2(16, 16);
    dim3 gridDim2((N + blockDim2.x - 1) / blockDim2.x, (M + blockDim2.y - 1) / blockDim2.y);
    scale_inlier_kernel<<<gridDim2, blockDim2>>>(
        C_int32.data_ptr<int32_t>(),
        weight_scale.data_ptr<float>(),
        act_scale.data_ptr<float>(),
        M, N,
        output.data_ptr<float>());
    cudaEventRecord(scale_end);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in scale_inlier_kernel (packedW version): %s\n", cudaGetErrorString(err));
    }

    // 4) Quantize and accumulate outliers.
    cudaEventRecord(outlier_start);
    int nnz = act_outlier.numel();
    if (nnz > 0) {
        auto quantized_outlier = torch::empty_like(act_outlier, torch::dtype(torch::kInt8).device(act_outlier.device()));
        int threads_out = 256;
        int blocks_out = (nnz + threads_out - 1) / threads_out;
        quantize_outliers_kernel<<<blocks_out, threads_out>>>(
            reinterpret_cast<const __half*>(act_outlier.data_ptr<at::Half>()),
            act_outlier_indices.data_ptr<int64_t>(),
            act_outlier_scale.data_ptr<float>(),
            nnz,
            n_lva_o,
            quantized_outlier.data_ptr<int8_t>());
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in quantize_outliers_kernel (outliers, packedW): %s\n", cudaGetErrorString(err));
        }
        // Assuming act_outlier_indices is a [2, nnz] tensor:
        const int64_t* row_indices = act_outlier_indices.data_ptr<int64_t>();
        const int64_t* col_indices = row_indices + act_outlier_indices.size(1);
        accumulate_outliers_kernel<<<blocks_out, threads_out>>>(
            quantized_outlier.data_ptr<int8_t>(),
            row_indices,
            col_indices,
            act_outlier_scale.data_ptr<float>(),
            nnz,
            N,
            output.data_ptr<float>());
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in accumulate_outliers_kernel (outliers, packedW): %s\n", cudaGetErrorString(err));
        }
    }
    cudaEventRecord(outlier_end);

    cudaDeviceSynchronize();
    cudaEventRecord(end);
    cudaEventSynchronize(end);

    // Compute elapsed times (in milliseconds).
    float total_time = 0.0f, pack_time = 0.0f, gemm_time = 0.0f, scale_time = 0.0f, outlier_time = 0.0f;
    cudaEventElapsedTime(&total_time, start, end);
    cudaEventElapsedTime(&pack_time, pack_start, pack_end);
    cudaEventElapsedTime(&gemm_time, gemm_start, gemm_end);
    cudaEventElapsedTime(&scale_time, scale_start, scale_end);
    cudaEventElapsedTime(&outlier_time, outlier_start, outlier_end);

    printf("Fused Quantized GEMM (packedW) timing (ms):\n");
    printf("  Total time:           %f\n", total_time);
    printf("  Activation packing:   %f (%.2f%%)\n", pack_time, (pack_time / total_time) * 100.0f);
    printf("  GEMM:                 %f (%.2f%%)\n", gemm_time, (gemm_time / total_time) * 100.0f);
    printf("  Scaling:              %f (%.2f%%)\n", scale_time, (scale_time / total_time) * 100.0f);
    printf("  Outlier processing:   %f (%.2f%%)\n", outlier_time, (outlier_time / total_time) * 100.0f);

    // Clean up events.
    cudaEventDestroy(start);
    cudaEventDestroy(end);
    cudaEventDestroy(pack_start);
    cudaEventDestroy(pack_end);
    cudaEventDestroy(gemm_start);
    cudaEventDestroy(gemm_end);
    cudaEventDestroy(scale_start);
    cudaEventDestroy(scale_end);
    cudaEventDestroy(outlier_start);
    cudaEventDestroy(outlier_end);
}


//--------------------------------------------------------------
// unpack kernel for FP16 matrices in row-major order.
// Added parameter 'physical_stride' to use the padded row width.
__global__ void unpack_int4_kernel_fp16_rowmajor(
    const int8_t* __restrict__ quantized,
    const float* __restrict__ scale,
    int total_elements,
    int cols,
    int physical_stride,   // padded row length (in int8 elements)
    __half* __restrict__ output)
{
    int total_pairs = total_elements / 2;
    int valid_pairs_per_row = cols / 2;
    int pair_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int index0 = pair_idx * 2;
    if (pair_idx < total_pairs) {
        int row = index0 / cols;
        int pair_in_row = (index0 % cols) / 2;
        if (pair_in_row < valid_pairs_per_row) {
            int8_t packed = quantized[row * physical_stride + pair_in_row];
            int nibble0 = packed & 0xF;
            int q0 = (nibble0 >= 8) ? nibble0 - 16 : nibble0;
            if (index0 < total_elements) {
                float s = scale[row];
                float val = q0 * s;
                output[index0] = __float2half(val);
            }
            int nibble1 = (packed >> 4) & 0xF;
            int q1 = (nibble1 >= 8) ? nibble1 - 16 : nibble1;
            int index1 = index0 + 1;
            if (index1 < total_elements) {
                float s = scale[row];
                float val = q1 * s;
                output[index1] = __float2half(val);
            }
        }
    }
}

//--------------------------------------------------------------
// dequantize kernel for outliers.
// Now the kernel takes separate pointers for row and column indices.
__global__ void dequantize_outliers_kernel(
    const int8_t* __restrict__ quantized_outlier,
    const int64_t* __restrict__ row_indices,
    const int64_t* __restrict__ col_indices,
    const float* __restrict__ s_outlier,  
    int nnz,
    int D,                              // number of FP16 elements per row
    __half* __restrict__ output)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nnz) {
        int row = row_indices[idx];
        int col = col_indices[idx];
        int8_t q = quantized_outlier[idx];
        float scale = s_outlier[idx];
        float dequant = ((int)q) * scale;
        output[row * D + col] = __float2half(dequant);
    }
}

//--------------------------------------------------------------
// Full quantization kernel (inlier + outlier).
void quantize_cuda(
    torch::Tensor inlier,
    torch::Tensor static_s,
    int qmin,
    int qmax,
    torch::Tensor quantized_inlier,
    torch::Tensor outlier_values,
    torch::Tensor outlier_indices,  // remains unchanged (for debug prints)
    torch::Tensor s_outlier,
    int n_lva_o,
    torch::Tensor quantized_outlier)
{
    int total_elements = inlier.numel();
    int D = inlier.size(1);
    int threads = 256;
    int num_pairs = total_elements / 2;
    int blocks = (num_pairs + threads - 1) / threads;
    int padded_virtualD = ((D + 15) / 16) * 16;  // padded physical row length in int8 elements
    pack_int4_kernel_fp16_rowmajor<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(inlier.data_ptr<at::Half>()),
        static_s.data_ptr<float>(),
        total_elements,
        D,
        padded_virtualD,
        quantized_inlier.data_ptr<int8_t>());
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in quantize_cuda (pack_int4_kernel_fp16_rowmajor): %s\n", cudaGetErrorString(err));
    }
    int nnz = outlier_values.numel();
    threads = 256;
    blocks = (nnz + threads - 1) / threads;
    quantize_outliers_kernel<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(outlier_values.data_ptr<at::Half>()),
        outlier_indices.data_ptr<int64_t>(),
        s_outlier.data_ptr<float>(),
        nnz,
        n_lva_o,
        quantized_outlier.data_ptr<int8_t>());
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in quantize_cuda (quantize_outliers_kernel): %s\n", cudaGetErrorString(err));
    }
    cudaDeviceSynchronize();
}

//--------------------------------------------------------------
// Full dequantization kernel (inlier + outlier).
// Updated dequantize_cuda in your CUDA file.
void dequantize_cuda(
    torch::Tensor quantized_inlier,
    torch::Tensor static_s,
    int B_L,
    int D,
    torch::Tensor quantized_outlier,
    torch::Tensor outlier_indices,
    torch::Tensor s_outlier,
    int nnz,
    torch::Tensor output)
{
    int total_elements = B_L * D;
    int threads = 256;
    int num_pairs = total_elements / 2;
    int blocks = (num_pairs + threads - 1) / threads;
    int padded_virtualD = ((D + 15) / 16) * 16;
    // Dequantize inliers.
    unpack_int4_kernel_fp16_rowmajor<<<blocks, threads>>>(
        quantized_inlier.data_ptr<int8_t>(),
        static_s.data_ptr<float>(),
        total_elements,
        D,
        padded_virtualD,
        reinterpret_cast<__half*>(output.data_ptr<at::Half>())
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Error in dequantize_cuda (unpack_int4_kernel_fp16_rowmajor): %s\n", cudaGetErrorString(err));
    }

    // Now dequantize outliers.
    if (nnz > 0) {
        // Ensure outlier_indices is contiguous.
        torch::Tensor out_idx = outlier_indices.contiguous();
        // Extract row and column pointers.
        const int64_t* row_ptr = out_idx.data_ptr<int64_t>(); 
        const int64_t* col_ptr = row_ptr + out_idx.size(1);  // shape is [2, nnz]

        int threads_out = 256;
        int blocks_out = (nnz + threads_out - 1) / threads_out;
        dequantize_outliers_kernel<<<blocks_out, threads_out>>>(
            quantized_outlier.data_ptr<int8_t>(),
            row_ptr,
            col_ptr,
            s_outlier.data_ptr<float>(),
            nnz,
            D,  
            reinterpret_cast<__half*>(output.data_ptr<at::Half>())
        );
        err = cudaGetLastError();
        if (err != cudaSuccess) {
            printf("Error in dequantize_cuda (dequantize_outliers_kernel): %s\n", cudaGetErrorString(err));
        }
    }
    cudaDeviceSynchronize();
}


//--------------------------------------------------------------
// PYBIND11 module.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("fused_quantized_gemm_cuda", 
          &fused_quantized_gemm_cuda, 
          "Fused GEMM with on-device FP16->int4 quantization (weights + activation + outlier accumulation).");

    m.def("fused_quantized_gemm_packedW_cuda", 
          &fused_quantized_gemm_packedW_cuda,
          "Fused GEMM where weight is already packed int4, activation is packed on-device, with outlier accumulation.");
    
    m.def("fused_quantized_gemm_packedW_instrumented_cuda",
         &fused_quantized_gemm_packedW_instrumented_cuda,
         "Fused GEMM where weight is already packed int4, with instrumentation."
    );

    m.def("pack_int4_kernel_fp16_rowmajor_cuda",
          &pack_int4_kernel_fp16_rowmajor_host,
          "Launcher for pack_int4_kernel_fp16_rowmajor kernel.");
    
    m.def("quantize_cuda", &quantize_cuda, "Quantization CUDA kernel (full) for testing (FP16 input)");
    m.def("dequantize_cuda", &dequantize_cuda, "Dequantization CUDA kernel (full) for testing (FP16 input)");
}
