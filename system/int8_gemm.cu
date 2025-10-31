#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cmath>
#include <cuda_fp16.h>
#include <vector>
#include <stdio.h>
#include <inttypes.h>
#include <stdexcept>

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/arch/wmma.h"
#include "cutlass/half.h"
#include "cutlass/gemm/device/gemm_universal.h"

//---------------------------------------------------------------------
// Global definition of CUTLASS GEMM operator for int4 (packed) x int8 GEMM.
// Note: A (weight) is stored in int4 (packed into int8) in row-major order,
// and B (activation) is stored as int8 in column-major order.
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

//---------------------------------------------------------------------
// Kernel: Pack FP16 weight matrix (row-major) into int4 format.
// Two FP16 values are quantized (to 4 bits each) and packed into one int8.
__global__ void pack_int4_kernel_fp16_rowmajor(
    const __half* __restrict__ input,   // FP16 input [rows, cols]
    const float scale,                  // FP32 scale factor
    int total_elements,                 // rows * cols (logical FP16 count)
    int cols,                           // number of FP16 values per row
    int physical_stride,                // padded row length (in int8 elements), >= cols/2
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

// Host launcher for the pack_int4 kernel.
void pack_int4_kernel_fp16_rowmajor_host(
    torch::Tensor input_fp16,   // FP16 [rows, cols]
    float scale,                // FP32 scalar scale factor
    int rows,
    int cols,
    int padded_virtualK,        // padded physical stride (in int8 elements)
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

//---------------------------------------------------------------------
// Kernel: Pack FP16 activation matrix (B) from row-major FP16 into int8,
// storing the result in column-major order. Each FP16 value is quantized using
// the provided scale factor.
__global__ void pack_int8_kernel_B_fp16_colmajor(
    const __half* __restrict__ input,   // FP16 input in row-major order [rows, cols]
    float scale,                        // FP32 scale factor
    int rows,                           // number of rows (K dimension)
    int cols,                           // number of columns (N dimension)
    int physical_stride,                // padded number of rows (>= rows) for column-major storage
    int8_t* __restrict__ output)        // output activation matrix in int8, column-major order
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = rows * cols;
    if (idx < total_elements) {
        int k = idx / cols;
        int n = idx % cols;
        float x = __half2float(input[k * cols + n]);
        int q = roundf(x / scale);
        q = max(-127, min(q, 127));
        output[n * physical_stride + k] = static_cast<int8_t>(q);
    }
}

//---------------------------------------------------------------------
// Function to perform int4 (A) x int8 (B) GEMM with int32 output using CUTLASS.
// A_int4: Packed weight matrix in int4 format (stored as int8), row-major, with shape [M, padded_K],
//         where padded_K*2 >= logical K.
// B_fp16: Activation matrix in FP16, row-major, with shape [K, N] (logical dimensions).
// M, N, K: Logical GEMM dimensions (A is MxK, B is KxN).
// scale_B: Scale factor used to quantize B to int8.
torch::Tensor int4_int8_gemm(
    torch::Tensor A_int4,  // Packed int4 weight (stored as int8)
    torch::Tensor B_fp16,  // FP16 activation matrix (row-major)
    int M, int N, int K,   // GEMM dimensions: A: M x K, B: K x N
    float scale_B          // Scale factor for quantizing B to int8
)
{
    // Check that inputs are CUDA tensors.
    TORCH_CHECK(A_int4.is_cuda(), "A_int4 must be a CUDA tensor");
    TORCH_CHECK(B_fp16.is_cuda(), "B_fp16 must be a CUDA tensor");

    // Check data types.
    TORCH_CHECK(A_int4.scalar_type() == torch::kInt8, "A_int4 must be int8");
    TORCH_CHECK(B_fp16.scalar_type() == at::kHalf, "B_fp16 must be half (fp16)");

    // Expect A_int4 to be 2D: [M, padded_K] (padded_K*2 >= K).
    TORCH_CHECK(A_int4.dim() == 2, "A_int4 must be a 2D tensor");
    // Expect B_fp16 to be 2D: [K, N] (row-major).
    TORCH_CHECK(B_fp16.dim() == 2, "B_fp16 must be a 2D tensor");

    // Allocate output tensor C (int32) of shape [M, N] (row-major).
    auto options = torch::TensorOptions().dtype(torch::kInt32).device(B_fp16.device());
    torch::Tensor C_int32 = torch::empty({M, N}, options);

    // Allocate temporary tensor for packed B (int8) in column-major order.
    // We use physical_stride = K (no extra padding).
    auto options_int8 = torch::TensorOptions().dtype(torch::kInt8).device(B_fp16.device());
    torch::Tensor B_int8 = torch::empty({K, N}, options_int8);

    // Launch kernel to pack B from FP16 to int8.
    int total_elements = K * N;
    int threads = 256;
    int blocks = (total_elements + threads - 1) / threads;
    pack_int8_kernel_B_fp16_colmajor<<<blocks, threads>>>(
        reinterpret_cast<const __half*>(B_fp16.data_ptr<at::Half>()),
        scale_B,
        K,
        N,
        K,   // physical_stride = K
        B_int8.data_ptr<int8_t>()
    );
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[pack_int8_kernel_B_fp16_colmajor] CUDA error: %s\n", cudaGetErrorString(err));
    }

    // Compute leading dimensions.
    // For A: A_int4 is stored as int8 where each element packs 2 int4 values.
    int padded_K = A_int4.size(1);
    int lda = padded_K * 2;  // effective number of int4 elements per row

    // For B (packed in column-major), the leading dimension is K.
    int ldb = K;
    // For C (row-major), the leading dimension is N.
    int ldc = N;

    // Set alpha = 1 and beta = 0.
    cutlass::half_t alpha(1);
    cutlass::half_t  beta(0);

    // Build CUTLASS GEMM arguments.
    typename GemmInt4Int8::Arguments args(
        cutlass::gemm::GemmUniversalMode::kGemm,         // mode
        cutlass::gemm::GemmCoord{M, N, K},                 // problem size
        1,                                               // split-k slices
        {alpha, beta},                                   // epilogue output op parameters
        reinterpret_cast<const void*>(A_int4.data_ptr<int8_t>()), // pointer A
        reinterpret_cast<const void*>(B_int8.data_ptr<int8_t>()), // pointer B
        reinterpret_cast<const void*>(C_int32.data_ptr<int32_t>()),// pointer C
        reinterpret_cast<void*>(C_int32.data_ptr<int32_t>()),       // pointer D (output)
        /* Strides: */
        static_cast<int64_t>(lda),  // stride_A
        static_cast<int64_t>(ldb),  // stride_B
        static_cast<int64_t>(ldc),  // stride_C
        static_cast<int64_t>(ldc),  // stride_D
        /* Layout-specific strides: */
        static_cast<int64_t>(lda),  // A layout stride
        static_cast<int64_t>(ldb),  // B layout stride
        static_cast<int64_t>(ldc),  // C layout stride
        static_cast<int64_t>(ldc),  // D layout stride
        /* Extra pointers for split-K reduction (unused): */
        nullptr, nullptr, nullptr
    );

    // Instantiate the GEMM operator.
    GemmInt4Int8 gemm_op;

    // Run CUTLASS GEMM.
    cutlass::Status status = gemm_op(args);
    if (status != cutlass::Status::kSuccess) {
        throw std::runtime_error("CUTLASS GEMM failed");
    }

    return C_int32;
}

//---------------------------------------------------------------------
// PyBind11 binding code.
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("pack_int4_kernel_fp16_rowmajor", &pack_int4_kernel_fp16_rowmajor_host,
          "Pack FP16 matrix (row-major) into int4 format");
    m.def("int4_int8_gemm", &int4_int8_gemm,
          "Perform int4 (packed) x int8 GEMM (C = A * B) using CUTLASS on SM80");
}
