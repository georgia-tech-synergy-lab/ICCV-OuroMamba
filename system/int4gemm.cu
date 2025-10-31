#include <torch/extension.h>
#include <torch/torch.h>
#include <ATen/ATen.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <stdexcept>
#include <cuda_fp16.h>  // for __half and __half2float

// CUTLASS includes
#include "cutlass/cutlass.h"
#include "cutlass/half.h"
#include "cutlass/gemm/device/gemm.h"
#include "cutlass/layout/matrix.h"

//--------------------------------------------------------------
// CUDA kernel to pack A (row-major) from fp16.
// Each output element packs two fp16 values (after rounding/clipping)
// into a single int8, where each nibble holds one 4-bit value.
__global__ void pack_int4_kernel_A_fp16(const __half *input, int8_t *output, int M, int virtual_K) {
  int row = blockIdx.y * blockDim.y + threadIdx.y;
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // col index in packed output (0 <= col < virtual_K/2)
  int packedK = virtual_K / 2;
  if (row < M && col < packedK) {
    int idx0 = row * virtual_K + (col * 2);
    int idx1 = idx0 + 1;
    // Convert fp16 to float then round to nearest int.
    int a = __float2int_rn(__half2float(input[idx0]));
    int b = __float2int_rn(__half2float(input[idx1]));
    // Clip to range [-8,7]
    a = max(-8, min(a, 7));
    b = max(-8, min(b, 7));
    // Convert negatives to 4-bit two's complement representation.
    if (a < 0) a += 16;
    if (b < 0) b += 16;
    output[row * packedK + col] = static_cast<int8_t>((b << 4) | (a & 0x0F));
  }
}

//--------------------------------------------------------------
// CUDA kernel to pack B (from fp16) along the first dimension.
// This kernel writes the output in column-major order.
// For each column of B, it packs pairs of fp16 values.
__global__ void pack_int4_kernel_B_fp16(const __half *input, int8_t *output, int virtual_K, int N) {
  int col = blockIdx.x * blockDim.x + threadIdx.x;  // column index in B (0 <= col < N)
  int row = blockIdx.y * blockDim.y + threadIdx.y;  // row index in packed output (0 <= row < virtual_K/2)
  int packedK = virtual_K / 2;
  if (col < N && row < packedK) {
    int idx0 = (row * 2) * N + col;
    int idx1 = idx0 + N;
    int a = __float2int_rn(__half2float(input[idx0]));
    int b = __float2int_rn(__half2float(input[idx1]));
    a = max(-8, min(a, 7));
    b = max(-8, min(b, 7));
    if (a < 0) a += 16;
    if (b < 0) b += 16;
    // Write output in column-major order.
    output[col * packedK + row] = static_cast<int8_t>((b << 4) | (a & 0x0F));
  }
}

//--------------------------------------------------------------
// CUTLASS int4 GEMM using packed inputs.
// "virtual_K" is the original number of int4 (unpacked) values per row.
//
// NOTE: We keep int32 accumulators, but return final output in half.
//       So ElementOutput = cutlass::half_t, while ElementAccumulator = int32_t.
cudaError_t CutlassInt4Gemm(
    int M,
    int N,
    int virtual_K,
    const cutlass::int4b_t *A,
    int lda,
    const cutlass::int4b_t *B,
    int ldb,
    cutlass::half_t *C,         // Now a pointer to half-precision
    int ldc,
    cutlass::half_t alpha,      // half alpha
    cutlass::half_t beta        // half beta
) {
  using ElementAccumulator = int32_t;             // accumulators in int32
  using ElementComputeEpilogue = cutlass::half_t; // epilogue scaling in half

  using ElementInputA = cutlass::int4b_t;
  using ElementInputB = cutlass::int4b_t;
  using ElementOutput = cutlass::half_t;          // final output in half

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

//--------------------------------------------------------------
// PyTorch wrapper for int4 GEMM.
// A and B are fp16 tensors with shapes: A: [M, virtual_K], B: [virtual_K, N].
// The kernel packs them on-device (with rounding/clipping) before GEMM.
// Final result is returned as half.
torch::Tensor int4_gemm(
    torch::Tensor A,
    torch::Tensor B,
    int virtual_K,
    float alpha_float = 1.0f,   // Expose alpha as float from Python
    float beta_float  = 0.0f    // Expose beta as float from Python
) {
  // Check inputs
  TORCH_CHECK(A.is_cuda(), "A must be a CUDA tensor");
  TORCH_CHECK(B.is_cuda(), "B must be a CUDA tensor");
  TORCH_CHECK(A.is_contiguous(), "A must be contiguous");
  TORCH_CHECK(B.is_contiguous(), "B must be contiguous");
  TORCH_CHECK(A.dtype() == torch::kHalf, "A must be of type fp16 (torch.half)");
  TORCH_CHECK(B.dtype() == torch::kHalf, "B must be of type fp16 (torch.half)");

  // Extract dimensions
  int M = A.size(0);
  int K = A.size(1);
  int N = B.size(1);
  TORCH_CHECK(K == virtual_K, "A's second dimension must equal virtual_K");
  TORCH_CHECK(B.size(0) == virtual_K, "B's first dimension must equal virtual_K");
  TORCH_CHECK((virtual_K % 2) == 0, "virtual_K must be even for int4 packing");

  int packedK = virtual_K / 2;
  auto options_int8 = torch::TensorOptions().dtype(torch::kInt8).device(A.device());

  // Allocate temporary packed tensors.
  // A_packed shape: [M, packedK] (row-major)
  torch::Tensor A_packed = torch::empty({M, packedK}, options_int8);

  // B_packed shape: [N, packedK] (to be interpreted as column-major)
  torch::Tensor B_packed = torch::empty({N, packedK}, options_int8);

  // Create CUDA events for timing (optional).
  cudaEvent_t event_start, event_pack_end, event_stop;
  cudaEventCreate(&event_start);
  cudaEventCreate(&event_pack_end);
  cudaEventCreate(&event_stop);

  // Record event before packing.
  cudaEventRecord(event_start);

  // Launch packing kernels.
  dim3 blockDim(16, 16);

  // For A
  dim3 gridDim_A((packedK + blockDim.x - 1) / blockDim.x,
                 (M + blockDim.y - 1) / blockDim.y);

  pack_int4_kernel_A_fp16<<<gridDim_A, blockDim>>>(
      reinterpret_cast<const __half*>(A.data_ptr<at::Half>()),
      A_packed.data_ptr<int8_t>(),
      M,
      virtual_K
  );

  // For B
  dim3 gridDim_B((N + blockDim.x - 1) / blockDim.x,
                 (packedK + blockDim.y - 1) / blockDim.y);

  pack_int4_kernel_B_fp16<<<gridDim_B, blockDim>>>(
      reinterpret_cast<const __half*>(B.data_ptr<at::Half>()),
      B_packed.data_ptr<int8_t>(),
      virtual_K,
      N
  );

  cudaError_t err = cudaGetLastError();
  TORCH_CHECK(err == cudaSuccess, "Kernel launch error: ", cudaGetErrorString(err));
  err = cudaDeviceSynchronize();
  TORCH_CHECK(err == cudaSuccess, "Kernel execution error: ", cudaGetErrorString(err));

  // Record event after packing.
  cudaEventRecord(event_pack_end);

  // Allocate output tensor as half
  auto options_half = torch::TensorOptions().dtype(torch::kHalf).device(A.device());
  torch::Tensor C = torch::empty({M, N}, options_half);

  // LDA, LDB, LDC: measured in number of elements
  int lda = virtual_K;  // (A is row-major, each row has virtual_K int4 elements)
  int ldb = virtual_K;  // (B is column-major, each column has virtual_K int4 elements)
  int ldc = N;          // (C is row-major, each row has N half outputs)

  // Convert alpha/beta from float to half
  cutlass::half_t alpha_half(alpha_float);
  cutlass::half_t beta_half(beta_float);

  // Call CUTLASS GEMM with half output
  err = CutlassInt4Gemm(
      M,
      N,
      virtual_K,
      reinterpret_cast<cutlass::int4b_t*>(A_packed.data_ptr<int8_t>()),
      lda,
      reinterpret_cast<cutlass::int4b_t*>(B_packed.data_ptr<int8_t>()),
      ldb,
      reinterpret_cast<cutlass::half_t*>(C.data_ptr<at::Half>()),
      ldc,
      alpha_half,
      beta_half
  );
  if (err != cudaSuccess) {
    throw std::runtime_error("CUTLASS int4 GEMM kernel failed");
  }

  // Record event after GEMM.
  cudaEventRecord(event_stop);
  cudaEventSynchronize(event_stop);

  // Compute elapsed times in milliseconds (optional).
  float pack_time = 0, gemm_time = 0;
  cudaEventElapsedTime(&pack_time, event_start, event_pack_end);
  cudaEventElapsedTime(&gemm_time, event_pack_end, event_stop);
  float total_time = pack_time + gemm_time;

  // Print timing information (optional).
  printf("Packing time: %f ms (%.2f%%)\n", pack_time, (pack_time / total_time * 100.0f));
  printf("GEMM time: %f ms (%.2f%%)\n", gemm_time, (gemm_time / total_time * 100.0f));

  // Clean up events.
  cudaEventDestroy(event_start);
  cudaEventDestroy(event_pack_end);
  cudaEventDestroy(event_stop);

  // Return the half-precision result
  return C;
}

//--------------------------------------------------------------
// PyTorch Module Registration
//--------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("int4_gemm",
        &int4_gemm,
        "Int4 GEMM using CUTLASS (CUDA) with on-device packing from fp16, "
        "accumulation in int32, and final output in half",
        py::arg("A"),
        py::arg("B"),
        py::arg("virtual_K"),
        py::arg("alpha") = 1.0f,
        py::arg("beta") = 0.0f
  );
}
