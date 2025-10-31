/***************************************************************************************************
 * Copyright (c) 2017 - 2025 NVIDIA CORPORATION & AFFILIATES.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted
 * provided that the following conditions are met:
 *
 *   1. Redistributions of source code must retain the above copyright notice, this list of
 *      conditions and the following disclaimer.
 *   2. Redistributions in binary form must reproduce the above copyright notice, this list of
 *      conditions and the following disclaimer in the documentation and/or other materials
 *      provided with the distribution.
 *   3. Neither the name of the copyright holder nor the names of its contributors may be used
 *      to endorse or promote products derived from this software without specific prior written
 *      permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR
 * IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR
 * CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
 * WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY
 * WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/*! \file
    \brief Example: Minimal GEMM kernel using a custom epilogue visitor in CUTLASS 3.7.0.

    This shows how to supply all template parameters to EpilogueWithVisitor,
    define a "toy" Mma that the compiler recognizes (so it can find
    WarpMmaOperator::IteratorC, etc.), and define a custom visitor with a
    nested OutputTileIterator.
*/

#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/matrix_coord.h"
#include "cutlass/semaphore.h"
#include "cutlass/trace.h"

#include "cutlass/epilogue/threadblock/epilogue_with_visitor.h"
#include "cutlass/epilogue/threadblock/shared_load_iterator.h"

/////////////////////////////////////////////////////////////////////////////////////////////////
// A minimal "toy" Mma type just so we can demonstrate the structure. In real code,
// you would replace this with your int4-based or half-based Mma (threadblock-scoped
// matrix multiply) from CUTLASS.
/////////////////////////////////////////////////////////////////////////////////////////////////
struct ToyMma {

  /// The shape of the threadblock tile (M, N, K)
  using Shape = cutlass::gemm::GemmShape<128, 128, 8>;

  /// Warps per block, etc. (just a placeholder)
  struct WarpCount {
    static int const kM = 2;
    static int const kN = 2;
    static int const kK = 1;
    static int const kCount = kM * kN * kK;
  };

  /// The accumulator type (pretend we're using half accumulators).
  using ElementC = cutlass::half_t;

  /// Example "IteratorA". In a real kernel, you'd have a specialized iterator
  /// for row-major or column-major int4/half data. Here we just define layout.
  struct IteratorA {
    using Element = cutlass::half_t;
    using Layout  = cutlass::layout::RowMajor;
    /// Typically you'd have an AccessType sub-type here, but let's pretend:
    struct AccessType { static int const kElements = 1; };

    /// For constructing in device code
    struct Params {
      CUTLASS_HOST_DEVICE
      Params(Layout const &layout) : layout_(layout) { }
      Layout layout_;
    };

    CUTLASS_DEVICE
    IteratorA(Params const &p, Element *ptr, cutlass::MatrixCoord extent,
              int thread_idx, cutlass::MatrixCoord tb_offset) {
      // Not implemented, just a stub
    }
  };

  /// Example "IteratorB"
  struct IteratorB {
    using Element = cutlass::half_t;
    using Layout  = cutlass::layout::RowMajor;
    struct AccessType { static int const kElements = 1; };

    struct Params {
      CUTLASS_HOST_DEVICE
      Params(Layout const &layout) : layout_(layout) { }
      Layout layout_;
    };

    CUTLASS_DEVICE
    IteratorB(Params const &p, Element *ptr, cutlass::MatrixCoord extent,
              int thread_idx, cutlass::MatrixCoord tb_offset) {
      // Not implemented, just a stub
    }
  };

  /// A "warp operator" used for partial tile iteration, etc.
  struct Operator {
    /// For example, shape of each warp-level MMA
    using Shape = cutlass::gemm::GemmShape<64,64,8>;
    /// Just a placeholder
    struct Policy { struct Operator { using InstructionShape = cutlass::gemm::GemmShape<16,8,8>; }; };
  };

  /// The "Policy" for Mma. Typically includes the warp tile layout, accumulator
  /// iteration, etc. For demonstration, we define minimal stubs so we can
  /// reference them in Epilogue.
  struct Policy {
    /// Type used to iterate over accumulators in the epilogue
    struct AccumulatorFragmentIterator {};

    /// Type used to iterate over warp tile in the epilogue
    struct WarpTileIterator {};

    /// Lane-level shape
    struct LaneMmaShape {};
  };

  /// Shared storage for the mainloop
  union SharedStorage { };

  /// Type of accumulator fragment
  using FragmentC = cutlass::Array<ElementC, 32>; // for example

  /// Number of pipeline stages
  static int const kStages = 2;

  CUTLASS_DEVICE
  ToyMma(SharedStorage &shared_storage, int thread_idx, int warp_idx, int lane_idx) { }

  /// Perform the mainloop
  CUTLASS_DEVICE
  void operator()(
    int gemm_k_iterations,
    FragmentC &accum,
    IteratorA &iteratorA,
    IteratorB &iteratorB,
    FragmentC const &src_accum)
  {
    // pretend we're doing something; just zero the accum
    for (int i = 0; i < int(accum.size()); ++i) {
      accum[i] = cutlass::half_t(0);
    }
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
// A minimal "threadblock swizzle" that just uses identity mapping. In real code, you might use
// cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle or something else.
/////////////////////////////////////////////////////////////////////////////////////////////////
struct ToyThreadblockSwizzle {

  /// Compute the tiled shape: how many threadblocks (tile of MxN) fit into MxN problem_size
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord get_tiled_shape(
    cutlass::gemm::GemmCoord problem_size,
    cutlass::gemm::GemmCoord block_shape,
    int batch_count = 1) const
  {
    int grid_m = (problem_size.m() + block_shape.m() - 1) / block_shape.m();
    int grid_n = (problem_size.n() + block_shape.n() - 1) / block_shape.n();
    int grid_k = batch_count; // or for split-k, etc.
    return {grid_m, grid_n, grid_k};
  }

  /// Return the tile offset for a given block index
  CUTLASS_HOST_DEVICE
  cutlass::gemm::GemmCoord get_tile_offset(int /*log_tile*/) const {
    // Identity: use blockIdx.x => N dimension, blockIdx.y => M dimension, blockIdx.z => K dimension
    return cutlass::gemm::GemmCoord(blockIdx.y, blockIdx.x, blockIdx.z);
  }

  CUTLASS_HOST_DEVICE
  int get_log_tile(cutlass::gemm::GemmCoord /*grid_tiled_shape*/) const {
    return 0; // not used in this toy example
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// A custom visitor that does row/column scaling or partial reduction. For demonstration only.
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  typename ThreadblockShape_,
  typename ElementAccumulator_,
  typename ElementOutput_
>
struct RowColScaleVisitor {

  using ThreadblockShape   = ThreadblockShape_;
  using ElementAccumulator = ElementAccumulator_;
  using ElementOutput      = ElementOutput_;

  /// Suppose we also want to track "max" or "sum" in separate buffers
  using ElementNorm = float;
  using ElementSum  = float;

  /// Number of elements each thread processes per access
  static int const kElementsPerAccess = 2; // example

  /// Fragment type from the mainloop
  using FragmentOutput = cutlass::Array<ElementAccumulator, kElementsPerAccess>;

  /// The tile iterator we must define for epilogue usage
  struct OutputTileIterator {
    struct Params {
      int ldm;
      CUTLASS_HOST_DEVICE
      Params() : ldm(0) {}
      CUTLASS_HOST_DEVICE
      Params(cutlass::layout::RowMajor const &layout) : ldm(int(layout.stride(0))) {}
    };
    // Real code would define store() / operator++ etc. as needed
  };

  /// Shared storage (if needed)
  struct SharedStorage {
    // empty for now
  };

  /// Arguments
  struct Arguments {
    ElementNorm *ptr_Max;
    ElementSum  *ptr_Sum;

    Arguments() : ptr_Max(nullptr), ptr_Sum(nullptr) {}
    Arguments(ElementNorm *pMax, ElementSum *pSum) : ptr_Max(pMax), ptr_Sum(pSum) {}
  };

  /// Params (device-side)
  struct Params {
    ElementNorm *ptr_Max;
    ElementSum  *ptr_Sum;
    CUTLASS_HOST_DEVICE
    Params() : ptr_Max(nullptr), ptr_Sum(nullptr) {}
    CUTLASS_HOST_DEVICE
    Params(Arguments const &args) : ptr_Max(args.ptr_Max), ptr_Sum(args.ptr_Sum) {}
  };

private:
  Params &params_;
  SharedStorage &shared_storage_;
  int extent_;
  int thread_idx_, warp_idx_, lane_idx_;
  typename OutputTileIterator::Params params_C_, params_D_;
  ElementOutput *ptr_C_, *ptr_D_;
  cutlass::MatrixCoord threadblock_offset_;
  int block_offset_;

public:
  /// Constructor
  CUTLASS_DEVICE
  RowColScaleVisitor(
    Params const &params,
    SharedStorage &shared_storage,
    int extent,
    int thread_idx,
    int warp_idx,
    int lane_idx,
    typename OutputTileIterator::Params params_C,
    typename OutputTileIterator::Params params_D,
    ElementOutput *ptr_C,
    ElementOutput *ptr_D,
    ElementNorm *ptr_Max,
    ElementSum  *ptr_Sum,
    cutlass::MatrixCoord threadblock_offset,
    int block_offset
  )
  : params_(const_cast<Params &>(params)),  // careful: in practice you'd store by value or so
    shared_storage_(shared_storage),
    extent_(extent),
    thread_idx_(thread_idx),
    warp_idx_(warp_idx),
    lane_idx_(lane_idx),
    params_C_(params_C),
    params_D_(params_D),
    ptr_C_(ptr_C),
    ptr_D_(ptr_D),
    threadblock_offset_(threadblock_offset),
    block_offset_(block_offset)
  {
    if (ptr_Max) { params_.ptr_Max = ptr_Max; }
    if (ptr_Sum) { params_.ptr_Sum = ptr_Sum; }
  }

  CUTLASS_DEVICE
  void set_k_partition(int partition_idx, int partition_count) {
    // handle partial reduction across K if needed
  }

  CUTLASS_DEVICE
  void set_batch_index(int batch_idx) {
    // handle batched array index if needed
  }

  /// Called by the epilogue for each iteration of the accumulator tile
  template <typename AccumulatorFragment>
  CUTLASS_DEVICE
  void visit(AccumulatorFragment const &frag, int idx_in_tile) {
    // Example: do row/column scaling, partial store, etc.
    // In real code, you'd compute the offset from idx_in_tile, write `frag` to
    // ptr_C_ or ptr_D_, etc. Possibly update ptr_Max/ptr_Sum.
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// We define an EpilogueWithVisitor type for our "RowColScaleVisitor".
// CUTLASS 3.7.0's EpilogueWithVisitor expects up to EIGHT template parameters.
// We must provide each carefully.
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  typename ThreadblockShape,
  typename WarpMmaOperator,
  int PartitionsK,
  typename AccumulatorFragmentIterator,
  typename WarpTileIterator,
  typename Visitor
>
using EpilogueWithRowColScale = cutlass::epilogue::threadblock::EpilogueWithVisitor<
  ThreadblockShape,
  WarpMmaOperator,
  PartitionsK,
  AccumulatorFragmentIterator,
  WarpTileIterator,
  Visitor,
  /// Default SharedLoadIterator:
  cutlass::epilogue::threadblock::SharedLoadIterator<
    // If you have a specific InstructionShape or alignment, specify them:
    typename WarpMmaOperator::Shape,       // warp shape
    typename Visitor::ElementAccumulator,  // what we're loading from shared?
    Visitor::kElementsPerAccess
  >,
  /// Default Padding:
  cutlass::epilogue::threadblock::EpilogueWithVisitorPadding
>;

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// Finally, the GEMM kernel with the custom visitor-based epilogue.
//
/////////////////////////////////////////////////////////////////////////////////////////////////
template <
  typename Mma_,                  // Our "ToyMma" or real Mma
  typename ThreadblockSwizzle_    // Our "ToyThreadblockSwizzle" or real swizzle
>
struct GemmWithRowColScaleVisitor {

  using Mma = Mma_;
  using ThreadblockSwizzle = ThreadblockSwizzle_;

  /// The warp operator type from Mma::Operator. Must define:
  /// - Possibly shape, etc. We also rely on Mma::Policy for
  ///   AccumulatorFragmentIterator, WarpTileIterator, LaneMmaShape.
  using WarpMmaOperator = typename Mma::Operator;

  /// For demonstration: We build an "Epilogue" that uses our custom visitor.
  /// Supply all the template parameters that EpilogueWithVisitor demands:
  using Epilogue = EpilogueWithRowColScale<
    typename Mma::Shape,                               // ThreadblockShape
    WarpMmaOperator,                                   // WarpMmaOperator
    /*PartitionsK=*/1,                                 // Typically 1 for non-split-K
    typename Mma::Policy::AccumulatorFragmentIterator, // from Mma policy
    typename Mma::Policy::WarpTileIterator,            // from Mma policy
    RowColScaleVisitor<        // Our visitor type
      typename Mma::Shape,
      typename Mma::ElementC,
      cutlass::half_t          // example output type
    >
  >;

  /// Our visitor is Epilogue::Visitor
  using EpilogueVisitor = typename Epilogue::Visitor;

  /// If your visitor has "ElementNorm" or "ElementSum", expose them
  using ElementNorm = typename EpilogueVisitor::ElementNorm;
  using ElementSum  = typename EpilogueVisitor::ElementSum;

  /// Now define a struct for the kernel's "Arguments"
  struct Arguments {
    cutlass::gemm::GemmUniversalMode mode;
    cutlass::gemm::GemmCoord problem_size;
    int batch_count;

    // Tensors
    cutlass::TensorRef<typename Mma::IteratorA::Element,
                       typename Mma::IteratorA::Layout> ref_A;
    cutlass::TensorRef<typename Mma::IteratorB::Element,
                       typename Mma::IteratorB::Layout> ref_B;

    cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> ref_C;
    cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> ref_D;

    // For your row/col scaling
    ElementNorm *ptr_Max;
    ElementSum  *ptr_Sum;

    int64_t batch_stride_A;
    int64_t batch_stride_B;

    // The visitor's own user-facing Args
    typename EpilogueVisitor::Arguments epilogue_visitor;

    Arguments():
      mode(cutlass::gemm::GemmUniversalMode::kGemm),
      batch_count(1),
      ptr_Max(nullptr),
      ptr_Sum(nullptr),
      batch_stride_A(0),
      batch_stride_B(0)
    { }

    Arguments(
      cutlass::gemm::GemmUniversalMode mode_,
      cutlass::gemm::GemmCoord problem_size_,
      int batch_count_,
      cutlass::TensorRef<typename Mma::IteratorA::Element,
                         typename Mma::IteratorA::Layout> ref_A_,
      cutlass::TensorRef<typename Mma::IteratorB::Element,
                         typename Mma::IteratorB::Layout> ref_B_,
      cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> ref_C_,
      cutlass::TensorRef<cutlass::half_t, cutlass::layout::RowMajor> ref_D_,
      ElementNorm *ptr_Max_,
      ElementSum  *ptr_Sum_,
      int64_t batch_stride_A_,
      int64_t batch_stride_B_,
      typename EpilogueVisitor::Arguments epilogue_visitor_
    )
    : mode(mode_),
      problem_size(problem_size_),
      batch_count(batch_count_),
      ref_A(ref_A_),
      ref_B(ref_B_),
      ref_C(ref_C_),
      ref_D(ref_D_),
      ptr_Max(ptr_Max_),
      ptr_Sum(ptr_Sum_),
      batch_stride_A(batch_stride_A_),
      batch_stride_B(batch_stride_B_),
      epilogue_visitor(epilogue_visitor_)
    { }
  };

  /// The "Params" struct that the kernel actually uses
  struct Params {
    cutlass::gemm::GemmCoord problem_size;
    cutlass::gemm::GemmCoord grid_tiled_shape;
    int swizzle_log_tile;

    typename Mma::IteratorA::Params params_A;
    typename Mma::IteratorB::Params params_B;

    /// Use the visitor's OutputTileIterator for C/D
    typename EpilogueVisitor::OutputTileIterator::Params params_C;
    typename EpilogueVisitor::OutputTileIterator::Params params_D;

    cutlass::gemm::GemmUniversalMode mode;
    int batch_count;
    int gemm_k_size;

    void * ptr_A;
    void * ptr_B;
    cutlass::half_t * ptr_C;
    cutlass::half_t * ptr_D;

    ElementNorm * ptr_Max;
    ElementSum  * ptr_Sum;

    int64_t batch_stride_A;
    int64_t batch_stride_B;

    /// The visitor's Params
    typename EpilogueVisitor::Params epilogue_visitor;

    CUTLASS_HOST_DEVICE
    Params()
    : swizzle_log_tile(0), batch_count(0), gemm_k_size(0),
      mode(cutlass::gemm::GemmUniversalMode::kGemm),
      ptr_A(nullptr), ptr_B(nullptr),
      ptr_C(nullptr), ptr_D(nullptr),
      ptr_Max(nullptr), ptr_Sum(nullptr),
      batch_stride_A(0), batch_stride_B(0)
    { }

    CUTLASS_HOST_DEVICE
    Params(Arguments const &args)
    : problem_size(args.problem_size),
      swizzle_log_tile(0),
      params_A(args.ref_A.layout()),
      params_B(args.ref_B.layout()),
      params_C(args.ref_C.layout()),
      params_D(args.ref_D.layout()),
      mode(args.mode),
      batch_count(args.batch_count),
      gemm_k_size(args.problem_size.k()),
      ptr_A(args.ref_A.data()),
      ptr_B(args.ref_B.data()),
      ptr_C(args.ref_C.data()),
      ptr_D(args.ref_D.data()),
      ptr_Max(args.ptr_Max),
      ptr_Sum(args.ptr_Sum),
      batch_stride_A(args.batch_stride_A),
      batch_stride_B(args.batch_stride_B),
      epilogue_visitor(args.epilogue_visitor)
    {
      ThreadblockSwizzle threadblock_swizzle;
      grid_tiled_shape = threadblock_swizzle.get_tiled_shape(
        args.problem_size,
        {Mma::Shape::kM, Mma::Shape::kN, Mma::Shape::kK},
        args.batch_count
      );
      swizzle_log_tile = threadblock_swizzle.get_log_tile(grid_tiled_shape);
    }
  };

  /// Shared memory structure
  union SharedStorage {
    typename Mma::SharedStorage main_loop;
    struct {
      typename Epilogue::SharedStorage epilogue;
      typename Epilogue::Visitor::SharedStorage visitor;
    } epilo;
  };

  /// Constructor
  CUTLASS_DEVICE
  GemmWithRowColScaleVisitor() { }

  /// Check alignment, etc.
  static cutlass::Status can_implement(cutlass::gemm::GemmCoord const &problem_size) {
    // For brevity, skip or always return success
    return cutlass::Status::kSuccess;
  }
  static cutlass::Status can_implement(Arguments const &args) {
    return can_implement(args.problem_size);
  }

  /// The main "operator()" that the CUDA kernel calls
  CUTLASS_DEVICE
  void operator()(Params const &params, SharedStorage &shared_storage) {

    ThreadblockSwizzle threadblock_swizzle;
    auto threadblock_tile_offset = threadblock_swizzle.get_tile_offset(params.swizzle_log_tile);

    if (threadblock_tile_offset.m() >= params.grid_tiled_shape.m() ||
        threadblock_tile_offset.n() >= params.grid_tiled_shape.n()) {
      return;
    }

    int thread_idx = threadIdx.x;
    int warp_idx   = __shfl_sync(0xffffffff, thread_idx / 32, 0);
    int lane_idx   = thread_idx % 32;

    // (Optional) handle offset for split-K or batched. We'll skip that for brevity:
    int offset_k = 0;
    int problem_size_k = params.problem_size.k();

    using ElementA = typename Mma::IteratorA::Element;
    using ElementB = typename Mma::IteratorB::Element;

    // Recast pointers
    ElementA *ptr_A = static_cast<ElementA *>(params.ptr_A);
    ElementB *ptr_B = static_cast<ElementB *>(params.ptr_B);

    // Coordinates in A and B
    cutlass::MatrixCoord tb_offset_A{
      threadblock_tile_offset.m() * Mma::Shape::kM, offset_k
    };
    cutlass::MatrixCoord tb_offset_B{
      offset_k, threadblock_tile_offset.n() * Mma::Shape::kN
    };

    // Construct Mma iterators
    typename Mma::IteratorA iteratorA(
      params.params_A,
      ptr_A,
      {params.problem_size.m(), problem_size_k},
      thread_idx,
      tb_offset_A
    );

    typename Mma::IteratorB iteratorB(
      params.params_B,
      ptr_B,
      {problem_size_k, params.problem_size.n()},
      thread_idx,
      tb_offset_B
    );

    // Construct Mma
    Mma mma(shared_storage.main_loop, thread_idx, warp_idx, lane_idx);
    typename Mma::FragmentC accum;
    accum.clear();

    int gemm_k_iters = (problem_size_k - offset_k + Mma::Shape::kK - 1) / Mma::Shape::kK;
    // mainloop
    mma(gemm_k_iters, accum, iteratorA, iteratorB, accum);

    // Now create the epilogue visitor
    cutlass::MatrixCoord threadblock_offset(
      threadblock_tile_offset.m() * Mma::Shape::kM,
      threadblock_tile_offset.n() * Mma::Shape::kN
    );

    using EpilogueVisitor = typename Epilogue::Visitor;
    EpilogueVisitor epilogue_visitor(
      params.epilogue_visitor,
      shared_storage.epilo.visitor,
      params.problem_size.m() * params.problem_size.n(), // example extent
      thread_idx,
      warp_idx,
      lane_idx,
      params.params_C,
      params.params_D,
      params.ptr_C,
      params.ptr_D,
      params.ptr_Max,
      params.ptr_Sum,
      threadblock_offset,
      blockIdx.y * params.problem_size.m() // example
    );

    // Possibly set k-partition info
    if (params.mode == cutlass::gemm::GemmUniversalMode::kGemm) {
      epilogue_visitor.set_k_partition(threadblock_tile_offset.k(), params.grid_tiled_shape.k());
    }

    // Construct the epilogue
    typename Epilogue::SharedStorage &epilogue_smem = shared_storage.epilo.epilogue;
    Epilogue epilogue(epilogue_smem, thread_idx, warp_idx, lane_idx);

    // Run the epilogue
    epilogue(epilogue_visitor, accum);
  }
};

/////////////////////////////////////////////////////////////////////////////////////////////////
//
// End of single-file example
//
/////////////////////////////////////////////////////////////////////////////////////////////////
