### Build (WIP)
Activate a Python 3.10 environment with PyTorch, and download cutlass version 3.7 and unzip the tarball in the root directory of this repo. CUTLASS: https://github.com/NVIDIA/cutlass/releases
Then run `python3 setup_submission.py install`. Note that the `setup_submission.py` will look for the cutlass library in the following directory
```
cutlass_path = "cutlass-3.7.0/include"
file_path = os.path.join(os.getcwd(), cutlass_path)
```
This sets up a PyTorch extension library called `custom_quantization_cuda` that you can import.
The library has two key functions for int4-int8 mixed GEMM.
1. pack_int4_kernel_fp16_rowmajor
```
pack_int4_kernel_fp16_rowmajor(
    weight_fp16,      # FP16 weight matrix [M x K]
    weight_scale,     # quantization scale for weights
    M,                # number of rows
    K,                # number of FP16 columns (logical length)
    padded_K,         # padded physical width in int8 elements
    packed_weight     # output tensor (packed int4 stored as int8)
)
```
2. custom_gemm_with_outliers
```
torch_extension.custom_gemm_with_outliers(
    packed_weight,
    act_fp16_col,  # activation matrix in column-major order
    weight_scale,
    act_scale,
    outlier_indices,
    outlier_scales,
    output_cuda
)
```
### Run tests
python3 test_submission.py

This tests will measure the speed of the reguar FP16 GEMM, pack the "weight" matrix to int4, and then call the the `custom_gemm_with_outliers` kernel, which quantizes the "activation" matrix and deliver the mixed precision GEMM. The script will measure performance of the mixed precision GEMM against FP16 GEMM.
