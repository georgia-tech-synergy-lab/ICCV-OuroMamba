from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cutlass_path = "cutlass-3.7.0/include"
file_path = os.path.join(os.getcwd(), cutlass_path)

setup(
    name='custom_quantization_cuda',
    ext_modules=[
        CUDAExtension('custom_quantization_cuda', sources=['custom_quantization_cuda.cu'], include_dirs=[file_path], extra_compile_args={
    'cxx': ["-std=c++20"],
    'nvcc': ["-std=c++20", '-allow-unsupported-compiler', "--expt-relaxed-constexpr", "-Xcompiler=-fpermissive", "-DCUTLASS_USE_TENSOR_CORES=1"]
}
),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
