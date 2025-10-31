from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

cutlass_path = "cutlass-3.7.0/include"
file_path = os.path.join(os.getcwd(), cutlass_path)

setup(
    name='custom_quantization_cuda',
    ext_modules=[
        CUDAExtension(
            'int4_gemm',
            sources=['int4gemm.cu'],
            include_dirs=[file_path],
            extra_compile_args={
                'cxx': ["-std=c++20"],
                'nvcc': [
                    "-std=c++20",
                    "-allow-unsupported-compiler",
                    "-Xcompiler=-fpermissive",
                    "-DCUTLASS_ARCH_MMA_SM80_SUPPORTED=1",
                    "-gencode=arch=compute_80,code=compute_80",
                    "-gencode=arch=compute_80,code=sm_80",
                    "-arch=sm_80"
                ]
            }
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
