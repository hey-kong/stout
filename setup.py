from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    ext_modules=[
        CUDAExtension(
            name="stout.quant.int4_ext._C",
            sources=[
                "python/stout/quant/int4_ext/int4_ext.cpp",
                "python/stout/quant/int4_ext/int4_ext_cuda.cu",
            ],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
