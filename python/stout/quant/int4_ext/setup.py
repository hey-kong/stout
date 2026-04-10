from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# NOTE: The repository-level `setup.py` is the recommended install entrypoint.
# This setup script is kept for direct extension-only builds.
setup(
    name="quant.int4_ext",
    ext_modules=[
        CUDAExtension(
            name="quant.int4_ext._C",
            sources=["int4_ext.cpp", "int4_ext_cuda.cu"],
            extra_compile_args={
                "cxx": ["-O3"],
                "nvcc": ["-O3", "--use_fast_math"],
            },
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
