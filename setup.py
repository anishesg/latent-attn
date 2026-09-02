from setuptools import setup, find_packages
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os

src_dir = os.path.join(os.path.dirname(__file__), "src")
csrc_dir = os.path.join(os.path.dirname(__file__), "csrc")

setup(
    name="latent_attn",
    version="0.1.0",
    description="Fused multi-head latent attention with absorbed Q/K projections",
    author="anishesg",
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name="latent_attn._C",
            sources=[
                os.path.join(csrc_dir, "bindings.cpp"),
                os.path.join(src_dir, "fused_mla.cu"),
                os.path.join(src_dir, "naive_mla.cu"),
                os.path.join(src_dir, "fused_mla_batched.cu"),
            ],
            include_dirs=[src_dir, csrc_dir],
            extra_compile_args={
                "cxx": ["-std=c++17", "-O3"],
                "nvcc": [
                    "-std=c++17",
                    "-O3",
                    "-arch=sm_80",
                    "--use_fast_math",
                    "-DTILE_SEQ=32",
                ],
            },
            libraries=["cublas"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
    python_requires=">=3.8",
    install_requires=["torch>=2.0"],
)
