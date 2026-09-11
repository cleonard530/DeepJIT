from pathlib import Path

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


HERE = Path(__file__).resolve().parent
DEEP_JIT_ROOT = HERE.parents[1]


setup(
    name="deep-jit-stable-abi-probe",
    ext_modules=[
        CUDAExtension(
            name="_deep_jit_stable_abi_probe",
            sources=[str(HERE / "probe.cpp")],
            include_dirs=[str(DEEP_JIT_ROOT / "include")],
            extra_compile_args={
                "cxx": [
                    "-std=c++20",
                    "-O2",
                    "-g1",
                    "-Wno-attributes",
                    "-DTORCH_TARGET_VERSION=0x020a000000000000",
                    "-DUSE_CUDA",
                ],
            },
            extra_link_args=["-ldl"],
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
