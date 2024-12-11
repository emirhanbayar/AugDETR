import os
import glob
import torch
from torch.utils.cpp_extension import CUDA_HOME, CppExtension, CUDAExtension
from setuptools import find_packages, setup

requirements = ["torch", "torchvision"]

def get_extensions():
    this_dir = os.path.dirname(os.path.abspath(__file__))
    extensions_dir = os.path.join(this_dir, "src")

    # Create all necessary directories for build
    os.makedirs(os.path.join(this_dir, "build", "temp.linux-x86_64-cpython-38",
                            "home", "ebayar", "AugDETR-HAE", "models", "dino", "ops", "src", "cpu"), exist_ok=True)
    os.makedirs(os.path.join(this_dir, "build", "temp.linux-x86_64-cpython-38",
                            "home", "ebayar", "AugDETR-HAE", "models", "dino", "ops", "src", "cuda"), exist_ok=True)

    main_file = glob.glob(os.path.join(extensions_dir, "*.cpp"))
    source_cpu = glob.glob(os.path.join(extensions_dir, "cpu", "*.cpp"))
    source_cuda = glob.glob(os.path.join(extensions_dir, "cuda", "*.cu"))

    sources = main_file + source_cpu
    extension = CppExtension
    extra_compile_args = {"cxx": []}
    define_macros = []

    if torch.cuda.is_available() and CUDA_HOME is not None:
        extension = CUDAExtension
        sources += source_cuda
        define_macros += [("WITH_CUDA", None)]
        
        extra_compile_args["cxx"] = ["-std=c++14"]
        nvcc_flags = [
            "-DCUDA_HAS_FP16=1",
            "-D__CUDA_NO_HALF_OPERATORS__",
            "-D__CUDA_NO_HALF_CONVERSIONS__",
            "-D__CUDA_NO_HALF2_OPERATORS__",
            "-gencode=arch=compute_70,code=sm_70",
            "-gencode=arch=compute_75,code=sm_75",
            "-gencode=arch=compute_80,code=sm_80",
            "--compiler-bindir=/home/ebayar/anaconda3/envs/dino/bin/x86_64-conda-linux-gnu-gcc",  # Use GCC 9
            "--expt-relaxed-constexpr",
            "--extended-lambda",
        ]
        
        extra_compile_args["nvcc"] = nvcc_flags
    else:
        raise NotImplementedError('Cuda is not available')

    sources = [os.path.join(extensions_dir, s) for s in sources]
    include_dirs = [extensions_dir]
    ext_modules = [
        extension(
            "MultiScaleDeformableAttention",
            sources,
            include_dirs=include_dirs,
            define_macros=define_macros,
            extra_compile_args=extra_compile_args,
        )
    ]
    return ext_modules

setup(
    name="MultiScaleDeformableAttention",
    version="1.0",
    author="Weijie Su",
    url="https://github.com/fundamentalvision/Deformable-DETR",
    description="PyTorch Wrapper for CUDA Functions of Multi-Scale Deformable Attention",
    packages=find_packages(exclude=("configs", "tests",)),
    ext_modules=get_extensions(),
    cmdclass={"build_ext": torch.utils.cpp_extension.BuildExtension},
)