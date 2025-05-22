try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from torch.utils.cpp_extension import BuildExtension, CppExtension, CUDAExtension
import numpy
import os 

os.environ["CC"] = "gcc"
os.environ["CXX"] = "gcc"

# Get the numpy include directory.
numpy_include_dir = numpy.get_include()
# include_dirs = [numpy.get_include()]

# Extensions
# pykdtree (kd tree)
# Compile with the command below if running into errors building kdtree
# gcc -MMD -MF /special_curriculum/ShapeGeneration/occupancy_networks/build/temp.linux-x86_64-3.6/im2mesh/utils/libkdtree/pykdtree/_kdtree_core.o.d -pthread -B /opt/conda/envs/mesh_funcspace/compiler_compat -Wl,--sysroot=/ -Wsign-compare -DNDEBUG -g -fwrapv -O3 -Wall -Wstrict-prototypes -fPIC -I/opt/conda/envs/mesh_funcspace/lib/python3.6/site-packages/numpy/core/include -I/opt/conda/envs/mesh_funcspace/lib/python3.6/site-packages/numpy/core/include -I/opt/conda/envs/mesh_funcspace/include/python3.6m -c -c /special_curriculum/ShapeGeneration/occupancy_networks/im2mesh/utils/libkdtree/pykdtree/_kdtree_core.c -o /special_curriculum/ShapeGeneration/occupancy_networks/build/temp.linux-x86_64-3.6/im2mesh/utils/libkdtree/pykdtree/_kdtree_core.o -std=c99 -O3 -fopenmp -DTORCH_API_INCLUDE_EXTENSION_H '-DPYBIND11_COMPILER_TYPE="_gcc"' '-DPYBIND11_STDLIB="_libstdcpp"' '-DPYBIND11_BUILD_ABI="_cxxabi1011"' -DTORCH_EXTENSION_NAME=kdtree -D_GLIBCXX_USE_CXX11_ABI=0

pykdtree = Extension(
    'im2mesh.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'im2mesh/utils/libkdtree/pykdtree/kdtree.c',
        'im2mesh/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    # language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'], #'-std=c99'
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir],
)

# mcubes (marching cubes algorithm)
mcubes_module = Extension(
    'im2mesh.utils.libmcubes.mcubes',
    sources=[
        'im2mesh/utils/libmcubes/mcubes.pyx',
        'im2mesh/utils/libmcubes/pywrapper.cpp',
        'im2mesh/utils/libmcubes/marchingcubes.cpp'
    ],
    language='c++',
    extra_compile_args=['-std=c++11'],
    include_dirs=[numpy_include_dir]
)

# triangle hash (efficient mesh intersection)
triangle_hash_module = Extension(
    'im2mesh.utils.libmesh.triangle_hash',
    sources=[
        'im2mesh/utils/libmesh/triangle_hash.pyx'
    ],
    libraries=['m', "stdc++"],  # Unix-like specific
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'im2mesh.utils.libmise.mise',
    sources=[
        'im2mesh/utils/libmise/mise.pyx'
    ],
)

# simplify (efficient mesh simplification)
simplify_mesh_module = Extension(
    'im2mesh.utils.libsimplify.simplify_mesh',
    sources=[
        'im2mesh/utils/libsimplify/simplify_mesh.pyx'
    ],
    include_dirs=[numpy_include_dir]
)

# voxelization (efficient mesh voxelization)
voxelize_module = Extension(
    'im2mesh.utils.libvoxelize.voxelize',
    sources=[
        'im2mesh/utils/libvoxelize/voxelize.pyx'
    ],
    libraries=['m']  # Unix-like specific
)

# DMC extensions
dmc_pred2mesh_module = CppExtension(
    'im2mesh.dmc.ops.cpp_modules.pred2mesh',
    sources=[
        'im2mesh/dmc/ops/cpp_modules/pred_to_mesh_.cpp',
    ]   
)

dmc_cuda_module = CUDAExtension(
    'im2mesh.dmc.ops._cuda_ext', 
    sources=[
        'im2mesh/dmc/ops/src/extension.cpp',
        'im2mesh/dmc/ops/src/curvature_constraint_kernel.cu',
        'im2mesh/dmc/ops/src/grid_pooling_kernel.cu',
        'im2mesh/dmc/ops/src/occupancy_to_topology_kernel.cu',
        'im2mesh/dmc/ops/src/occupancy_connectivity_kernel.cu',
        'im2mesh/dmc/ops/src/point_triangle_distance_kernel.cu',
    ]
)

# Gather all extension modules
ext_modules = [
    pykdtree,                 # compile seperately
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
    # dmc_pred2mesh_module,
    # dmc_cuda_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    include_dirs = [numpy_include_dir],
    cmdclass={
        'build_ext': BuildExtension
    }
)
