try:
    from setuptools import setup
    from setuptools.command.build_ext import build_ext as build_ext_orig
except ImportError:
    from distutils.core import setup
    from distutils.command.build_ext import build_ext as build_ext_orig

from distutils.extension import Extension
from Cython.Build import cythonize
import numpy
import platform


# Get the numpy include directory.
numpy_include_dir = numpy.get_include()

# Custom build class to use gcc for C code
class CustomBuildExt(build_ext_orig):
    def build_extensions(self):
        for ext in self.extensions:
            if ext.language == 'c':
                self.compiler.compiler_so[0] = 'gcc'
        super().build_extensions()

# Extensions
# pykdtree (kd tree)
pykdtree = Extension(
    'im2mesh.utils.libkdtree.pykdtree.kdtree',
    sources=[
        'im2mesh/utils/libkdtree/pykdtree/kdtree.c',
        'im2mesh/utils/libkdtree/pykdtree/_kdtree_core.c'
    ],
    language='c',
    extra_compile_args=['-std=c99', '-O3', '-fopenmp'],
    extra_link_args=['-lgomp'],
    include_dirs=[numpy_include_dir]
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
    libraries=['m'] if platform.system() == 'Linux' else [],
    include_dirs=[numpy_include_dir]
)

# mise (efficient mesh extraction)
mise_module = Extension(
    'im2mesh.utils.libmise.mise',
    sources=[
        'im2mesh/utils/libmise/mise.pyx'
    ],
    include_dirs=[numpy_include_dir]
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
    libraries=['m'] if platform.system() == 'Linux' else [],
    include_dirs=[numpy_include_dir]
)

# Gather all extension modules
ext_modules = [
    pykdtree,
    mcubes_module,
    triangle_hash_module,
    mise_module,
    simplify_mesh_module,
    voxelize_module,
]

setup(
    ext_modules=cythonize(ext_modules),
    cmdclass={
        'build_ext': CustomBuildExt
    }
)
