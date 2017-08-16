from distutils.core import setup
from distutils.extension import Extension
from Cython.Distutils import build_ext
import numpy

setup(cmdclass={'build_ext': build_ext},
      ext_modules=[Extension("evolve",
                             ["evolve.pyx"],
                             language='c++',
                             include_dir=[...],
                             include_dirs=[numpy.get_include()]),
                   Extension("CTRNN",
                             ["CTRNN.pyx"],
                             language='c++',
                             include_dir=[...],
                             include_dirs=[numpy.get_include()]),
                   Extension("simulate",
                             ["simulate.pyx"],
                             language='c++',
                             include_dir=[...],
                             include_dirs=[numpy.get_include()])],
      )

# from distutils.core import setup
# from Cython.Build import cythonize
#
# setup(
#     ext_modules=cythonize("evolve.pyx")
# )
#
# python3 setup.py build_ext --inplace
