# -*- coding: utf-8 -*-

from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize

import numpy

ext_modules = [
    Extension(
        "hypersphere0",
        ["hypersphere0.pyx"],
        include_dirs=[numpy.get_include()],
        extra_compile_args=['-fopenmp'],
        extra_link_args=['-fopenmp'],
    )
]

setup(
    name="hypersphere-parallel",
    ext_modules = cythonize(ext_modules,
                            annotate=True,
                        )
)

# =============================================================================
# To understand the setup.py more fully look at the official distutils
# documentation. To compile the extension for use in the current directory use:
# 
# $ python setup.py build_ext --inplace
# =============================================================================
