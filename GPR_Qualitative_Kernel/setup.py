# -*- coding: utf-8 -*-

from distutils.core import setup
from Cython.Build import cythonize

import numpy

setup(
    ext_modules = cythonize("hypersphere_cython.pyx"),
                   include_dirs=[numpy.get_include()]
)

# =============================================================================
# To understand the setup.py more fully look at the official distutils
# documentation. To compile the extension for use in the current directory use:
# 
# $ python setup.py build_ext --inplace
# =============================================================================
