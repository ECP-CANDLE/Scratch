#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:54:59 2017

@author: johnbauer
"""
import numpy as np

import cython
cimport cpython.array

cimport numpy as np
from libc.math cimport sin, cos, pi

FLOAT64 = np.float64
ctypedef np.float64_t FLOAT64_t


def HyperSphere_lower_triangular(int dim, np.ndarray zeta):
    cdef np.ndarray L_arr = np.zeros((dim, dim), dtype=FLOAT64)
    # Memoryview of the numpy array
    cdef FLOAT64_t [:, :] L = L_arr
    cdef int r, s, j
    
    L[0,0] = 1.0
    for r in range(1, dim):
        for s in range(r):
            L[r,s] = cos(zeta[r,s])
            for j in range(s):
                L[r,s] *= sin(zeta[r,j])
        L[r,r] = 1.0
        for j in range(r):
            L[r,r] *= sin(zeta[r,j])
    return L_arr

# =============================================================================
# TODO: note that parameters are log-transformed by default
# **and the derivative is with respect to the log-transformed parameter**
# Thus require an extra factor of zeta on derivatives
# Use this version when parameters are not log-transformed
# Check theta-setter to be sure
# =============================================================================
def HyperSphere_lower_triangular_derivative(int dim, np.ndarray zeta,
                                            int dr, int ds):
    # TODO: return a 3-d array instead of a list
    cdef int m = dim*(dim-1)//2
    cdef np.ndarray dL_arr = np.zeros((dim, dim), dtype=FLOAT64)
    cdef FLOAT64_t [:, :] dL = dL_arr
    cdef size_t r, s, j
    cdef FLOAT64_t z
    
    for s in range(dr):
        if 0 <= ds <= s:
            z = zeta[dr,s]
            dL[dr,s] = cos(z) if s != ds else -sin(z)
            for j in range(s):
                z = zeta[dr,j]
                dL[dr,s] *= sin(z) if j != ds else cos(z)
    dL[dr,dr] = 1.0 
    for j in range(dr):
        z = zeta[dr,j]
        dL[dr,dr] *= sin(z) if j != ds else cos(z)
    return dL_arr

# =============================================================================
# TODO: note that parameters are log-transformed 
# **and the derivative is with respect to the log-transformed parameter**
# Thus require an extra factor of zeta on derivatives
# Need to document the gradient calculation, or re-engineer parameters
# to omit log transform
# =============================================================================
def HyperSphere_lower_triangular_derivative_log(int dim, np.ndarray zeta,
                                            int dr, int ds):
    # TODO: return a 3-d array instead of a list
    cdef int m = dim*(dim-1)//2
    cdef np.ndarray dL_arr = np.zeros((dim, dim), dtype=FLOAT64)
    cdef FLOAT64_t [:, :] dL = dL_arr
    cdef size_t r, s, j
    cdef FLOAT64_t z
    
    for s in range(dr):
        if 0 <= ds <= s:
            z = zeta[dr,s]
            dL[dr,s] = cos(z) if s != ds else -z * sin(z)
            for j in range(s):
                z = zeta[dr,j]
                dL[dr,s] *= sin(z) if j != ds else z * cos(z)
    dL[dr,dr] = 1.0 
    for j in range(dr):
        z = zeta[dr,j]
        dL[dr,dr] *= sin(z) if j != ds else z * cos(z)
    return dL_arr
