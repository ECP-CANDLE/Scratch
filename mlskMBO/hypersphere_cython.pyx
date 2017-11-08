#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 08:54:59 2017

@author: johnbauer
"""
import numpy as np

import cython

cimport numpy as np
from libc.math cimport sin, cos, pi

#FLOAT64 = np.float64
ctypedef np.float64_t FLOAT64_t



@cython.boundscheck(False)
@cython.initializedcheck(False)  # turn off negative index wrapping for entire function
def initialize_trig_values(int dim,
                           np.ndarray zeta,
                           np.ndarray cos_zeta,
                           np.ndarray sin_zeta):
    
    cdef FLOAT64_t [:, :] z = zeta
    cdef FLOAT64_t [:, :] C = cos_zeta
    cdef FLOAT64_t [:, :] S = sin_zeta
    
    for i in range(dim):
        for j in range(i):
            C[i,j] = cos(z[i,j])
            S[i,j] = sin(z[i,j])
            

@cython.boundscheck(False)
@cython.initializedcheck(False)  # wraparound: turn off negative index wrapping for entire function
def HyperSphere_lower_triangular(np.ndarray L_arr,
                                 int dim,
                                 np.ndarray cos_zeta,
                                 np.ndarray sin_zeta):
    
    #cdef np.ndarray L_arr = np.zeros((dim, dim), dtype=FLOAT64)
    # Memoryviews of the numpy arrays
    cdef FLOAT64_t [:, :] L = L_arr
    cdef FLOAT64_t [:, :] C = cos_zeta
    cdef FLOAT64_t [:, :] S = sin_zeta

    cdef int r, s, j
    cdef FLOAT64_t L_rs
    
    L[0,0] = 1.0
    for r in range(dim):
        L_rs = 1.0
        for j in range(r):
            L_rs *= S[r,j]
        L[r,r] = L_rs
        for s in range(r):
            L_rs = C[r,s]
            for j in range(s):
                L_rs *= S[r,j]
            L[r,s] = L_rs
    #return L_arr

# =============================================================================
# TODO: note that parameters are log-transformed by default
# **and the derivative is with respect to the log-transformed parameter**
# Thus require an extra factor of zeta on derivatives
# Use this version when parameters are *(*not** log-transformed
# Check theta-setter to be sure
# =============================================================================
@cython.boundscheck(False)
@cython.initializedcheck(False)  # turn off negative index wrapping for entire function
def HyperSphere_lower_triangular_derivative(np.ndarray dL_arr,
                                            int dim,
                                            int dr,
                                            int ds,
                                            np.ndarray cos_zeta,
                                            np.ndarray sin_zeta):                                            
    # Memoryviews of the numpy arrays
    cdef FLOAT64_t [:, :] dL = dL_arr
    cdef FLOAT64_t [:, :] C = cos_zeta
    cdef FLOAT64_t [:, :] S = sin_zeta
    
    cdef FLOAT64_t dL_drs
    cdef size_t r, s, j
    
    for s in range(dr):
        if ds <= s:
            dL_drs = C[dr,s] if s != ds else -S[dr,s]
            for j in range(s):
                dL_drs *= S[dr,j] if j != ds else C[dr,j]
            dL[dr,s] = dL_drs
    # now set the diagonal, i.e. s = dr
    dL_drs = 1.0 
    for j in range(dr):
        dL_drs *= S[dr,j] if j != ds else C[dr,j]
    dL[dr,dr] = dL_drs
    
    #return dL_arr






# =============================================================================
# # Keeping this around only for reference, 
# # TODO: move it somewhere else soon
# 
# # =============================================================================
# # TODO: note that parameters are log-transformed by default
# # **and the derivative is with respect to the log-transformed parameter**
# # Thus require an extra factor of zeta on derivatives
# # Use this version when parameters are *(*not** log-transformed
# # Check theta-setter to be sure
# # =============================================================================
# @cython.boundscheck(False)
# @cython.initializedcheck(False)  # turn off negative index wrapping for entire function
# def HyperSphere_lower_triangular_derivative_old(int dim,
#                                                 np.ndarray zeta,
#                                                 int dr, int ds):
#     # TODO: return a 3-d array instead of a list
#     cdef int m = dim*(dim-1)//2
#     cdef np.ndarray dL_arr = np.zeros((dim, dim), dtype=FLOAT64)
#     cdef FLOAT64_t [:, :] dL = dL_arr
#     cdef FLOAT64_t [:, :] zed = zeta
#     cdef FLOAT64_t z
#     cdef size_t r, s, j
#     
#     for s in range(dr):
#         if 0 <= ds <= s:
#             z = zed[dr,s]
#             dL[dr,s] = cos(z) if s != ds else -sin(z)
#             for j in range(s):
#                 z = z[dr,j]
#                 dL[dr,s] *= sin(z) if j != ds else cos(z)
#     dL[dr,dr] = 1.0 
#     for j in range(dr):
#         z = z[dr,j]
#         dL[dr,dr] *= sin(z) if j != ds else cos(z)
#         
#     return dL_arr
# 
# 
# 
# @cython.boundscheck(False)
# @cython.initializedcheck(False)  # wraparound: turn off negative index wrapping for entire function
# def HyperSphere_lower_triangular_old(int dim, np.ndarray zeta):
#     cdef np.ndarray L_arr = np.zeros((dim, dim), dtype=FLOAT64)
#     # Memoryviews of the numpy arrays
#     cdef FLOAT64_t [:, :] L = L_arr
#     cdef FLOAT64_t [:, :] z = zeta
#     cdef int r, s, j
#     
#     L[0,0] = 1.0
#     for r in range(1, dim):
#         for s in range(r):
#             L[r,s] = cos(z[r,s])
#             for j in range(s):
#                 L[r,s] *= sin(z[r,j])
#         L[r,r] = 1.0
#         for j in range(r):
#             L[r,r] *= sin(z[r,j])
#     return L_arr
# 
# # =============================================================================
# # TODO: note that parameters are log-transformed by default
# # **and the derivative is with respect to the log-transformed parameter**
# # Thus require an extra factor of zeta on derivatives
# # Use this version when parameters are *(*not** log-transformed
# # Check theta-setter to be sure
# # =============================================================================
# @cython.boundscheck(False)
# @cython.initializedcheck(False)  # turn off negative index wrapping for entire function
# def HyperSphere_lower_triangular_derivative_old(int dim,
#                                                 np.ndarray zeta,
#                                                 int dr, int ds):
#     # TODO: return a 3-d array instead of a list
#     cdef int m = dim*(dim-1)//2
#     cdef np.ndarray dL_arr = np.zeros((dim, dim), dtype=FLOAT64)
#     cdef FLOAT64_t [:, :] dL = dL_arr
#     cdef FLOAT64_t [:, :] zed = zeta
#     cdef FLOAT64_t z
#     cdef size_t r, s, j
#     
#     for s in range(dr):
#         if 0 <= ds <= s:
#             z = zed[dr,s]
#             dL[dr,s] = cos(z) if s != ds else -sin(z)
#             for j in range(s):
#                 z = z[dr,j]
#                 dL[dr,s] *= sin(z) if j != ds else cos(z)
#     dL[dr,dr] = 1.0 
#     for j in range(dr):
#         z = z[dr,j]
#         dL[dr,dr] *= sin(z) if j != ds else cos(z)
#     return dL_arr
# 
# # =============================================================================
# # TODO: note that parameters are log-transformed 
# # **and the derivative is with respect to the log-transformed parameter**
# # Thus require an extra factor of zeta on derivatives
# # Need to document the gradient calculation, or re-engineer parameters
# # to omit log transform
# # =============================================================================
# @cython.boundscheck(False)
# @cython.initializedcheck(False)  # turn off negative index wrapping for entire function
# def HyperSphere_lower_triangular_derivative_log_old(int dim,
#                                                     np.ndarray zeta,
#                                                     int dr,
#                                                     int ds):
#     # TODO: return a 3-d array instead of a list
#     cdef int m = dim*(dim-1)//2
#     cdef np.ndarray dL_arr = np.zeros((dim, dim), dtype=FLOAT64)
#     cdef FLOAT64_t [:, :] dL = dL_arr
#     cdef FLOAT64_t [:, :] zed = zeta
#     cdef size_t r, s, j
#     cdef FLOAT64_t z
#     
#     for s in range(dr):
#         if 0 <= ds <= s:
#             z = zed[dr,s]
#             dL[dr,s] = cos(z) if s != ds else -z * sin(z)
#             for j in range(s):
#                 z = z[dr,j]
#                 dL[dr,s] *= sin(z) if j != ds else z * cos(z)
#     dL[dr,dr] = 1.0 
#     for j in range(dr):
#         z = z[dr,j]
#         dL[dr,dr] *= sin(z) if j != ds else z * cos(z)
#     return dL_arr
# 
# =============================================================================
