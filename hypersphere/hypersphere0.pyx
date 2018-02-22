# -*- coding: utf-8 -*-
#cython --annotate --compile-args=-fopenmp --link-args=-gomp

import numpy as np

import cython
from cython.parallel import prange
cimport openmp

cimport numpy as np
from libc.math cimport sin, cos, pi

#FLOAT64 = np.float64
ctypedef np.float64_t FLOAT64_t

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def initialize_trig_values(int dim,
                           np.ndarray zeta,
                           np.ndarray cos_zeta,
                           np.ndarray sin_zeta):
    
    cdef FLOAT64_t [:, :] z = zeta
    cdef FLOAT64_t [:, :] C = cos_zeta
    cdef FLOAT64_t [:, :] S = sin_zeta
    cdef int i, j
    
    with nogil:
        for i in prange(dim):
            for j in prange(i):
                C[i,j] = cos(z[i,j])
                S[i,j] = sin(z[i,j])
            

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)  # wraparound: turn off negative index wrapping for entire function
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
    
    with nogil:
        L[0,0] = 1.0
        for r in prange(dim):
            L_rs = 1.0
            for j in prange(r):
                L_rs *= S[r,j]
            L[r,r] = L_rs
            for s in prange(r):
                L_rs = C[r,s]
                for j in prange(s):
                    L_rs *= S[r,j]
                L[r,s] = L_rs
    #return L_arr

@cython.boundscheck(False)
@cython.initializedcheck(False)  # turn off negative index wrapping for entire function
@cython.wraparound(False)
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
    cdef int r, s, j
    
    with nogil:
        for s in prange(dr):
            if ds <= s:
                dL_drs = C[dr,s] if s != ds else -S[dr,s]
                for j in prange(s):
                    dL_drs *= S[dr,j] if j != ds else C[dr,j]
                dL[dr,s] = dL_drs
        # now set the diagonal, i.e. s = dr
        dL_drs = 1.0 
        for j in prange(dr):
            dL_drs *= S[dr,j] if j != ds else C[dr,j]
        dL[dr,dr] = dL_drs

    #return dL_arr
    
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)  # turn off negative index wrapping
cdef void HyperSphere_lower_triangular_derivative_row(FLOAT64_t [:, :] dLs,
                                                int k,
                                                int dim,
                                                int dr,
                                                int ds,
                                                FLOAT64_t [:, :] C,
                                                FLOAT64_t [:, :] S) nogil:                                            
    # Memoryviews of the numpy arrays
    #cdef FLOAT64_t [:] dL_dr = dL_arr
    #cdef FLOAT64_t [:, :] C = cos_zeta
    #cdef FLOAT64_t [:, :] S = sin_zeta
    
    cdef FLOAT64_t dL_drs
    cdef int r, s, j
    
    for s in prange(dr):
        if ds <= s:
            dL_drs = C[dr,s] if s != ds else -S[dr,s]
            for j in prange(s):
                dL_drs *= S[dr,j] if j != ds else C[dr,j]
            dLs[s,k] = dL_drs
    # now set the diagonal, i.e. s = dr
    dL_drs = 1.0 
    for j in prange(dr):
        dL_drs *= S[dr,j] if j != ds else C[dr,j]
    dLs[dr,k] = dL_drs
        
# row vector by matrix multiplication
# v (rows) X (rows by cols)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)  # turn off negative index wrapping
cdef void vector_by_matrix_mul(FLOAT64_t [:, :] vX, FLOAT64_t [:, :] v, int k,
                         FLOAT64_t [:, :] X, int rows, int cols) nogil:
    cdef:
        int i, j
        FLOAT64_t v_i
    
    for i in prange(rows):
        v_i = v[i, k]
        for j in prange(cols):
            vX[j, k] += v_i * X[i, j]

@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
def HyperSphere_full_gradient(np.ndarray X,
                              int dim,
                              np.ndarray L,
                              np.ndarray i_index,
                              np.ndarray j_index,
                              np.ndarray cos_zeta,
                              np.ndarray sin_zeta):

    print("Full Gradient!")
    cdef int m = dim * (dim - 1) // 2
    cdef gradstack = np.zeros((dim, dim, m), dtype=np.float64)
    cdef:
        FLOAT64_t [:, :, :] grads = gradstack
        #np.ndarray L = self._lower_triangular()
        FLOAT64_t [:, :] dLLt = np.zeros((dim, m), dtype=np.float64)
        FLOAT64_t [:, :] dLLtXt = np.zeros((dim, m), dtype=np.float64)
        FLOAT64_t [:, :] dLs = np.zeros((dim, m), dtype=np.float64)
    
        int N = X.shape[0]
        FLOAT64_t [:, :] Lt = L.T
        FLOAT64_t [:, :] Xt = X.T

    HyperSphere_fool_gradient(grads,
                             dLLtXt,
                             dLLt,
                             dLs,
                             dim,
                             m,
                             N,
                             i_index,
                             j_index,
                             Xt,
                             Lt,
                             cos_zeta,
                             sin_zeta)
    return gradstack


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)  # turn off negative index wrapping
cdef void HyperSphere_fool_gradient(FLOAT64_t [:, :, :] grads,
                                    FLOAT64_t [:, :] dLLtXt,
                                    FLOAT64_t [:, :] dLLt,
                                    FLOAT64_t [:, :] dLs,
                                    np.int_t dim,
                                    np.int_t m,
                                    np.int_t N,
                                    np.int64_t [:] i_ind,
                                    np.int64_t [:] j_ind,
                                    FLOAT64_t [:, :] Xt,
                                    FLOAT64_t [:, :] Lt,
                                    FLOAT64_t [:, :] C,
                                    FLOAT64_t [:, :] S) nogil:    
    cdef:
        int i, k, dr, ds
        
    for k in prange(m):
        dr = i_ind[k]
        ds = j_ind[k]
        #dL = dLs[:, k]
        HyperSphere_lower_triangular_derivative_row(dLs, k, dim, dr, ds, C, S)
        #dLLt = dL.dot(L.T) # row vector by matrix
        vector_by_matrix_mul(dLLt, dLs, k, Lt, dim, dim)
        #dLLtXt = dLLt.dot(X.T)
        vector_by_matrix_mul(dLLtXt, dLLt, k, Xt, dim, dim)
        for i in range(N): # try without prange
            grads[dr, i, k] = dLLtXt[i, k]
            grads[i, dr, k] = dLLtXt[i, k]
 
# TODO: break up the above?
            
# =============================================================================
# @cython.boundscheck(False)
# @cython.initializedcheck(False)
# @cython.wraparound(False)  # turn off negative index wrapping
# def xxx_HyperSphere_full_gradient(np.ndarray gradstack,
#                               np.int_t dim, np.ndarray X,
#                               np.ndarray i_index, np.ndarray j_index,
#                               np.ndarray L, np.ndarray cos_zeta,
#                               np.ndarray sin_zeta):
# 
#     cdef int m = dim * (dim - 1) // 2
#     dLLt_ = np.zeros((dim, m), dtype=np.float64)
#     dLLtXt_ = np.zeros((dim, m), dtype=np.float64)
#     dLs_ = np.zeros((dim, m), dtype=np.float64)
# 
#     cdef int N = X.shape[0]
#     cdef:
#         #int i, k, dr, ds
#         np.int64_t [:] i_ind = i_index
#         np.int64_t [:] j_ind = j_index
#         
#         FLOAT64_t [:, :, :] grads = gradstack
#         FLOAT64_t [:, :] dLs = dLs_
#         #FLOAT64_t [:] dL
#         FLOAT64_t [:, :] dLLt = dLLt_
#         FLOAT64_t [:, :] dLLtXt = dLLtXt_
#         FLOAT64_t [:, :] C = cos_zeta
#         FLOAT64_t [:, :] S = sin_zeta
#         FLOAT64_t [:, :] Lt = L.T
#         FLOAT64_t [:, :] Xt = X.T
# 
#     # gradstack has shape (dim, dim, m)
#     #k, i_index, j_index = enumerate(np.tril_indices(dim, -1)
# 
#     HyperSphere_full_gradient(grads, dLLtXt, dLLt, dLs,
#                                dim, m, N, i_ind, j_ind, Xt, Lt, C, S)
# 
# =============================================================================