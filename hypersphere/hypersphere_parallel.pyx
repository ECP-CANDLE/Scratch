import numpy as np

import cython
from cython.parallel import prange
cimport numpy as np

from libc.math cimport sin, cos, pi

#FLOAT64 = np.float64
ctypedef np.float64_t FLOAT64_t

# This could easily be done with broadcasting:  
# C = np.cos(z)
@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False)
def initialize_trig_values(int dim,
                           np.ndarray zeta,
                           np.ndarray cos_zeta,
                           np.ndarray sin_zeta):
    
    cdef FLOAT64_t [:, :] z = zeta
    cdef FLOAT64_t [:, :] C = cos_zeta
    cdef FLOAT64_t [:, :] S = sin_zeta
    cdef int i, j
    
    for i in range(dim):
        for j in range(i):
            C[i,j] = cos(z[i,j])
            S[i,j] = sin(z[i,j])


@cython.boundscheck(False)
@cython.initializedcheck(False)
@cython.wraparound(False) # wraparound: turn off negative index wrapping for entire function
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

@cython.boundscheck(False)
@cython.initializedcheck(False) 
@cython.wraparound(False)
def HyperSphere_lower_triangular_derivative(np.ndarray dL_arr,
                                            int dim,
                                            int k,
                                            int dr,
                                            int ds,
                                            np.ndarray cos_zeta,
                                            np.ndarray sin_zeta):                                            
    # Memoryviews of the numpy arrays
    cdef FLOAT64_t [:, :, :] dL = dL_arr
    cdef FLOAT64_t [:, :] C = cos_zeta
    cdef FLOAT64_t [:, :] S = sin_zeta
    
    cdef FLOAT64_t dL_drs
    cdef int r, s, j
    
    for s in range(dr):
        if ds <= s:
            dL_drs = C[dr,s] if s != ds else -S[dr,s]
            for j in range(s):
                dL_drs *= S[dr,j] if j != ds else C[dr,j]
            dL[dr,s,k] = dL_drs
    # now set the diagonal, i.e. s = dr
    dL_drs = 1.0 
    for j in range(dr):
        dL_drs *= S[dr,j] if j != ds else C[dr,j]
    dL[dr,dr,k] = dL_drs

    #return dL_arr
    
# keep track of dr separately; k implies both dr and ds; c.f.
# for k, (dr, ds) in enumerate(np.tril_indices(dim, -1)):
@cython.boundscheck(False)
@cython.initializedcheck(False) 
@cython.wraparound(False)
def HyperSphere_lower_triangular_derivative_row(np.ndarray dL_arr,
                                                int dim,
                                                int k,
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
    
    for s in range(dr):
        if ds <= s:
            dL_drs = C[dr,s] if s != ds else -S[dr,s]
            for j in range(s):
                dL_drs *= S[dr,j] if j != ds else C[dr,j]
            dL[k,s] = dL_drs
    # now set the diagonal, i.e. s = dr
    dL_drs = 1.0 
    for j in range(dr):
        dL_drs *= S[dr,j] if j != ds else C[dr,j]
    dL[k,dr] = dL_drs

    #return dL_arr

@cython.boundscheck(False)
@cython.initializedcheck(False) 
@cython.wraparound(False)
def HyperSphere_ltderiv_row_gradient(X, dLLtXt):
    cdef:
        int dim = X.shape[1]
        int N = X.shape[0]
        int m = dim * (dim - 1) // 2
        
    grad = np.zeros((N, N, m), dtype=np.float64)

    i_index, j_index = np.tril_indices(dim, -1)

    # MemoryViews and temporary variables: here, all end in _
    cdef:
        FLOAT64_t[:,:,:] grad_ = grad
        FLOAT64_t[:,:] X_ = X
        FLOAT64_t[:,:] dLLtXt_ = dLLtXt
        #FLOAT64_t[:] dLLtXtk_ = np.zeros(N, dtype=np.float64)
        long[:] i_ind_ = i_index
        int k, dr, i, j_
        FLOAT64_t X_i_dr_
        FLOAT64_t grad_i_k_
        
    with nogil:
        for k in prange(m):
            dr = i_ind_[k]
            #for i_ in range(N):
            #    X_dr_[i_] = X_[i_,dr]
            #    dLLtXtk_[i_] = dLLtXt_[k, i_]
            # relying on Xdr typically being dummy-coded, therefore sparse
            for i in range(N):
                # since X is either 0 or 1, could simply copy when X_i_dr_ is 1
                # Since numpy broadcasting can't be used, copy into the row
                # and corresponding column (but skip zeros)
                X_i_dr_ = X_[i, dr]
                if X_i_dr_ != 0.0:
                    for j_ in range(N):
                        grad_i_k_ = X_i_dr_ * dLLtXt_[k,j_]
                        grad_[i, j_, k] = grad_i_k_
                        grad_[j_, i, k] = grad_i_k_
                grad_[i, i, k] = 0.0
    return grad
