# -*- coding: utf-8 -*-

from __future__ import print_function
from math import pi, sin, cos

import random
import timeit
import numpy as np
# Run the cell immediately above first
import hypersphere0 as hs

# =============================================================================
# Cythonize construction of matrices and gradient for efficiency
# =============================================================================
class HyperSphere(object):
    """Parameterizes the d-1-dimensional surface of a d-dimensional hypersphere
    using a lower triangular matrix with d*(d-1)/2 parameters, each in the 
    interval (0, pi).
    """
    def __init__(self, dim, zeta=[]):
        m = dim*(dim-1)//2
        self.dim = dim
        if isinstance(zeta, (list, tuple, np.ndarray)) and len(zeta):
            assert len(zeta) == m, "Expecting {0}*({0}-1)/2 elements".format(dim)
        elif isinstance(zeta, (int, float, np.float64, np.int64)):
            zeta = [zeta]
        else:
            zeta = [pi/4.0]*m
        zeta_lt = np.zeros((dim, dim))
        # lower triangular indices, offset -1 to get below-diagonal elements
        for z, ind in zip(zeta, zip(*np.tril_indices(dim, -1))):
            zeta_lt[ind] = z
        # set the diagonal to 1
        #np.fill_diagonal(zeta_lt.s, 1.0)

        self.zeta = zeta_lt
        
        # initialize trig values 
        self._cos_zeta = None
        self._sin_zeta = None
        self._initialize_trig_values()
        
        self._lt = None
        # CAUTION do not cache values at initialization without making sure
        # clone_with_theta will trigger their recalculation !
        #self._lt = self._lower_triangular()
       
    def _initialize_trig_values(self):
        cos_zeta = np.zeros((self.dim, self.dim), dtype=np.float64)
        sin_zeta = np.zeros((self.dim, self.dim), dtype=np.float64)
        
        hs.initialize_trig_values(self.dim, self.zeta, cos_zeta, sin_zeta)
        
        self._cos_zeta = cos_zeta
        self._sin_zeta = sin_zeta
        
    @property
    def cos_zeta(self):
        return self._cos_zeta

    @property
    def sin_zeta(self):
        return self._sin_zeta

    # see HyperSphere_test for pure Python equivalent
    def _lower_triangular(self):
        if self._lt is None:
            lt = np.zeros((self.dim, self.dim))
            hs.HyperSphere_lower_triangular(lt,
                                         self.dim,
                                         self.cos_zeta,
                                         self.sin_zeta)
            self._lt = lt
        return self._lt
    
    @property
    def correlation(self):
        lt = self._lower_triangular()
        corr = lt.dot(lt.T)
        # this is not strictly needed, 
        # but numerical error may cause some diagonal values to be
        # slightly off
        np.fill_diagonal(corr, 1.0)
        return corr

    # see HyperSphere_test for pure Python equivalent
    def _lower_triangular_derivative(self):
        dim = self.dim
        #zeta = self.zeta
        cos_zeta = self.cos_zeta
        sin_zeta = self.sin_zeta
        
        dLstack = []
        for dr, ds in zip(*np.tril_indices(dim, -1)):
            dL = np.zeros((dim, dim), dtype=np.float64)
            
            hs.HyperSphere_lower_triangular_derivative(dL, dim, dr, ds,
                                                    cos_zeta, sin_zeta)
            dLstack.append(dL)
        return dLstack
    
    @property
    def gradient(self):
        L = self._lower_triangular()
        dLstack = self._lower_triangular_derivative()
        gradstack = []
        for dL in dLstack:
            dLLt = dL.dot(L.T)
            grad = dLLt + dLLt.T
            gradstack.append(grad)
        return gradstack

class HyperSphere_full_gradient_parallel(HyperSphere):
    def _lower_triangular_derivative_row(self, dr, ds):
        """For given dr, all non-zero elements are in the same row"""
        dim = self.dim
        #zeta = self.zeta
        cos_zeta = self.cos_zeta
        sin_zeta = self.sin_zeta
        
        dL_dr = np.zeros((dim,), dtype=np.float64)
            
        hs.HyperSphere_lower_triangular_derivative_row(dL_dr, dim, dr, ds,
                                                    cos_zeta, sin_zeta)
        return dL_dr    
    
    def full_gradient(self, X):
        print("oh, foo")
        dim = self.dim
        L = self._lower_triangular()
        i_index, j_index = np.tril_indices(dim, -1)
        return hs.HyperSphere_full_gradient(X,
                                            dim,
                                            L,
                                            i_index,
                                            j_index,
                                            self.cos_zeta,
                                            self.sin_zeta)
    
    def X_full_gradient(self, X):
        dim = self.dim
        m = dim * (dim - 1) // 2
        
        gradstack = np.zeros((dim, dim, m), dtype=np.float64)
        i_index, j_index = np.tril_indices(dim, -1)
        
        L = self._lower_triangular()
        
        cos_zeta = self.cos_zeta
        sin_zeta = self.sin_zeta
        
        print(gradstack.shape)
        print(dim)
        print(X)
        print(i_index)
        print(j_index)
        print(cos_zeta)
        print(sin_zeta)

        dLLt = np.zeros((dim, m), dtype=np.float64)
        dLLtXt = np.zeros((dim, m), dtype=np.float64)
        dLs = np.zeros((dim, m), dtype=np.float64)

        N = X.shape[0]
        Lt = L.T
        Xt = X.T
        
        hs.HyperSphere_full_gradient(gradstack,
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
# =============================================================================
#         hs.HyperSphere_full_gradient(gradstack,
#                                      dim,
#                                      X,
#                                      i_index, j_index,
#                                      L,
#                                      cos_zeta,
#                                      sin_zeta)
#         return gradstack
# =============================================================================
