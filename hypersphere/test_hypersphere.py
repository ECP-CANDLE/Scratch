# -*- coding: utf-8 -*-
from hypersphere_class import HyperSphere

import numpy as np
from numpy import pi, cos, sin
import random

# =============================================================================
# Pure Python implememntation, compare to Cython version for testing
# =============================================================================
class HyperSphere_pure(HyperSphere):
    """Parameterizes the d-1-dimensional surface of a d-dimensional hypersphere
    using a lower triangular matrix with d*(d-1)/2 parameters, each in the 
    interval (0, pi).
    """
    def __init__(self, dim, zeta=[]):
        m = dim*(dim-1)//2
        self.dim = dim
        if isinstance(zeta, (list, tuple)) and len(zeta):
            assert len(zeta) == m, "Expecting {0}*({0}-1)/2 elements".format(dim)
        elif isinstance(zeta, (int, float, np.float64, np.int64)):
            zeta = [zeta]*m
        else:
            zeta = [pi/4.0]*m
        zeta_lt = np.zeros((dim, dim))
        # lower triangular indices, offset -1 to get below-diagonal elements
        for th, ind in zip(zeta, zip(*np.tril_indices(dim,-1))):
            zeta_lt[ind] = th
        # set the diagonal to 1
        for i in range(dim):
            zeta_lt[i,i] = 1.0
        self.zeta = zeta_lt
        self._lt = None
        #self._lt = self._lower_triangular()
            
    def _lower_triangular(self):
        if self._lt is None:
            dim = self.dim
            zeta = self.zeta
            L = np.zeros((dim, dim), dtype=np.float64)
            # auxilary array to avoid recomputing sin
            sin_r = np.zeros(dim, dtype=np.float64)
            for r in range(dim):
                L_rr = 1.0
                for j in range(r):
                    sin_rj = sin(zeta[r,j])
                    sin_r[j] = sin_rj
                    L_rr *= sin_rj
                L[r,r] = L_rr
                for s in range(r):
                    L_rs = cos(zeta[r,s])
                    for j in range(s):
                        L_rs *= sin_r[j]
                    L[r,s] = L_rs
            self._lt = L
        return self._lt
    
#    @property
#    def correlation(self):
#        lt = self._lower_triangular()
#        return lt.dot(lt.T)

    def _lower_triangular_derivative(self):
        dim = self.dim
        zeta = self.zeta
        #dL[0,0] = 0.0
        dLstack = []
        # auxiliary arrays to avoid recomputing trig functions
        C = np.zeros((dim, dim), dtype=np.float64)
        S = np.zeros((dim, dim), dtype=np.float64)
        for r, s in zip(*np.tril_indices(dim, -1)):
            zeta_rs = zeta[r,s]
            C[r,s] = cos(zeta_rs)
            S[r,s] = sin(zeta_rs)
        for dr, ds in zip(*np.tril_indices(dim, -1)):
            dL = np.zeros((dim, dim), dtype=np.float64)
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
            dLstack.append(dL)
        return dLstack
    
def test_correlation(dim, pure=True):
    m = dim*(dim-1)//2
    zeta = np.array([random.uniform(0, pi) for i in range(m)])
    h = HyperSphere_pure(dim, zeta) if pure else HyperSphere(dim, zeta)
    return h.correlation
    
def test_gradient(dim, pure=True):
    m = dim*(dim-1)//2
    zeta = np.array([random.uniform(0, pi) for i in range(m)])
    h = HyperSphere_pure(dim, zeta) if pure else HyperSphere(dim, zeta)
    return h.gradient