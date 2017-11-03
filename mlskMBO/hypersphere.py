#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:49:13 2017

@author: johnbauer
"""
from __future__ import print_function
from math import pi, sin, cos

import random
import timeit
import numpy as np
import hypersphere_cython as hs

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
        zeta = self.zeta
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
            zeta = [zeta]
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

class HyperSphere_pure_original(HyperSphere_pure):
    def _lower_triangular(self):
        if self._lt is None:
            dim = self.dim
            zeta = self.zeta
            L = np.zeros((dim, dim), dtype=np.float64)
            L[0,0] = 1.0
            for r in range(1, dim):
                for s in range(r):
                    L[r,s] = cos(zeta[r,s])
                    for j in range(s):
                        L[r,s] *= sin(zeta[r,j])
                L[r,r] = 1.0
                for j in range(r):
                    L[r,r] *= sin(zeta[r,j])
            self._lt = L
        return self._lt

    def _lower_triangular_derivative(self):
        dim = self.dim
        zeta = self.zeta
        #dL[0,0] = 0.0
        dLstack = []
        for dr, ds in zip(*np.tril_indices(dim, -1)):
            dL = np.zeros((dim, dim), dtype=np.float64)
            for s in range(dr):
                if 0 <= ds <= s:
                    dL[dr,s] = cos(zeta[dr,s]) if s != ds else -sin(zeta[dr,s])
                    for j in range(s):
                        dL[dr,s] *= sin(zeta[dr,j]) if j != ds else cos(zeta[dr,j])
            # now set the diagonal
            dL[dr,dr] = 1.0 
            for j in range(dr):
                dL[dr,dr] *= sin(zeta[dr,j]) if j != ds else cos(zeta[dr,j])
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
    
if __name__ == "__main__":
    N = 1000
    usec = 1e6 / N
        
    template = "     {:6} {:11} {:10.3f} sec, {:15.9f} usec per repetition"
    
    if True:
        hc = HyperSphere(4)
        hp = HyperSphere_pure(4)
        #ho = HyperSphere_pure_original(4)
        
        print("\nChecking Correlation for equivalence to pure python")
        print((hc.correlation - hp.correlation).max())
        print("Result should be zero")
        print("\nChecking Gradient for equivalence to pure python")
        for gc, gp in zip(hc.gradient, hp.gradient):
            print((gc-gp).max())
        print("All should be zero")
            
#        print("Correlations")
#        print(hc.correlation)
#        print(hp.correlation)
#        print(ho.correlation)
#        
#        print("Gradients")
#        print(hc.gradient)
#        print(hp.gradient)
#        print(ho.gradient)
#        
#        print("Derivatives")
#        print(hc._lower_triangular_derivative())
#        print(ho._lower_triangular_derivative())
#        print(hp._lower_triangular_derivative())
    
    for d in range(2,25):
        print("\n")
        print("*"*80)
        print("\nDimension = {}, Number of repetitions = {}\n".format(d, N))
        
        result = timeit.timeit("test_correlation({})".format(d),
                            setup="from __main__ import test_correlation",
                            number=N)
        print(template.format("Python", "Correlation", result, result*usec))
        
        result = timeit.timeit("test_correlation({}, pure=False)".format(d),
                            setup="from __main__ import test_correlation",
                            number=N)
        print(template.format("Cython", "Correlation", result, result*usec))
    
        result = timeit.timeit("test_gradient({})".format(d),
                            setup="from __main__ import test_gradient",
                            number=N)
        print(template.format("Python", "Gradient", result, result*usec))
    
    
        result = timeit.timeit("test_gradient({}, pure=False)".format(d),
                            setup="from __main__ import test_gradient",
                            number=N)    
        print(template.format("Cython", "Gradient", result, result*usec))
