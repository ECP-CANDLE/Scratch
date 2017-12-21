#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 20 14:12:03 2017

@author: johnbauer
"""
from __future__ import print_function
from math import pi, sin, cos, sqrt, acos
from scipy.linalg import cholesky

import random
import timeit
import numpy as np
#import hypersphere_cython as hs

# =============================================================================
# Pure Python version for compatibility
# (use Cythonized construction of matrices and gradient for efficiency)
# =============================================================================
class HyperSphere(object):
    """Parameterizes the d-1-dimensional surface of a d-dimensional hypersphere
    using a lower triangular matrix with d*(d-1)/2 parameters, each in the 
    interval (0, pi).  Used for constructing correlation matrices over
    dummy-coded variables representing a categorical variable (factor).
    
    Parameters
    ----------
    dim     int dimension, the number of category levels
    zeta    optional, list or array of parameter values, between 0 and pi
            If supplied, the length must be dim * (dim - 1) / 2 
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
        zeta_lt[np.tril_indices(dim, -1)] = np.array(zeta, dtype=np.float64)
        # set the diagonal to 1
        np.fill_diagonal(zeta_lt, 1.0)

        self.zeta = zeta_lt
        
        # initialize trig values 
        self._cos_zeta = None
        self._sin_zeta = None
        self._initialize_trig_values()
        
        self._lt = None
        # CAUTION do not cache values at initialization because
        # clone_with_theta will not trigger their recalculation with the new
        # parameter values
       
    def _initialize_trig_values(self):
        cos_zeta = np.cos(self.zeta)
        sin_zeta = np.sin(self.zeta)
                
        self._cos_zeta = cos_zeta
        self._sin_zeta = sin_zeta
        
    @property
    def cos_zeta(self):
        return self._cos_zeta

    @property
    def sin_zeta(self):
        return self._sin_zeta
    
    def _lower_triangular(self):
        if self._lt is None:
            dim = self.dim
            L = np.zeros((dim, dim), dtype=np.float64)
            C = self.cos_zeta
            S = self.sin_zeta
            for r in range(dim):
                L_rr = 1.0
                for j in range(r):
                    L_rr *= S[r,j]
                L[r,r] = L_rr
                for s in range(r):
                    L_rs = C[r,s]
                    for j in range(s):
                        L_rs *= S[r,j]
                    L[r,s] = L_rs
            self._lt = L
        return self._lt

    def _lower_triangular_derivative(self):
        dim = self.dim
        #dL[0,0] = 0.0
        dLstack = []
        # auxiliary arrays to avoid recomputing trig functions
        C = self.cos_zeta
        S = self.sin_zeta
        for dr in range(dim):
            for ds in range(dr):
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

    @property
    def correlation(self):
        """Correlation matrix corresponding to zeta."""
        lt = self._lower_triangular()
        corr = lt.dot(lt.T)
        # this is not strictly needed, 
        # but numerical error may cause some diagonal values to be
        # slightly off
        np.fill_diagonal(corr, 1.0)
        return corr
    
    @property
    def gradient(self):
        """List of gradients, one for each parameter.
        
        Not used directly for computations; primarily for testing."""
        L = self._lower_triangular()
        dLstack = self._lower_triangular_derivative()
        gradstack = []
        for dL in dLstack:
            dLLt = dL.dot(L.T)
            grad = dLLt + dLLt.T
            np.fill_diagonal(grad, 0)
            gradstack.append(grad)
        return gradstack
    
    # TODO: check that correlation(zeta(C)) == C
    #       and zeta(correlation(z)) == z
    @staticmethod
    def zeta(correlation):
        """"Hypersphere parameterization of a kernel or correlation matrix. """
        K = correlation
        assert isinstance(K, np.ndarray), "Correlations must be an array"
        assert K.shape[0] == K.shape[1], "Correlations must be square"
        np.fill_diagonal(K, 1.0)
        L = cholesky(K, lower=True)
        C = np.zeros_like(L)
        S = np.zeros_like(L)
        dim = L.shape[0]
        for r in range(1, dim):
            C[r,0] = L[r,0]
            prod = sqrt(1.0 - C[r,0]**2)
            S[r,0] = prod
            for s in range(1,r):
                C[r,s] = L[r,s]/prod
                S[r,s] = sqrt(1.0 - C[r,s]**2)
                prod *= S[r,s]
            #print("check: {} = {} : difference {}".format(L[r,r], prod, L[r,r] - prod))
        return np.arccos(C[np.tril_indices(dim, -1)])