#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  2 09:49:13 2017

@author: johnbauer
"""
from __future__ import print_function
from math import pi, sin, cos, sqrt, acos
from scipy.linalg import cholesky

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
        zeta_lt[np.tril_indices(dim, -1)] = np.array(zeta, dtype=np.float64)
        # set the diagonal to 1
        np.fill_diagonal(zeta_lt, 1.0)
#        for z, ind in zip(zeta, zip(*np.tril_indices(dim, -1))):
#            zeta_lt[ind] = z
#        # set the diagonal to 1
#        #np.fill_diagonal(zeta_lt.s, 1.0)

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
        for dr in range(1, dim):
            for ds in range(dr):  
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
            np.fill_diagonal(grad, 0)
            gradstack.append(grad)
        return gradstack
    
    # TODO: migrate this to MultiplicativeCorrelation kernel 
    @staticmethod
    def zeta(mult_correlations):
        # assume all entries are positive, will be true if produced by MC model

        t = np.array(mult_correlations, dtype=np.float64)
        t = np.exp(-t)
        t = np.atleast_2d(t)
        T = t.T.dot(t)
        np.fill_diagonal(T, 1.0)
        L = cholesky(T, lower=True)
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
            print("check: {} = {} : difference {}".format(L[r,r], prod, L[r,r] - prod))
        return np.arccos(C[np.tril_indices(dim, -1)])
    
class HyperSphere_full_gradient_pure(HyperSphere):
    def _lower_triangular_derivative_row(self, dr, ds):
        """For given dr, all non-zero elements are in the same row"""
        dim = self.dim
        #zeta = self.zeta
        cos_zeta = self.cos_zeta
        sin_zeta = self.sin_zeta
        
        dL_dr = np.zeros(dim, dtype=np.float64)
            
        hs.HyperSphere_lower_triangular_derivative_row(dL_dr, dim, dr, ds,
                                                       cos_zeta, sin_zeta)
        return dL_dr    
    
    def full_gradient(self, X):
        dim = self.dim
        gradstack = []
        L = self._lower_triangular()
        # these for loops will use prange in OpenMP
        for dr in range(1, dim):
            for ds in range(dr):
                dL = self._lower_triangular_derivative_row(dr, ds)
                dLLt = dL.dot(L.T)
                dLLtXt = dLLt.dot(X.T)
                #grad = np.zeros((dim, dim), dtype=np.float64)
                #grad = np.outer(X[:, dr], dLLtXt)
                #grad = grad + grad.T
                # or:
                grad = np.zeros((dim, dim), dtype=np.float64)
                grad[dr] = dLLtXt
                grad[:,dr] = dLLtXt
                print(grad)
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
        m = dim * (dim - 1) // 2
        self.dim = dim
        if isinstance(zeta, (list, tuple)) and len(zeta):
            assert len(zeta) == m, "Expecting {0}*({0}-1)/2 elements".format(dim)
        elif isinstance(zeta, (int, float, np.float64, np.int64)):
            zeta = [zeta]
        else:
            zeta = [pi / 4.0] * m
        zeta_lt = np.zeros((dim, dim), dtype=np.float64)
        # lower triangular indices, offset -1 to get below-diagonal elements
        zeta_lt[np.tril_indices(dim, -1)] = np.array(zeta, dtype=np.float64)
#        # set the diagonal to 1
#        np.fill_diagonal(zeta_lt, 1.0)
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
    import pandas as pd
    import seaborn as sns

    from collections import defaultdict
            
    template = "{:3} {:6} {:11} {:10.3f} sec, {:15.9f} usec per repetition"
    
    if True:
        hc = HyperSphere(6)
        hp = HyperSphere_pure(6)
        #ho = HyperSphere_pure_original(4)
        
        print("\nChecking Correlation for equivalence to pure python")
        diff = hc.correlation - hp.correlation
        print(diff.min(), diff.max())
        print("Difference should be zero")
        print("\nChecking Gradient for equivalence to pure python")
        for gc, gp in zip(hc.gradient, hp.gradient):
            diff = gc - gp
            print(diff.min(), diff.max())
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
    
    def test(function, dimension, pure, N):
        
        test = "test_{}({}, pure={})".format(function, dimension, pure)
        time = timeit.timeit(test,
                               setup="from __main__ import test_{}".format(function),
                               number=N)
        return dict(function=function, dimension=dimension,
                    code="Python" if pure else "Cython",
                    N=N,
                    time=time)
        
    class TimeitData():
        def __init__(self):
            self.data = defaultdict(list)
            
        def add_result(self, result_dict):
            data = self.data
            for k, v in result_dict.items():
                data[k].append(v)
                
        @property
        def dataframe(self):
            return pd.DataFrame(self.data)

    td = TimeitData()

    N = 200
    max_dim = 31 #31 # was 61; suppress test code when set to 2
    
    usec = 1e6 / N
    
    for d in range(2,max_dim):
        for f in ["correlation", "gradient"]:
            for pure in [False, True]:
                result_dict = test(f, d, pure, N)
                time = result_dict["time"]
                print(template.format(d,
                                      "Python" if pure else "Cython",
                                      f,
                                      time,
                                      time*usec))
                td.add_result(result_dict)
    
    # read cached data if tests are suppressed
    if max_dim == 2:
        df = pd.read_csv("HyperSphere_test_60.csv")
    else:
        df = td.dataframe
        
    df["color"] = df.code.replace(["Cython", "Python"], ["r", "b"])
    df['parameters'] = df.dimension * (df.dimension - 1) / 2
    
    df_corr = df[df.function == "correlation"]
    df_grad = df[df.function == "gradient"]
    
    df_corr.plot.scatter('dimension', 'time', c=df.color, title='Correlation')
    df_grad.plot.scatter('dimension', 'time', c=df.color, title='Gradient')
 
    df_corr.plot.scatter('parameters', 'time', c=df.color, title='Correlation')
    df_grad.plot.scatter('parameters', 'time', c=df.color, title='Gradient')

    sns.lmplot(x='dimension', y='time', data=df_corr, order=2, hue="code")
    sns.lmplot(x='dimension', y='time', data=df_grad, order=2, hue="code")

    sns.lmplot(x='parameters', y='time', data=df_corr, order=2, hue="code")
    sns.lmplot(x='parameters', y='time', data=df_grad, order=2, hue="code")

    df_corr_cython = df_corr[df_corr.code == 'Cython']
    df_corr_python = df_corr[df_corr.code == 'Python']
    
    df_corr_cython.index = range(len(df_corr_cython))
    df_corr_python.index = range(len(df_corr_python))
    
    df_corr_compare = pd.concat([df_corr_cython, df_corr_python], keys=['cython', 'python'], axis=1)
    df_corr_compare['ratio'] = df_corr_compare[('python', 'time')] / df_corr_compare[('cython', 'time')]

    df_grad_cython = df_grad[df_grad.code == 'Cython']
    df_grad_python = df_grad[df_grad.code == 'Python']

    df_grad_cython.index = range(len(df_grad_cython))
    df_grad_python.index = range(len(df_grad_python))
    
    df_grad_compare = pd.concat([df_grad_cython, df_grad_python], keys=['cython', 'python'], axis=1)
    df_grad_compare['ratio'] = df_grad_compare[('python', 'time')] / df_grad_compare[('cython', 'time')]


    df_corr_ratio = pd.DataFrame({ 'ratio' : df_corr_compare['ratio'], 'parameters' : df_corr_compare[('cython', 'parameters')]})
    df_grad_ratio = pd.DataFrame({ 'ratio' : df_grad_compare['ratio'], 'parameters' : df_grad_compare[('cython', 'parameters')]})

    sns.lmplot(x='parameters', y='ratio', data=df_corr_ratio, order=4)
    sns.lmplot(x='parameters', y='ratio', data=df_grad_ratio, order=4)
    
    # TODO: follow up on alternative numpy broadcast calculation of gradient
    if False:
        L = hc._lower_triangular()
        dllt = np.stack(hc._lower_triangular_derivative()).dot(L.T)
        ldlt = np.transpose(dllt, [0,2,1])
        g = dllt + ldlt
        g = np.moveaxis(g, 0, 2)
        hcg = np.dstack(hc.gradient)
        diff = g - hcg
        print(diff.max(), diff.min())
        print(g[:,:,0])
        print(hcg[:,:,0])
        print(g[:,:,-1])
        print(hcg[:,:,-1])
        
# =============================================================================
#     df_corr_compare = df_corr.pivot(index='dimension', columns='code')
#     df_corr_compare['ratio'] = df_corr_compare[('time','Python')] / df_corr_compare[('time','Cython')]
# 
#     df_grad_compare = df_grad.pivot(index='dimension', columns='code')
#     df_grad_compare['ratio'] = df_grad_compare[('time','Python')] / df_grad_compare[('time','Cython')]
#     
#     df_corr_compare.plot.scatter(('parameters', 'Cython'), 'ratio', title='Correlation')
#     df_grad_compare.plot.scatter(('parameters', 'Cython'), 'ratio', title='Gradient')
#     
# =============================================================================

# =============================================================================
#     for d in range(2,2):
#         print("\n")
#         print("*"*80)
#         print("\nDimension = {}, Number of repetitions = {}\n".format(d, N))
#                 
#         time = timeit.timeit("test_correlation({})".format(d),
#                             setup="from __main__ import test_correlation",
#                             number=N)
#         print(template.format("Python", "Correlation", time, time*usec))
#         
#         time = timeit.timeit("test_correlation({}, pure=False)".format(d),
#                             setup="from __main__ import test_correlation",
#                             number=N)
#         print(template.format("Cython", "Correlation", time, time*usec))
#     
#         time = timeit.timeit("test_gradient({})".format(d),
#                             setup="from __main__ import test_gradient",
#                             number=N)
#         print(template.format("Python", "Gradient", time, time*usec))
#     
#     
#         time = timeit.timeit("test_gradient({}, pure=False)".format(d),
#                             setup="from __main__ import test_gradient",
#                             number=N)    
#         print(template.format("Cython", "Gradient", time, time*usec))
# 
# =============================================================================
