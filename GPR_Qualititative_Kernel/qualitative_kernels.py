#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:09:33 2017

@author: johnbauer
"""
from __future__ import print_function

from hypersphere import HyperSphere
#import hypersphere_cython as hs

from math import pi, sin, cos, log, sqrt
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import Kernel
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel, CompoundKernel
#from sklearn.gaussian_process.kernels import Hyperparameter
#from sklearn.gaussian_process.kernels import NormalizedKernelMixin
from kernels import Kernel
from kernels import RBF, ConstantKernel, CompoundKernel
from kernels import Hyperparameter
from kernels import NormalizedKernelMixin
from sklearn.preprocessing import StandardScaler
from scipy.linalg import cholesky

import logging
logging.basicConfig(filename='CategoricalKernel.log',level=logging.DEBUG)


def _check_category_scale(X, scale):
    scale = np.squeeze(scale).astype(float)
    if np.ndim(scale) > 1:
        raise ValueError("scale cannot be of dimension greater than 1")
    if np.ndim(scale) == 1 and X.shape[1] != scale.shape[0]:
        raise ValueError("Anisotropic kernel must have the same number of "
                         "dimensions as data (%d!=%d)"
                         % (scale.shape[0], X.shape[1]))
    return scale

class Projection(Kernel):
    def __init__(self, columns, name="proj", kernel=None):
        """
        
        kernel:     Kernel object, defaults to RBF()
        name:       string to be used in reporting parameters
        columns:    integer or list of integer indices of columns to project onto
        """
        
        if kernel is None:
            kernel = RBF([1.0] * len(columns))
        assert isinstance(kernel, Kernel), "Kernel instance required"
        self.kernel = kernel
        self.name = name
        self.columns = columns
        # if this gets too tedious go back to using pandas,
        # which handles int/list of ints transparently
        assert isinstance(columns, (list, tuple, int, np.ndarray)), "must be int or list of ints"
        self.columns = [columns] if isinstance(columns, int) else columns
        assert all(isinstance(i, int) for i in self.columns), "must be integers"
        
    def __call__(self, X, Y=None, eval_gradient=False):
        # TODO: pass parameters to RBF to initialize
        # or consider instantiating K1 outside, pass it into init

#        X1 = pd.DataFrame(np.atleast_2d(X))[self.columns] 
#        Y1 = pd.DataFrame(np.atleast_2d(Y))[self.columns] if Y is not None else None

        X1 = np.atleast_2d(X)[:,self.columns] 
        Y1 = np.atleast_2d(Y)[:,self.columns] if Y is not None else None
        
        return self.kernel(X1, Y1, eval_gradient=eval_gradient)
    
# =============================================================================
#     @property
#     def kernel(self):
#         return self._kernel
# =============================================================================

# =============================================================================
# Propose a UnaryOperator class to include Exponentiation kernel, 
# Projection, SimpleCategoricalKernel, and so on
# The sequel is copied wholesale from ExponentiationKernel
# =============================================================================
    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict(kernel=self.kernel, columns=self.columns, name=self.name)
        #params = dict(columns=self.columns)
        #name_ = "{}{}__".format(self.name, self.columns)
        if deep:
            deep_items = self.kernel.get_params().items()
            #params.update((name_ + k, val) for k, val in deep_items)
            #params.update(("kernel__{}".format(k), val) for k, val in deep_items)
            params.update(("{}__{}".format(self.name, k), val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            name = "{}__{}".format(self.name, hyperparameter.name)
            r.append(Hyperparameter(name,
                                    hyperparameter.value_type,
                                    hyperparameter.bounds,
                                    hyperparameter.n_elements,
                                    hyperparameter.log))
        return r

    @property
    def theta(self):
        """Returns the (flattened, log-transformed) non-fixed hyperparameters.

        Note that theta are typically the log-transformed values of the
        kernel's hyperparameters as this representation of the search space
        is more amenable for hyperparameter search, as hyperparameters like
        length-scales naturally live on a log-scale.

        Returns
        -------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        return self.kernel.theta

    @theta.setter
    def theta(self, theta):
        """Sets the (flattened, log-transformed) non-fixed hyperparameters.

        Parameters
        ----------
        theta : array, shape (n_dims,)
            The non-fixed, log-transformed hyperparameters of the kernel
        """
        self.kernel.theta = theta

    @property
    def bounds(self):
        """Returns the log-transformed bounds on the theta.

        Returns
        -------
        bounds : array, shape (n_dims, 2)
            The log-transformed bounds on the kernel's hyperparameters theta
        """
        return self.kernel.bounds

    def __eq__(self, b):
        if type(self) != type(b):
            return False
        return (self.kernel == b.kernel and self.columns == b.columns)


    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        X1 = np.atleast_2d(X)[:,self.columns]
        return self.kernel.diag(X1)
    
    def __repr__(self):
        if self.name:
            return "{{Factor[{1}] -> {0}}}".format(self.kernel, self.name)
        else:
            return "{{Project{1} -> {0}}}".format(self.kernel, self.columns)

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return self.kernel.is_stationary()

# =============================================================================
# TODO: This is now obsolete by making RBF the default kernel of projecion
# =============================================================================
class Factor(Projection):
    def __init__(self, columns, name):
        super(Factor, self).__init__(columns,
                                     name,
                                     kernel=RBF([1.0] * len(columns)))

    def get_params(self, deep=True):
        """Get parameters of this kernel.

        Parameters
        ----------
        deep: boolean, optional
            If True, will return the parameters for this estimator and
            contained subobjects that are estimators.

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.
        """
        params = dict(columns=self.columns, name=self.name)
        #params = dict(columns=self.columns)
        #name_ = "{}{}__".format(self.name, self.columns)
        if deep:
            deep_items = self.kernel.get_params().items()
            #params.update((name_ + k, val) for k, val in deep_items)
            #params.update(("kernel__{}".format(k), val) for k, val in deep_items)
            params.update(("{}__{}".format(self.name, k), val) for k, val in deep_items)
        return params

# =============================================================================
# Assumes X includes a factor f which has been dummy coded into columns
# F = (f_0, f_1, ... f_dim-1)
# Use with Projection to extract the columns
# =============================================================================
class  UnrestrictiveCorrelation(NormalizedKernelMixin, Kernel):
    def __init__(self, dim, zeta=[], zeta_bounds=(0.0, pi)): # add 2*pi in hopes of eliminating difficulties with log transform
        # TODO: fix this so zeta can be a list or array
        self.dim = dim
        m = dim*(dim-1)/2
        zeta = np.array(zeta, dtype=np.float64)
        zeta = zeta if len(zeta) else np.array([pi/4.0]*m, dtype=np.float64)
        assert len(zeta) == m, "Expecting {0}*({0}-1)/2 elements".format(dim)
        self.zeta = zeta
        self.zeta_bounds = zeta_bounds
        # TODO: other models for correlation structure (exchangeable, multiplicative)
        # DON'T try to cache HyperSphere --- clone_with_theta won't reset it 
        #self._hs = HyperSphere(dim, zeta)
        #self.corr = self.hs.correlation

    def __call__(self, X, Y=None, eval_gradient=False):
        #logging.debug("Factor: evaluate kernel for zeta:\n{}".format(self.zeta))
        #print("Factor: evaluate kernel for zeta:\n{}".format(self.zeta))
        assert X.shape[1] == self.dim, "Wrong dimension for X"
        if Y is not None:
            assert Y.shape[1] == self.dim, "Wrong dimension for Y"

        Y1 = Y if Y is not None else X
        
        h = self.hypersphere
        
        K = X.dot(h.correlation).dot(Y1.T)

        if eval_gradient:
            if Y is not None:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            G = h.gradient
            # G is a list of arrays
            assert all(g.shape[0] == X.shape[1] for g in G), "Incompatible dimensions"
            grad = []
            for g in G:
                grad.append(X.dot(g).dot(X.T))
            grad_stack = np.dstack(grad)
            return K, grad_stack
        else:
            return K
        
    @property
    def hypersphere(self):
        return HyperSphere(self.dim, self.zeta)
    
    @property
    def correlation(self):
        #return self.hs.correlation
        return self.hypersphere.correlation
        
    def naive__call__(self, X, Y=None, eval_gradient=False):
        # retain for testing purposes
        N = X.shape[0]
        Y = Y if Y is not None else X
        M = Y.shape[0]
        K = np.zeros((N,M), dtype=np.float64)
        corr = self.correlation
        # for correctness use naive implementation
        # TODO: use broadcasting or kronecker product, verify against naive
        c = X[:,self.column]
        for i in range(N):
            for j in range(M):
                K[i,j] = corr[c[i],c[j]]
        # TODO: the gradient is NOT CORRECT it needs to be 'stretched'
        # see __call__ for full (large!) gradient
        if eval_gradient:
            print("Naive Gradient is for debugging only")
            G = self.hypersphere.gradient()
            return K, G
        else:
            return K
        
# =============================================================================
#     def diag(self, X):
#         return np.ones([X.shape[0]], dtype=np.float64)
# =============================================================================

    def is_stationary(self):
        return False
    
    @property
    def hyperparameter_zeta(self):
        #n_elts = len(self.zeta) if np.iterable(self.zeta) else 1
        dim = self.dim
        m = dim * (dim - 1) // 2
        return  Hyperparameter(name="zeta", 
                               value_type="numeric", 
                               bounds=self.zeta_bounds, 
                               n_elements=m,
                               log=False)

class ExchangeableCorrelation(Kernel):
    def __init__(self, dim, zeta, zeta_bounds=(0.0, 1.0)):
        self.dim = dim
        self.zeta = zeta
        self.zeta_bounds = zeta_bounds
        
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        dim = self.dim
        assert dim == X.shape[1], "Dimension mismatch"

        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")
        assert X.shape[1] == Y.shape[1], "Dimension mismatch for X and Y"
                
        # correlation zeta is a single number between 0 and 1
        C = self.correlation
        
        K = X.dot(C).dot(Y.T)
        
        if eval_gradient:
            if not self.hyperparameter_zeta.fixed:
                # For untransformed coordinates:
                #K_gradient = np.ones((dim,dim), dytpe=np.float64)
                #np.fill_diagonal(K_gradient, 0.0)
                #K_gradient = X.dot(K_gradient).dot(X.T)
                # If log-transformed:
                # n.b. don't bother copying C since we're done with it otherwise
                K_gradient = C
                np.fill_diagonal(K_gradient, 0.0)
                K_gradient = X.dot(K_gradient).dot(X.T)
                return (K, np.dstack([K_gradient]))
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K

    @property
    def correlation(self):
        # correlation zeta is a single number between 0 and 1
        dim = self.dim
        C = np.empty((dim,dim), dtype=np.float64)
        C.fill(self.zeta)
        np.fill_diagonal(C, 1.0)
        return C
    
    def diag(self, X):
        """Returns the diagonal of the kernel k(X, X).

        The result of this method is identical to np.diag(self(X)); however,
        it can be evaluated more efficiently since only the diagonal is
        evaluated.

        Parameters
        ----------
        X : array, shape (n_samples_X, n_features)
            Left argument of the returned kernel k(X, Y)

        Returns
        -------
        K_diag : array, shape (n_samples_X,)
            Diagonal of kernel k(X, X)
        """
        return np.ones(X.shape[0])

    def is_stationary(self):
        return True
    
    def __repr__(self):
        return "ExchangeableCorrelation({0:.3g})".format(self.zeta)        
        
    @property
    def hyperparameter_zeta(self):
        return  Hyperparameter(name="zeta", 
                               value_type="numeric", 
                               bounds=self.zeta_bounds, 
                               n_elements=1,
                               log=False)

    def multiplicative_correlation(self, epsilon=0.00001):
        """Calculates the values used to initialize MultiplicativeCorrelation
        
        Use epsilon to bound values away from 0 and 1, guaranteeing 
        multiplicative correlation will have valid values."""
        
        c = self.zeta
        
        c = c if c > epsilon else epsilon
        c = c if c < 1.0 - epsilon else 1.0 - epsilon
        
        theta = -log(c) / 2.0 
        
        mc = np.empty(self.dim)
        mc.fill(theta)
        
        return mc
    
# =============================================================================
# TODO: decide if dim should be removed as a parameter for all these kernels
# use quadratic formula d = (sqrt(8*m + 1) + 1) // 2 to assert validity
# =============================================================================
class MultiplicativeCorrelation(NormalizedKernelMixin, Kernel):
    def __init__(self, dim, zeta, zeta_bounds=(0.0, 1.e6)):
        self.dim = dim
        self.zeta = np.array(zeta, dtype=np.float64)
        self.zeta_bounds = zeta_bounds
        
    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        if Y is None:
            Y = X
        elif eval_gradient:
            raise ValueError("Gradient can only be evaluated when Y is None.")
        assert X.shape[1] == Y.shape[1], "Dimension mismatch for X and Y"
        
        dim = self.dim
        assert dim == X.shape[1], "Dimension mismatch"
        assert dim == len(self.zeta), "Wrong number of parameters given"
        
        C = self.correlation
        
        K = X.dot(C).dot(Y.T)
        
        if eval_gradient:
            if not self.hyperparameter_zeta.fixed:
                # For untransformed coordinates:
                #K_gradient = np.ones((dim,dim), dytpe=np.float64)
                #np.fill_diagonal(K_gradient, 0.0)
                #K_gradient = X.dot(K_gradient).dot(X.T)
                # If log-transformed:
                # n.b. don't bother copying C since we're done with it otherwise
                K_gradient = -C
                np.fill_diagonal(K_gradient, 0.0)
                K_gradient = X.dot(K_gradient).dot(X.T)
                return (K, np.tile(K_gradient[:,:,np.newaxis], (1, 1, dim)))
            else:
                return K, np.empty((X.shape[0], X.shape[0], 0))
        else:
            return K

    @property
    def correlation(self):
        # zeta is a vector of length dim
        c = -self.zeta
        C = np.exp(c[:, np.newaxis] + c[np.newaxis, :])
        np.fill_diagonal(C, 1.0)
        return C
# =============================================================================
#     def diag(self, X):
#         """Returns the diagonal of the kernel k(X, X).
# 
#         The result of this method is identical to np.diag(self(X)); however,
#         it can be evaluated more efficiently since only the diagonal is
#         evaluated.
# 
#         Parameters
#         ----------
#         X : array, shape (n_samples_X, n_features)
#             Left argument of the returned kernel k(X, Y)
# 
#         Returns
#         -------
#         K_diag : array, shape (n_samples_X,)
#             Diagonal of kernel k(X, X)
#         """
#         return np.ones(X.shape[0])
# =============================================================================

    def is_stationary(self):
        return False
    
# =============================================================================
#     def __repr__(self):
#         return "{0:.3g}**2".format(np.sqrt(self.zeta))        
# =============================================================================
        
    @property
    def hyperparameter_zeta(self):
        return  Hyperparameter(name="zeta", 
                               value_type="numeric", 
                               bounds=self.zeta_bounds, 
                               n_elements=self.dim,
                               log=False)

    def unrestrictive_correlation(self):
        """Calculate parameters to initialize UnrestrictiveCorrelation kernel"""
        # assume all entries are positive, will be true if produced by MC model

        t = np.array(self.zeta, dtype=np.float64)
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


class Tensor(CompoundKernel):
    def __init__(self, kernels):
        super(Tensor, self).__init__(kernels)
        
    def __call__(self, X, Y=None, eval_gradient=False):
        """Computes product of all the kernels (and gradients)"""
        
        if eval_gradient:
            def _k_g_mul_(kg0, kg1):
                k0, g0 = kg0
                k1, g1 = kg1
                return k0 * k1,\
                    np.dstack((g0 * k1[:, :, np.newaxis],
                               g1 * k0[:, :, np.newaxis]))
            return reduce(_k_g_mul_,
                          (k(X, Y, eval_gradient=True) for k in self.kernels))            
        else:
            return reduce(lambda k0, k1 : k0 * k1,
                          (k(X, Y, eval_gradient=False) for k in self.kernels))
            
    def diag(self, X):
        return reduce(lambda d0, d1 : d0 * d1, (k.diag(X) for k in self.kernels))
    
    
class DirectSum(CompoundKernel):
    def __init__(self, kernels):
        super(DirectSum, self).__init__(kernels)
        
    def __call__(self, X, Y=None, eval_gradient=False):
        """Computes product of all the kernels (and gradients)"""
        
        if eval_gradient:
            def _k_g_add_(kg0, kg1):
                k0, g0 = kg0
                k1, g1 = kg1
                return k0 + k1, np.dstack((g0, g1))
            return reduce(_k_g_add_,
                          (k(X, Y, eval_gradient=True) for k in self.kernels))            
        else:
            return reduce(lambda k0, k1 : k0 + k1,
                          (k(X, Y, eval_gradient=False) for k in self.kernels))

    def diag(self, X):
        return reduce(lambda d0, d1 : d0 + d1, (k.diag(X) for k in self.kernels))


if __name__ == "__main__":
    
    X = np.array([[1,2,3,0],[2,1,3,0],[2,2,3,1],[1,4,3,2]])
    # note only two dimensions are being retained by the projection
    rbf = RBF(length_scale=np.ones(2))
    fubar = Projection([2,3], "proj", kernel=rbf)
    K = fubar(X)
    print("Projection Kernel:\n", K)
    print("Diagonal:\n", fubar.diag(X))
    
    lt = HyperSphere(2)
    print("2-parameter, lower triangular\n", lt._lower_triangular())
    
    lt1 = HyperSphere(3, [pi, pi/3, pi/6])
    print("3-parameter, lower triangular\n", lt1._lower_triangular())
    
    ltz = HyperSphere(5) #, [0.0]*10)
    print("5-parameter, lower triangular\n", ltz._lower_triangular())
    
    ltk =  UnrestrictiveCorrelation(3, [0.5,0.5,0.5])
    print("Factor Kernel\n", ltk)
    print("Factor Kernel\n", ltk(X[:,[0,1,2]]))

    #C5 = np.array([0,0,1,1,1,2,2,2,3,3]).reshape((-1,1))
    C = [0,0,1,1,1,2,2,2,3,3]
    e = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    D = np.array([e[c] for c in C]).reshape(-1,4)
    #print(C5.shape)
    ltk4 =  UnrestrictiveCorrelation(4, zeta=[2*pi+pi/6]*6)
    print("Factor Kernel {} dimensions".format(ltk4.dim))
    print(ltk4)
    print(ltk4.theta)
    print(ltk4(D))
    print(ltk4.hyperparameters)
    K, G = ltk4(D, eval_gradient=True)
    print(K)
    #print(G)

# =============================================================================
#     sk = SimpleCategoricalKernel(4)
#     print(sk.get_params(deep=False))
#     print(sk.hyperparameters)
# =============================================================================
    
    prod = ltk4 * ltk4
    print(prod.get_params(deep=False))
    print(prod.hyperparameters)
    
    ltk4.theta = [pi/i for i in range(1,7)]
    print(ltk4.theta)
    print(ltk4.zeta)
    
    t = Tensor([fubar, ltk4, RBF()])
    print(t)