#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:09:33 2017

@author: johnbauer
"""
from __future__ import print_function
from math import pi, sin, cos
import logging
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.preprocessing import StandardScaler

logging.basicConfig(filename='CategoricalKernel.log',level=logging.DEBUG)

class ProjectionKernel(Kernel):
    def __init__(self, kernel, columns, name="proj"):
        """
        
        kernel:     Kernel object, e.g. RBF()
        name:       string to be used in naming kernel parameters
        columns:    integer or list of integer indices of columns to project onto
        """
        assert isinstance(kernel, Kernel), "Kernel instance required"
        self._kernel = kernel
        self.name = name
        self.columns = columns
        # if this gets too tedious go back to using pandas,
        # which handles int/list of ints transparently
        assert isinstance(columns, (list, tuple, int)), "must be int or list of ints"
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
    
    @property
    def kernel(self):
        return self._kernel

# =============================================================================
# Propose a UnaryOperator class to include Exponentiation kernel, 
# ProjectionKernel, SimpleCategoricalKernel, and so on
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
        params = dict(kernel=self.kernel, columns=self.columns)
        #params = dict(columns=self.columns)
        name_ = "{}{}__".format(self.name, self.columns)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update((name_ + k, val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(Hyperparameter("{}__{}".format(self.name, hyperparameter.name),
                                    hyperparameter.value_type,
                                    hyperparameter.bounds,
                                    hyperparameter.n_elements))
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
        return "{0} projected on {1}".format(self.kernel, self.columns)

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return self.kernel.is_stationary()

    
class ExchangeableKernel(Kernel):
    # crude implementation, mixes ideas from Projection and Product kernels
    def __init__(self, n1, r):
        self.n1 = n1
        self.r = r
        
    def __call__(self, X, Y=None, eval_gradient=False):
        K1 = RBF()
        X1 = X[:,:self.n1]
        X2 = X[:,self.n1:]
        # crude implementation
        N = X.shape[0]
        K2 = np.ones((N,N))
        for i in range(N):
            ci = X[i,self.n1]
            for j in range(N):
                cj = X[j,self.n1]
                K2[i,j] = 1.0 if ci == cj else self.r
        if Y is not None:
            Y1 = Y[:,:self.n1]
        else:
            Y1 = None
        # make sure this is elementwise multiplication
        return K1(X1, Y1) * K2
    
    def diag(self, X):
        #return RBF().diag(X[:,self.n1])
        return np.ones([X.shape[0]], dtype=np.float64)
    
    def is_stationary(self):
        return False
    
# TODO: enforce dim*dim-1)/2 = len(theta), remove from signature
class HyperSphere(object):
    """Parameterizes the d-1-dimensional surface of a d-dimensional hypersphere
    using a lower triangular matrix with d*(d-1)/2 parameters, each in the 
    interval (0, pi).
    """
    def __init__(self, dim, zeta=[]):
        m = dim*(dim-1)//2
        self.dim = dim
        if zeta is not None and len(zeta):
            assert len(zeta) == m, "Expecting {0}*({0}-1)/2 elements".format(dim)
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
        self._lt = self._lower_triangular()
            
    def _lower_triangular(self):
        dim = self.dim
        zeta = self.zeta
        L = np.zeros((dim, dim), dtype=np.float64)
        L[0,0] = 1.0
        for r in range(1,dim):
            for s in range(r):
                L[r,s] = cos(zeta[r,s])
                for j in range(s):
                    L[r,s] *= sin(zeta[r,j])
            L[r,r] = 1.0
            for j in range(r):
                L[r,r] *= sin(zeta[r,j])
        return L
    
    @property
    def correlation(self):
        lt = self._lt
        return lt.dot(lt.T)

    def _lower_triangular_derivative(self):
        dim = self.dim
        zeta = self.zeta
        #dL[0,0] = 0.0
        dLstack = []
        for dr, ds in zip(*np.tril_indices(dim, -1)):
            dL = np.zeros((dim, dim), dtype=np.float64)
            for s in range(dr):
                if s == ds or ds in range(s):
                    dL[dr,s] = cos(zeta[dr,s]) if s != ds else -sin(zeta[dr,s])
                    for j in range(s):
                        dL[dr,s] *= sin(zeta[dr,j]) if j != ds else cos(zeta[dr,j])
            dL[dr,dr] = 1.0 
            for j in range(dr):
                dL[dr,dr] *= sin(zeta[dr,j]) if j != ds else cos(zeta[dr,j])
#                else:
#                    # strictly speaking could skip this since initialized to zero anyway
#                    dL[r,s] = 0.0
            dLstack.append(dL)
        return dLstack
    
    def gradient(self):
        L = self._lt
        dLstack = self._lower_triangular_derivative()
        gradstack = []
        for dL in dLstack:
            dLLt = dL.dot(L.T)
            grad = dLLt + dLLt.T
            gradstack.append(grad)
        return gradstack
        # TODO: call np.dstack here, us np.newaxis to handle matrix mult
        # cf kernels.py
        #return np.dstack(gradstack)
    
    
class SimpleCategoricalKernel(Kernel):
    def __init__(self, dim):     #, length_scale, length_scale_bounds=()):
        """Dummy-code the given column, put a RBF kernel
        on each of the variates, then return the product kernel
        
        If all length scales are small, assume little shared information
        between categories
        
        kernel will typically be RBF with a single length parameter
        (passing in an alternative kernel not currently implemented)
        """
#        assert isinstance(column, (list, tuple, int)), "must be int or list of ints"
#        self.column = [column] if isinstance(column, int) else column
#        assert all(isinstance(i, int) for i in self.column), "must be integers"
        self.dim = dim
        
        kernels = [ProjectionKernel(RBF(), [c]) for c in range(dim)]

        # combine the kernels into a single product kernel
        self.kernel = reduce(lambda k0, k1 : k0 * k1, kernels)

    def __call__(self, X, Y=None, eval_gradient=False):
        """Assumes dummy-coded data e.g. from a single column 
        project onto each category"""

        assert X.shape[1] == self.dim, "Wrong dimension for X"
        if Y is not None:
            assert Y.shape[1] == self.dim, "Wrong dimension for Y"
        
        return self.kernel(X, Y, eval_gradient=eval_gradient)
    

#    @property
#    def hyperparameter_length_scale(self):
#        return  Hyperparameter(name="length_scale", value_type="numeric", bounds=self.bounds, n_elements=len(self.dim))
    
# =============================================================================
# Propose a UnaryOperator class to include Exponentiation kernel, 
# ProjectionKernel, SimpleCategoricalKernel, and so on
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
        #params = dict(kernel=self.kernel, dim=self.dim)
        params = dict(dim=self.dim)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update((k, val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(Hyperparameter(hyperparameter.name,
                                    hyperparameter.value_type,
                                    hyperparameter.bounds,
                                    hyperparameter.n_elements))
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
        return (self.kernel == b.kernel and self.dim == b.dim)


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
        return self.kernel.diag(X)

    def __repr__(self):
        return "{0} dummy coded on {1} dimensions".format(self.kernel, self.dim)

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return self.kernel.is_stationary()

    
# =============================================================================
# Assumes X includes a factor f which has been dummy coded into columns
# F = (f_0, f_1, ... f_dim-1)
# Expects to receive a list of indices identifying the columns of F in X
# =============================================================================
class FactorKernel(Kernel):
    def __init__(self, dim, zeta=[], zeta_bounds=(2*pi, 3*pi)): # add 2*pi in hopes of eliminating difficulties with log transform
        # TODO: fix this so zeta can be a list or array
        self.dim = dim
        m = dim*(dim-1)/2
        zeta = zeta if zeta else [pi/4.0]*m
        assert len(zeta) == m, "Expecting {0}*({0}-1)/2 elements".format(dim)
        self.zeta = zeta
        self.zeta_bounds = zeta_bounds
        # TODO: other models for correlation structure (exchangeable, multiplicative)
        #self.hs = HyperSphere(dim, zeta)
        #self.corr = self.hs.correlation

    def __call__(self, X, Y=None, eval_gradient=False):
        logging.debug("Factor: evaluate kernel for zeta:\n{}".format(self.zeta))
        #print("Factor: evaluate kernel for zeta:\n{}".format(self.zeta))
        assert X.shape[1] == self.dim, "Wrong dimension for X"
        if Y is not None:
            assert Y.shape[1] == self.dim, "Wrong dimension for Y"

        Y1 = Y if Y is not None else X
        
        K = X.dot(self.correlation).dot(Y1.T)

        if eval_gradient:
            if Y is not None:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            #G = self.hs.gradient()
            G = HyperSphere(self.dim, self.zeta).gradient()
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
    def correlation(self):
        #return self.hs.correlation
        return HyperSphere(self.dim, self.zeta).correlation
        
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
            G = self.hs.gradient()
            return K, G
        else:
            return K
        
    def diag(self, X):
        return np.ones([X.shape[0]], dtype=np.float64)

    def is_stationary(self):
        return False
    
    @property
    def hyperparameter_zeta(self):
        return  Hyperparameter(name="zeta", value_type="numeric", bounds=self.zeta_bounds, n_elements=len(self.zeta))


# =============================================================================
# TODO: are these necessary, or redundant?
# =============================================================================
#        
#    @property
#    def theta(self):
#        """Returns the (flattened, log-transformed) non-fixed hyperparameters.
#
#        Note that theta are typically the log-transformed values of the
#        kernel's hyperparameters as this representation of the search space
#        is more amenable for hyperparameter search, as hyperparameters like
#        length-scales naturally live on a log-scale.
#
#        Returns
#        -------
#        theta : array, shape (n_dims,)
#            The non-fixed, log-transformed hyperparameters of the kernel
#        """
#        return self.kernel.theta
#
#    @theta.setter
#    def theta(self, theta):
#        """Sets the (flattened, log-transformed) non-fixed hyperparameters.
#
#        Parameters
#        ----------
#        theta : array, shape (n_dims,)
#            The non-fixed, log-transformed hyperparameters of the kernel
#        """
#        self.kernel.theta = theta
#
#    @property
#    def bounds(self):
#        """Returns the log-transformed bounds on the theta.
#
#        Returns
#        -------
#        bounds : array, shape (n_dims, 2)
#            The log-transformed bounds on the kernel's hyperparameters theta
#        """
#        return self.kernel.bounds

# =============================================================================
# class FactorKernel_defunct(Kernel):
#     def __init__(self, column, lt):
#         assert isinstance(lt, HyperSphere), "Expecting a lower triangular HyperSphere thingy"
#         self.column = column
#         self.hs = lt
#         lmat = sp.mat(lt.lower_triangular())
#         self.corr = lmat * lmat.T
#         
#     def __call__(self, X, Y=None, eval_gradient=False):
#         #assert False, "unimplemented"
#         N = X.shape[0]
#         K = np.zeros((N,N), dtype=np.float64)
#         corr = self.corr
#         # for correctness use naive implementation
#         # TODO: use broadcasting, verify against naive
#         c = X[:,self.column]
#         for i in range(N):
#             for j in range(N):
#                 K[i,j] = corr[c[i],c[j]]
#         return K
#         
#     def diag(self, X):
#         return np.ones([X.shape[0]], dtype=np.float64)
# 
#     def is_stationary(self):
#         return False   
# =============================================================================
    
    
if __name__ == "__main__":
    
    X = np.array([[1,2,3,0],[2,1,3,0],[2,2,3,1],[1,4,3,2]])
    # note only two dimensions are being retained by the projection
    rbf = RBF(length_scale=np.ones(2))
    fubar = ProjectionKernel(rbf, [2,3], "proj")
    K = fubar(X)
    print("Projection Kernel:\n", K)
    print("Diagonal:\n", fubar.diag(X))
    
    lt = HyperSphere(2)
    print("2-parameter, lower triangular\n", lt._lower_triangular())
    
    lt1 = HyperSphere(3, [pi, pi/3, pi/6])
    print("3-parameter, lower triangular\n", lt1._lower_triangular())
    
    ltz = HyperSphere(5) #, [0.0]*10)
    print("5-parameter, lower triangular\n", ltz._lower_triangular())
    
    ltk = FactorKernel(3, [0.5,0.5,0.5])
    print("Factor Kernel\n", ltk(X[:,[0,1,2]]))

    #C5 = np.array([0,0,1,1,1,2,2,2,3,3]).reshape((-1,1))
    C = [0,0,1,1,1,2,2,2,3,3]
    e = [[1,0,0,0], [0,1,0,0], [0,0,1,0], [0,0,0,1]]
    D = np.array([e[c] for c in C]).reshape(-1,4)
    #print(C5.shape)
    ltk4 = FactorKernel(4, zeta=[2*pi+pi/6]*6)
    print(ltk4(D))
    print(ltk4.hyperparameters)
    K, G = ltk4(D, eval_gradient=True)
    print(K)
    #print(G)

    sk = SimpleCategoricalKernel(4)
    print(sk.get_params(deep=False))
    print(sk.hyperparameters)
    
    prod = sk * ltk4
    print(prod.get_params(deep=False))
    print(prod.hyperparameters)
    
    ltk4.theta = [pi/4]*6
    print(ltk4.theta)
    print(ltk4.zeta)