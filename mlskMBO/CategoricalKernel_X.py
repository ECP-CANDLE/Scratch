#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Oct  9 09:09:33 2017

@author: johnbauer
"""
from __future__ import print_function
from math import pi, sin, cos
import numpy as np
import pandas as pd
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Kernel
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.gaussian_process.kernels import Hyperparameter
from sklearn.preprocessing import StandardScaler

# =============================================================================
# The Kernels here have been tested lightly and seem to be working correctly.
# However, in order to interact with GPR and the rest of sklearn, they need
# to be re-architected to operate on already-created dummy variables,
# relying on the ProjectionKernel to provide the data, rather than creating 
# dummies themselves from a designated column as implemented here.
# =============================================================================

class ProjectionKernel(Kernel):
    def __init__(self, kernel, columns):
        """
        
        kernel:     Kernel object, e.g. RBF()
        columns:    integer or list of integer indices of columns to project onto
        """
        assert isinstance(kernel, Kernel), "Kernel instance required"
        self.kernel = kernel
        self.columns = columns
        # if this gets too tedious go back to using pandas, w
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
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update(('kernel__' + k, val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(Hyperparameter("kernel__" + hyperparameter.name,
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
        return (self.kernel == b.kernel and self.exponent == b.exponent)


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
        return self.kernel(X1)
    
    def __repr__(self):
        return "{0} projected on [{1}]".format(self.kernel, self.columns)

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
    def __init__(self, dim, theta=[]):
        m = dim*(dim-1)//2
        if theta:
            assert len(theta) == m, "Expecting {}*({}-1)/2 elements".format(dim,dim)
        self.dim = dim
        theta = theta if theta else [pi/4.0]*m
        theta_lt = np.zeros((dim, dim))
        # lower triangular indices, offset -1 to get below-diagonal elements
        for th, ind in zip(theta, zip(*np.tril_indices(dim,-1))):
            theta_lt[ind] = th
        # set the diagonal to 1
        for i in range(dim):
            theta_lt[i,i] = 1.0
        self.theta = theta_lt
        self._lt = self._lower_triangular()
            
    def _lower_triangular(self):
        dim = self.dim
        theta = self.theta
        L = np.zeros((dim, dim), dtype=np.float64)
        L[0,0] = 1.0
        for r in range(1,dim):
            for s in range(r):
                L[r,s] = cos(theta[r,s])
                for j in range(s):
                    L[r,s] *= sin(theta[r,j])
            L[r,r] = 1.0
            for j in range(r):
                L[r,r] *= sin(theta[r,j])
        return L
    
    def correlation(self):
        lt = self._lt
        return lt.dot(lt.T)

    def _lower_triangular_derivative(self):
        dim = self.dim
        theta = self.theta
        #dL[0,0] = 0.0
        dLstack = []
        for dr, ds in zip(*np.tril_indices(dim, -1)):
            dL = np.zeros((dim, dim), dtype=np.float64)
            for s in range(dr):
                if s == ds or ds in range(s):
                    dL[dr,s] = cos(theta[dr,s]) if s != ds else -sin(theta[dr,s])
                    for j in range(s):
                        dL[dr,s] *= sin(theta[dr,j]) if j != ds else cos(theta[dr,j])
            dL[dr,dr] = 1.0 
            for j in range(dr):
                dL[dr,dr] *= sin(theta[dr,j]) if j != ds else cos(theta[dr,j])
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
    def __init__(self, column, dim):
        """Dummy-code the given column, put a RBF kernel
        on each of the variates, then return the product kernel
        
        If all length scales are small, assume little shared information
        between categories
        
        column is an integer indexing a column in X
        which contains integer values in range(dim)
        kernel will typically be RBF with a single length parameter
        (passing in an alternative kernel not currently implemented)
        """
        assert isinstance(column, (list, tuple, int)), "must be int or list of ints"
        self.column = [column] if isinstance(column, int) else column
        assert all(isinstance(i, int) for i in self.column), "must be integers"
        self.dim = dim
        
        cats = [i for i in range(self.dim)]
        kernels = [ProjectionKernel(RBF(), c) for c in cats]

        self.kernel = reduce(lambda k0, k1 : k0 * k1, kernels)

    def _get_dummies(self, X):
        X = np.atleast_2d(X)
        
        cats = [i for i in range(self.dim)]

        cx = X[:,self.column].astype(np.int64)
        
        assert set(cx).issubset(set(cats)), "Column entries must be in range({}".format(self.dim)
        
        cx = cx.flatten()
        cx = pd.Categorical(cx, categories=cats)
        dx = pd.get_dummies(cx) #.as_matrix()
        
        return dx
        
    def __call__(self, X, Y=None, eval_gradient=False):
        """Project onto a single column, dummy-code the column, 
        project onto each category"""
        
        dX = self._get_dummies(X)
        dY = self._get_dummies(Y) if Y is not None else None
        
        return self.kernel(dX, dY, eval_gradient=eval_gradient)
    
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
        params = dict(kernel=self.kernel, column=self.column)
        if deep:
            deep_items = self.kernel.get_params().items()
            params.update(('kernel__' + k, val) for k, val in deep_items)
        return params

    @property
    def hyperparameters(self):
        """Returns a list of all hyperparameter."""
        r = []
        for hyperparameter in self.kernel.hyperparameters:
            r.append(Hyperparameter("kernel__" + hyperparameter.name,
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
        return (self.kernel == b.kernel and self.exponent == b.exponent)


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
        return "{0} dummy coded on {1}".format(self.kernel, self.column)

    def is_stationary(self):
        """Returns whether the kernel is stationary. """
        return self.kernel.is_stationary()

    
class FactorKernel(Kernel):
    def __init__(self, column, dim, zeta, zeta_bounds=(0.0, pi)):
        # TODO: assert len(zeta) == dim*(dim-1)/2
        self.zeta = zeta
        self.zeta_bounds = zeta_bounds
        self.column = column
        self.dim = dim
        self.hs = HyperSphere(dim, zeta)
        self.corr = self.hs.correlation()

    def __call__(self, X, Y=None, eval_gradient=False):
        X = np.atleast_2d(X)
        #Y = np.atleast_2d(Y) if Y is not None else None
        #N = X.shape[0]
        #K = np.zeros((N,N), dtype=np.float64)
        corr = self.corr
        # TODO: use broadcasting, verify against naive
        cats = [i for i in range(self.dim)]
        
        cx = X[:,self.column].astype(np.int64)
        cx = pd.Categorical(cx, categories=cats)
        dx = pd.get_dummies(cx) #.as_matrix()
        
        if Y is not None:
            Y = np.atleast_2d(Y)
            cy = Y[:,self.column].astype(np.int64)
            cy = pd.Categorical(cy, categories=cats)
            dy = pd.get_dummies(cy)
        else:
            dy = dx
            
        K = dx.dot(corr).dot(dy.T)

        if eval_gradient:
            if Y is not None:
                raise ValueError("Gradient can only be evaluated when Y is None.")
            G = self.hs.gradient()
            # G is a list of arrays
            assert all(g.shape[0] == dx.shape[1] for g in G), "Incompatible dimensions"
            grad = []
            for g in G:
                grad.append(dx.dot(g).dot(dx.T))
            grad_stack = np.dstack(grad)
            return K, grad_stack
        else:
            return K
        
    def naive__call__(self, X, Y=None, eval_gradient=False):
        # retain for testing purposes
        N = X.shape[0]
        Y = Y if Y is not None else X
        M = Y.shape[0]
        K = np.zeros((N,M), dtype=np.float64)
        corr = self.corr
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
    fubar = ProjectionKernel(rbf, [2,3])
    K = fubar(X)
    print("Projection Kernel:\n", K)
    print("Diagonal:\n", fubar.diag(X))
    
    lt = HyperSphere(2)
    print("2-parameter, lower triangular\n", lt._lower_triangular())
    
    lt1 = HyperSphere(3, [pi, pi/3, pi/6])
    print("3-parameter, lower triangular\n", lt1._lower_triangular())
    
    ltz = HyperSphere(5) #, [0.0]*10)
    print("5-parameter, lower triangular\n", ltz._lower_triangular())
    
    ltk = FactorKernel(3, 3, [0.5,0.5,0.5])
    print("Factor Kernel\n", ltk(X))
    
    C5 = np.array([0,0,1,1,1,2,2,2,3,3]).reshape((-1,1))
    print(C5.shape)
    ltk5 = FactorKernel(0, 4, zeta=[pi/6]*6)
    print(ltk5(C5))
    print(ltk5.hyperparameters)
    K, G = ltk5(C5, eval_gradient=True)
    print(K)
    print(G)
