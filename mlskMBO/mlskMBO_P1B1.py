#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:36:59 2017

@author: johnbauer
"""



# =============================================================================
# Assumes mlskMBO is located in Scratch/mlskMBO, at the same level as Benchmarks
# GitHub
#    Scratch
#        mlskMBO
#    Benchmarks
#        common
#        Pilot1
#            common
#            P3B1
# =============================================================================

import os
import sys

# =============================================================================
# Add paths to Benchmarks to system paths to allow imports
# =============================================================================
    
file_path = os.path.dirname(os.path.realpath(__file__))

paths = {"common" : ['..', '..', 'Benchmarks', 'common'],
         "P1_common" : ['..', '..', 'Benchmarks', 'Pilot1', 'common'],
         "P1B1" : ['..', '..', 'Benchmarks', 'Pilot1', 'P1B1']
        }

for path in paths.values():
    lib_path = os.path.abspath(os.path.join(*[file_path]+path))
    sys.path.append(lib_path)

# TODO: change the name to run_data 
# TODO: remove this unless run.x.x.x.json logs written by keras are being used
# import nt3_run_data as nt3d
import p1b1_baseline_keras2 as p1b1k2
import p1b1
#import run_data
import parameter_set as prs
import CategoricalKernel as ck

from collections import defaultdict, namedtuple
from math import pi

#from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid #, ParameterSampler

import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats.distributions import expon

# data are correctly reshaped but warning is present anyway, 
#so suppress them all (bug in sklearn.optimize reported)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import logging
logging.basicConfig(filename='P1B1.log',level=logging.DEBUG)

# =============================================================================
# CONFIGURATION done here for now
# =============================================================================

# if True, parameter dictionaries will be sent to nt3_baseline_keras2
run_keras = True
plots = True

# Location of saved output
output_dir = os.path.join(file_path, 'save')
OUTPUT_SUBDIRECTORY = "experiment_0"

config_file_path = os.path.join(*[file_path]+paths["P1B1"])
CONFIG_FILE = os.path.join(config_file_path, 'p1b1_default_model.txt')
# read in global default parameter configuration
DEFAULT_PARAMS = p1b1.read_config_file(CONFIG_FILE)
# don't include points with ridiculously large losses, as defined here
# when fitting GPR model
MAX_LOSS = 5

PREFIX_SEP = "|"

# the target is validation_loss, could be training_loss or runtime_hours
TARGET = 'validation_loss'

# =============================================================================
# # see https://cran.r-project.org/web/packages/ParamHelpers/ParamHelpers.pdfmakeNum
# # the parameter names should match names of the arguments expected by the benchmark
# 
# # Current best val_corr: 0.96 for ae, 0.86 for vae
# # We are more interested in vae results
# 
# param.set <- makeParamSet(
#   # we optimize for ae and vae separately
#   makeDiscreteParam("model", values=c("ae", "vae")),
# 
#   # latent_dim impacts ae more than vae
#   makeDiscreteParam("latent_dim", values=c(2, 8, 32, 128, 512)),
# 
#   # use a subset of 978 landmark features only to speed up training
#   makeDiscreteParam("use_landmark_genes", values=c(True)),
# 
#   # large batch_size only makes sense when warmup_lr is on
#   makeDiscreteParam("batch_size", values=c(32, 64, 128, 256, 512, 1024)),
# 
#   # use consecutive 978-neuron layers to facilitate residual connections
#   makeDiscreteParam("dense", values=c("2000 600",
#                                       "978 978",
# 				      "978 978 978",
# 				      "978 978 978 978",
# 				      "978 978 978 978 978",
# 				      "978 978 978 978 978 978")),
# 
#   makeDiscreteParam("residual", values=c(True, False)),
# 
#   makeDiscreteParam("activation", values=c("relu", "sigmoid", "tanh")),
# 
#   makeDiscreteParam("optimizer", values=c("adam", "sgd", "rmsprop")),
# 
#   makeNumericParam("learning_rate", lower=0.00001, upper=0.1),
# 
#   makeDiscreteParam("reduce_lr", values=c(True, False)),
# 
#   makeDiscreteParam("warmup_lr", values=c(True, False)),
# 
#   makeNumericParam("drop", lower=0, upper=0.9),
# 
#   makeIntegerParam("epochs", lower=100, upper=200),
# )
# =============================================================================

activation = ["relu", "sigmoid", "tanh"]
#activation = ["softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
batch_size = [16, 32, 64, 128, 256, 512, 1024]
#dense = ["1800 600",
#         "978 978",
#         "978 978 978",
#         "978 978 978 978",
#         "978 978 978 978 978",
#         "978 978 978 978 978 978"]
dense = [
 '[1800, 600]',
 '[978, 978, 978, 978, 978, 978]',
 '[978, 978, 978, 978, 978]',
 '[978, 978, 978, 978]',
 '[978, 978, 978]',
 '[978, 978]']
latent_dim = [2, 8, 32, 128, 512]
model = ["ae", "vae"]
residual = [0, 1]
optimizer = ["adam", "sgd", "rmsprop"]
#optimizer = ["adam", "sgd", "rmsprop", "adagrad", "adadelta","adamax","nadam"]
reduce_lr = [0, 1]
warmup_lr = [0, 1]

# probably add this in param_update... where is it used?
use_landmark_genes = True
      
# =============================================================================
# ParameterGrid used for initial keras runs
# =============================================================================
def p1b1_parameter_grid():
    """Utility function to encapsulate ParameterGrid definition"""
    
    gdict = {"activation" : activation,
             "batch_size" : batch_size,
             "dense" : dense,
             "drop" : [0.0, 0.50, 0.90],
             "epochs" : [10,20], #[100, 150, 200],
             "latent_dim" : [2,8, 32, 128, 512],
             "learning_rate" : [0.00001, 0.05, 0.1],
             "model" : ["ae", "vae"],
             "optimizer" : optimizer,
             "residual" : residual,
             "reduce_lr" : reduce_lr,
             "warmup_lr" : warmup_lr
            }
    
    pg = ParameterGrid(gdict)
    return pg

# =============================================================================
# ParameterGrid used for initial keras runs
# =============================================================================
def p1b1_parameter_grid_optimize(num=50, fixed={}):
    """Utility function to encapsulate ParameterGrid definition
    
    num: number of points for numeric
    fixed: dictionary giving discrete values which are constant
    value should be supplied as a list with a single element
    ... consider using np.ogrid"""
    
    
    gdict = {"activation" : activation,
             "batch_size" : batch_size,
             "dense" : dense,
             "drop" : np.linspace(0.0, 0.9, num=num),
             "epochs" : np.arange(10, 21),
             "latent_dim" : [2, 8, 32, 128, 512],
             "learning_rate" : np.linspace(0.00001, 0.05, num=num),
             "model" : ["ae", "vae"],
             "optimizer" : optimizer,
             "residual" : residual,
             "reduce_lr" : reduce_lr,
             "warmup_lr" : warmup_lr
            }
    gdict.update(fixed)
    
    pg = ParameterGrid(gdict)
    return pg
# =============================================================================
# ParameterSet used for focus search after model fit
# =============================================================================
def p1b1_parameter_set(): 
    """Utility function to encapsulate ParameterSet definition"""
    
    ps = prs.ParameterSet()
    
    # switching batch_size to NumericList to enforce integer validation
    ps.add(prs.DiscreteParameter("activation", activation))
    ps.add(prs.NumericListParameter("batch_size", batch_size))
    ps.add(prs.DiscreteParameter("dense", dense))
    ps.add(prs.NumericParameter("drop", 0.0, 0.9))
    ps.add(prs.IntegerParameter("epochs",  10, 20)) #100, 200))
    ps.add(prs.NumericListParameter("latent_dim", latent_dim))
    ps.add(prs.NumericParameter("learning_rate", 0.00001, 0.1))
    ps.add(prs.DiscreteParameter("model", model))
    ps.add(prs.DiscreteParameter("optimizer", optimizer))
    ps.add(prs.DiscreteParameter("residual", residual))
    ps.add(prs.DiscreteParameter("reduce_lr", reduce_lr))
    ps.add(prs.DiscreteParameter("warmup_lr", warmup_lr))
    
    return ps



def gpr_search(data_df, X_columns, target=TARGET, 
               alpha=0.001,
               factors=[],
               verbose=True):
    """Gaussian Process Regression model
    
    data_df:    dataframe with data to model
    X_columns:  list of parameter names to use as predictors
    target:     name of field to predict
    factors:    list of parameters to be treated as categorical
                creates full-rank indicator variables using pandas get_dummies
    sample:     number of random starting points
    """
    # TODO: move all the print statements over to log
    X = data_df[X_columns]
    y = data_df[target]
    
#    # compare to GPR minimum below...
#    yidxmin = y.idxmin()
#    
#    y_star = y.iloc[yidxmin]
#    X_star = X.iloc[yidxmin]
#    
#    logging.debug("Best observed value")
#    logging.debug("Index: {}".format(yidxmin))
#    logging.debug(y_star)
#    logging.debug("Parameter values for best observed:")
#    logging.debug(X_star)
    
    # Create auxiliary dataframe with dummy-coded indicators 
    if factors:
        #data_with_dummies = prs.DataFrameWithDummies(X, dummies=['shared_nnet_spec', 'ind_nnet_spec'])
        #Xd = data_with_dummies.dataframe
        Xd = pd.get_dummies(X, columns=factors, prefix_sep=PREFIX_SEP) if factors else X
    else:
        Xd = X
        
    continuous_columns = []
    factor_columns = defaultdict(list)
    
    for i, name in enumerate(Xd.columns):
        n0 = name.split(PREFIX_SEP)[0]
        if n0 in factors:
            factor_columns[n0].append(i)
        else:
            continuous_columns.append(i)
    
    n_continuous = len(continuous_columns)
    
    # fairly generic Radial Basis Function kernel with scaling
    # fairly broad bounds on the hyperparameters (which are fit automatically) 
    #k = ConstantKernel(1.0, (0.001, 1000.0))
    
    # TODO: consider initializing with each variable's standard deviation
    kernel = ck.Projection(RBF([1.0]*n_continuous, (0.001, 1000.0)), continuous_columns, name="continuous")
    
    # TODO: move this elsewhere!
    simple = False
    if simple:
        for factor, columns in factor_columns.items():
            dim = len(columns)
            kernel *= ck.Projection(ck.SimpleCategoricalKernel(dim), columns, name=factor)
    else:
        factor_kernel = {}
        for factor, columns in factor_columns.items():
            dim = len(columns)
            fk = ck.FactorKernel(dim) #, [2*pi+pi/4.]*(dim*(dim-1)/2))
            factor_kernel[factor] = fk
            kernel *= ck.Projection(fk, columns, name=factor)
    
    kernel += ConstantKernel()
    
    print(kernel)
    
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   alpha=alpha,
                                   normalize_y=True,
                                   n_restarts_optimizer=20)
    #gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, normalize_y=True)
    
    gpr.fit(Xd, y)    
    # n.b. could scale y but for now handle y with normalize_y in GPR
    
    logging.debug("Fit Gaussian Process Regression:\n{}".format(gpr.kernel_.get_params()))
        
    gprpred = pd.DataFrame({"pred" : gpr.predict(Xd)})
    logging.debug("GPR predictions on training data")
    logging.debug(gprpred.describe())
    gpidx = gprpred.idxmin()
    y_gstar = y.iloc[gpidx]
    logging.debug("observed value corresponding to best prediction on training data")
    logging.debug(y_gstar)
    X_gstar = X.iloc[gpidx]
    logging.debug("Parameter values for best prediction in training data")
    logging.debug(X_gstar)
    if plots:
        gprpred.plot(title="GPR predictions on training data")
    
    lower_bound = Xd.min(axis=0)
    upper_bound = Xd.max(axis=0)
    
    bounds = [(lower, upper) for lower, upper in zip(lower_bound, upper_bound)]
    
    # using location of best actual value (yidxmin) as starting point
    # could try alternative starting points and return the best
    # preliminary tests on NT3 did not show local minima are a problem
    # there is a problem with optimize which can generate Deprecation Warning
    # even when the starting point is 2d and has the correct shape
    columns = Xd.columns
    # trying to track down bizarrely duplicated columns ...
    from collections import Counter
    columncount = Counter(columns)
    logging.debug("columns: {}".format(columns))
    logging.debug(columncount)
    result_data = defaultdict(list)
    for i in range(gprpred.shape[0]):
        # TODO: pick a float between 0 and 1 for each value
        # rescale using bounds as range, i.e. r * (upper - lower) + lower
        start_val = Xd.iloc[i].as_matrix().reshape(-1,1)
        #start_val = np.array(start_val).reshape(-1, 1)
        #start_val = Xs[yidxmin].reshape(-1,  1)
    # Fit the GPR model
        result = sp.optimize.minimize(gpr.predict, start_val, method='L-BFGS-B', bounds=bounds)
        rx = result.x
        pred = gpr.predict(result.x)
        for col, val in zip(columns, rx):
            result_data[col].append(val)
        # pred is an ndarray with shape (1,) so unpack it
        result_data['gpr_optimum'].append(pred[0])
    for k , v in result_data.items():
        logging.debug("{} {}".format(k, len(v)))
    # the dictionary will need to be decoded by a ParameterSet object
    result_data = pd.DataFrame(result_data)
    #result_data = pd.concat((result_data, gprpred), axis=1)
    result_idx = result_data.gpr_optimum.idxmin()
    result_star = result_data.iloc[result_idx]
    result_star = result_star[columns]
    opt = {col : val for col, val in zip(columns, result_star)}
    return opt if not verbose else opt, gpr, Xd, gprpred, result_data


def get_kernel(model, factor, columns, theta):
    dim = len(columns)
    if model == 'Simple':
        kernel = ck.Projection(ck.SimpleCategoricalKernel(dim), columns, name=factor)
    elif model == 'EC':
        kernel = ck.Projection(ck.ExchangeableCorrelation(dim, zeta=theta), columns, name=factor)
    elif model == 'MC':
        kernel = ck.Projection(ck.MultiplicativeCorrelation(dim, zeta=theta), columns, name=factor)
    else: # model == 'UC':
        kernel = ck.Projection(ck.FactorKernel(dim, zeta=theta), columns, name=factor)
    return kernel


def gpr_tensor_search_EC(data_df, X_columns, target=TARGET,
                         alpha=0.001,
                         factors=[],
                         verbose=True):
    """Gaussian Process Regression model
    
    data_df:    dataframe with data to model
    X_columns:  list of parameter names to use as predictors
    target:     name of field to predict
    factors:    list of parameters to be treated as categorical
                creates full-rank indicator variables using pandas get_dummies
    sample:     number of random starting points
    """
    # TODO: move all the print statements over to log
    X = data_df[X_columns]
    y = data_df[target]
    
#    # compare to GPR minimum below...
#    yidxmin = y.idxmin()
#    
#    y_star = y.iloc[yidxmin]
#    X_star = X.iloc[yidxmin]
#    
#    logging.debug("Best observed value")
#    logging.debug("Index: {}".format(yidxmin))
#    logging.debug(y_star)
#    logging.debug("Parameter values for best observed:")
#    logging.debug(X_star)
    
    # Create auxiliary dataframe with dummy-coded indicators 
    if factors:
        #data_with_dummies = prs.DataFrameWithDummies(X, dummies=['shared_nnet_spec', 'ind_nnet_spec'])
        #Xd = data_with_dummies.dataframe
        Xd = pd.get_dummies(X, columns=factors, prefix_sep=PREFIX_SEP) if factors else X
    else:
        Xd = X
        
    continuous_columns = []
    factor_columns = defaultdict(list)
    
    for i, name in enumerate(Xd.columns):
        n0 = name.split(PREFIX_SEP)[0]
        if n0 in factors:
            factor_columns[n0].append(i)
        else:
            continuous_columns.append(i)
    
    n_continuous = len(continuous_columns)
    
    # fairly generic Radial Basis Function kernel with scaling
    # fairly broad bounds on the hyperparameters (which are fit automatically) 
    kernels = [ConstantKernel(1.0, (0.001, 1000.0))]
    
    # TODO: consider initializing with each variable's standard deviation
    kernel = ck.Projection(RBF([1.0]*n_continuous, (0.001, 1000.0)), continuous_columns, name="continuous")
    kernels.append(kernel)
    # TODO: move this elsewhere!

    model = "EC"
    theta = 0.1
    for factor, columns in factor_columns.items():
            kernel = get_kernel(model, factor, columns,theta)
            kernels.append(kernel)
    
    kernel = ck.Tensor(kernels)
    
    print("Exchangeable Correlation Kernel for Gaussian Process Regression")
    print(kernel)
    
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   alpha=alpha,
                                   normalize_y=True,
                                   n_restarts_optimizer=20)
    #gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, normalize_y=True)
    
    gpr.fit(Xd, y)    
    # n.b. could scale y but for now handle y with normalize_y in GPR
    
    logging.debug("Fit Gaussian Process Regression:\n{}".format(gpr.kernel_.get_params()))
        
    gprpred = pd.DataFrame({"pred" : gpr.predict(Xd)})
    logging.debug("GPR predictions on training data")
    logging.debug(gprpred.describe())
    gpidx = gprpred.idxmin()
    y_gstar = y.iloc[gpidx]
    logging.debug("observed value corresponding to best prediction on training data")
    logging.debug(y_gstar)
    X_gstar = X.iloc[gpidx]
    logging.debug("Parameter values for best prediction in training data")
    logging.debug(X_gstar)
    if plots:
        gprpred.plot(title="GPR predictions on training data")
    
    lower_bound = Xd.min(axis=0)
    upper_bound = Xd.max(axis=0)
    
    bounds = [(lower, upper) for lower, upper in zip(lower_bound, upper_bound)]
    
    # using location of best actual value (yidxmin) as starting point
    # could try alternative starting points and return the best
    # preliminary tests on NT3 did not show local minima are a problem
    # there is a problem with optimize which can generate Deprecation Warning
    # even when the starting point is 2d and has the correct shape
    columns = Xd.columns
    # trying to track down bizarrely duplicated columns ...
    from collections import Counter
    columncount = Counter(columns)
    logging.debug("columns: {}".format(columns))
    logging.debug(columncount)
    result_data = defaultdict(list)
    for i in range(gprpred.shape[0]):
        # TODO: pick a float between 0 and 1 for each value
        # rescale using bounds as range, i.e. r * (upper - lower) + lower
        start_val = Xd.iloc[i].as_matrix().reshape(-1,1)
        #start_val = np.array(start_val).reshape(-1, 1)
        #start_val = Xs[yidxmin].reshape(-1,  1)
    # Fit the GPR model
        result = sp.optimize.minimize(gpr.predict, start_val, method='L-BFGS-B', bounds=bounds)
        rx = result.x
        pred = gpr.predict(result.x)
        for col, val in zip(columns, rx):
            result_data[col].append(val)
        # pred is an ndarray with shape (1,) so unpack it
        result_data['gpr_optimum'].append(pred[0])
    for k , v in result_data.items():
        logging.debug("{} {}".format(k, len(v)))
    # the dictionary will need to be decoded by a ParameterSet object
    result_data = pd.DataFrame(result_data)
    #result_data = pd.concat((result_data, gprpred), axis=1)
    result_idx = result_data.gpr_optimum.idxmin()
    result_star = result_data.iloc[result_idx]
    result_star = result_star[columns]
    opt = {col : val for col, val in zip(columns, result_star)}
    return opt if not verbose else opt, gpr, Xd, gprpred, result_data

class GPR_Data(object):
    """Given a dataframe, construct views for X, Y and dummy-coded factors"""
    def __init__(self, data_df, X_columns, factors=[], target=TARGET, 
                 prefix_sep=PREFIX_SEP):
        
        assert set(X_columns).issubset(set(data_df.columns)), "X_columns must be in dataframe's columns"
        assert set(factors).issubset(set(X_columns)), "Factors should be listed in X_columns"
        
        self.data = data_df
        self.factors = factors
        
        X = data_df[X_columns]
        y = data_df[target]
        
        # Create auxiliary dataframe with dummy-coded indicators 
        if factors:
            #data_with_dummies = prs.DataFrameWithDummies(X, dummies=['shared_nnet_spec', 'ind_nnet_spec'])
            #Xd = data_with_dummies.dataframe
            Xd = pd.get_dummies(X, columns=factors, prefix_sep=PREFIX_SEP) if factors else X
        else:
            Xd = X
            
        continuous_columns = []
        factor_columns = defaultdict(list)
        factor_values = defaultdict(list)
        
        for i, name in enumerate(Xd.columns):
            n = name.split(PREFIX_SEP)
            n0 = n[0]
            if n0 in factors:
                factor_columns[n0].append(i)
                factor_values[n0].append(PREFIX_SEP.join(n[1:]))
            else:
                continuous_columns.append(i)
                
        # TODO: create a new parameter set, just for the factors
        ps_factor = prs.ParameterSet()
        for name, values in factor_values.items():
            ps_factor.add(prs.DiscreteParameter(name, values))
    
        self.n_continuous = len(continuous_columns)
        self.parameter_set = ps_factor
        self.X = X
        self.Xd = Xd
        self.y = y
    
    
def gpr_tensor_search_MC(data_df, X_columns, gpr_ec, target=TARGET,
                         alpha=0.001,
                         factors=[],
                         verbose=True):
    """Gaussian Process Regression model
    
    data_df:    dataframe with data to model
    X_columns:  list of parameter names to use as predictors
    gpr_ec:     gpr model using EC for factors
    target:     name of field to predict
    factors:    list of parameters to be treated as categorical
                creates full-rank indicator variables using pandas get_dummies
    sample:     number of random starting points
    """
    # TODO: move all the print statements over to log
    X = data_df[X_columns]
    y = data_df[target]
    
#    # compare to GPR minimum below...
#    yidxmin = y.idxmin()
#    
#    y_star = y.iloc[yidxmin]
#    X_star = X.iloc[yidxmin]
#    
#    logging.debug("Best observed value")
#    logging.debug("Index: {}".format(yidxmin))
#    logging.debug(y_star)
#    logging.debug("Parameter values for best observed:")
#    logging.debug(X_star)
    
    # Create auxiliary dataframe with dummy-coded indicators 
    if factors:
        #data_with_dummies = prs.DataFrameWithDummies(X, dummies=['shared_nnet_spec', 'ind_nnet_spec'])
        #Xd = data_with_dummies.dataframe
        Xd = pd.get_dummies(X, columns=factors, prefix_sep=PREFIX_SEP) if factors else X
    else:
        Xd = X
        
    continuous_columns = []
    factor_columns = defaultdict(list)
    
    for i, name in enumerate(Xd.columns):
        n0 = name.split(PREFIX_SEP)[0]
        if n0 in factors:
            factor_columns[n0].append(i)
        else:
            continuous_columns.append(i)
    
    n_continuous = len(continuous_columns)
    
    # fairly generic Radial Basis Function kernel with scaling
    # fairly broad bounds on the hyperparameters (which are fit automatically) 
    kernels = [ConstantKernel(1.0, (0.001, 1000.0))]
    
    # TODO: consider initializing with each variable's standard deviation
    kernel = ck.Projection(RBF([1.0]*n_continuous, (0.001, 1000.0)), continuous_columns, name="continuous")
    kernels.append(kernel)
    # TODO: move this elsewhere!

    params = gpr_ec.kernel_.get_params()

    found = find_by_name(params)

    ec_kernel = {}
    for k, name in found.items():
        if name in factors:
            # factor_columns already knows columns
            #columns = params.get("{}__columns".format(k), [])
            ec_kernel[name] = params.get("{}__kernel".format(k), None)  
                
    model = "MC"
    for factor, columns in factor_columns.items():
        try:
            theta = ec_kernel[factor].initialize_multiplicative_correlation()
        except:
            print("Whoops! Factor {} not found".format(factor))
            theta = np.array([0.1] * len(columns))
        
        kernel = get_kernel(model, factor, columns, theta)
        kernels.append(kernel)
    
    kernel = ck.Tensor(kernels)
    print("Multiplicative Correlation Kernel for Gaussian Process Regression")
    print(kernel)
    
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   alpha=alpha,
                                   normalize_y=True,
                                   n_restarts_optimizer=20)
    #gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, normalize_y=True)
    
    gpr.fit(Xd, y)    
    # n.b. could scale y but for now handle y with normalize_y in GPR
    
    logging.debug("Fit Gaussian Process Regression:\n{}".format(gpr.kernel_.get_params()))
        
    gprpred = pd.DataFrame({"pred" : gpr.predict(Xd)})
    logging.debug("GPR predictions on training data")
    logging.debug(gprpred.describe())
    gpidx = gprpred.idxmin()
    y_gstar = y.iloc[gpidx]
    logging.debug("observed value corresponding to best prediction on training data")
    logging.debug(y_gstar)
    X_gstar = X.iloc[gpidx]
    logging.debug("Parameter values for best prediction in training data")
    logging.debug(X_gstar)
    if plots:
        gprpred.plot(title="GPR predictions on training data")
    
    lower_bound = Xd.min(axis=0)
    upper_bound = Xd.max(axis=0)
    
    bounds = [(lower, upper) for lower, upper in zip(lower_bound, upper_bound)]
    
    # using location of best actual value (yidxmin) as starting point
    # could try alternative starting points and return the best
    # preliminary tests on NT3 did not show local minima are a problem
    # there is a problem with optimize which can generate Deprecation Warning
    # even when the starting point is 2d and has the correct shape
    columns = Xd.columns
    # trying to track down bizarrely duplicated columns ...
    from collections import Counter
    columncount = Counter(columns)
    logging.debug("columns: {}".format(columns))
    logging.debug(columncount)
    result_data = defaultdict(list)
    for i in range(gprpred.shape[0]):
        # TODO: pick a float between 0 and 1 for each value
        # rescale using bounds as range, i.e. r * (upper - lower) + lower
        start_val = Xd.iloc[i].as_matrix().reshape(-1,1)
        #start_val = np.array(start_val).reshape(-1, 1)
        #start_val = Xs[yidxmin].reshape(-1,  1)
    # Fit the GPR model
        result = sp.optimize.minimize(gpr.predict, start_val, method='L-BFGS-B', bounds=bounds)
        rx = result.x
        pred = gpr.predict(result.x)
        for col, val in zip(columns, rx):
            result_data[col].append(val)
        # pred is an ndarray with shape (1,) so unpack it
        result_data['gpr_optimum'].append(pred[0])
    for k , v in result_data.items():
        logging.debug("{} {}".format(k, len(v)))
    # the dictionary will need to be decoded by a ParameterSet object
    result_data = pd.DataFrame(result_data)
    #result_data = pd.concat((result_data, gprpred), axis=1)
    result_idx = result_data.gpr_optimum.idxmin()
    result_star = result_data.iloc[result_idx]
    result_star = result_star[columns]
    opt = {col : val for col, val in zip(columns, result_star)}
    return opt if not verbose else opt, gpr, Xd, gprpred, result_data

def gpr_tensor_search_UC(data_df, X_columns, gpr_mc, target=TARGET,
                         alpha=0.001,
                         factors=[],
                         verbose=True):
    """Gaussian Process Regression model
    
    data_df:    dataframe with data to model
    X_columns:  list of parameter names to use as predictors
    gpr_ec:     gpr model using EC for factors
    target:     name of field to predict
    factors:    list of parameters to be treated as categorical
                creates full-rank indicator variables using pandas get_dummies
    sample:     number of random starting points
    """
    # TODO: move all the print statements over to log
    X = data_df[X_columns]
    y = data_df[target]
    
#    # compare to GPR minimum below...
#    yidxmin = y.idxmin()
#    
#    y_star = y.iloc[yidxmin]
#    X_star = X.iloc[yidxmin]
#    
#    logging.debug("Best observed value")
#    logging.debug("Index: {}".format(yidxmin))
#    logging.debug(y_star)
#    logging.debug("Parameter values for best observed:")
#    logging.debug(X_star)
    
    # Create auxiliary dataframe with dummy-coded indicators 
    if factors:
        #data_with_dummies = prs.DataFrameWithDummies(X, dummies=['shared_nnet_spec', 'ind_nnet_spec'])
        #Xd = data_with_dummies.dataframe
        Xd = pd.get_dummies(X, columns=factors, prefix_sep=PREFIX_SEP) if factors else X
    else:
        Xd = X
        
    continuous_columns = []
    factor_columns = defaultdict(list)
    
    for i, name in enumerate(Xd.columns):
        n0 = name.split(PREFIX_SEP)[0]
        if n0 in factors:
            factor_columns[n0].append(i)
        else:
            continuous_columns.append(i)
    
    n_continuous = len(continuous_columns)
    
    # fairly generic Radial Basis Function kernel with scaling
    # fairly broad bounds on the hyperparameters (which are fit automatically) 
    kernels = [ConstantKernel(1.0, (0.001, 1000.0))]
    
    # TODO: consider initializing with each variable's standard deviation
    kernel = ck.Projection(RBF([1.0]*n_continuous, (0.001, 1000.0)), continuous_columns, name="continuous")
    kernels.append(kernel)
    # TODO: move this elsewhere!

    params = gpr_mc.kernel_.get_params()

    found = find_by_name(params)

    uc_kernel = {}
    for k, name in found.items():
        if name in factors:
            # factor_columns already knows columns
            #columns = params.get("{}__columns".format(k), [])
            uc_kernel[name] = params.get("{}__kernel".format(k), None)  
                
    model = "UC"
    for factor, columns in factor_columns.items():
        try:
            theta = uc_kernel[factor].initialize_unrestrictive_correlation()
        except:
            print("Whoops! Factor {} not found".format(factor))
            dim = len(columns)
            theta = np.array([0.1] * (dim * (dim - 1)//2))
        
        kernel = get_kernel(model, factor, columns, theta)
        kernels.append(kernel)
    
    kernel = ck.Tensor(kernels)
    print("Multiplicative Correlation Kernel for Gaussian Process Regression")
    print(kernel)
    
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   alpha=alpha,
                                   normalize_y=True,
                                   n_restarts_optimizer=20)
    #gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, normalize_y=True)
    
    gpr.fit(Xd, y)    
    # n.b. could scale y but for now handle y with normalize_y in GPR
    
    logging.debug("Fit Gaussian Process Regression:\n{}".format(gpr.kernel_.get_params()))
        
    gprpred = pd.DataFrame({"pred" : gpr.predict(Xd)})
    logging.debug("GPR predictions on training data")
    logging.debug(gprpred.describe())
    gpidx = gprpred.idxmin()
    y_gstar = y.iloc[gpidx]
    logging.debug("observed value corresponding to best prediction on training data")
    logging.debug(y_gstar)
    X_gstar = X.iloc[gpidx]
    logging.debug("Parameter values for best prediction in training data")
    logging.debug(X_gstar)
    if plots:
        gprpred.plot(title="GPR predictions on training data")
    
    lower_bound = Xd.min(axis=0)
    upper_bound = Xd.max(axis=0)
    
    bounds = [(lower, upper) for lower, upper in zip(lower_bound, upper_bound)]
    
    # using location of best actual value (yidxmin) as starting point
    # could try alternative starting points and return the best
    # preliminary tests on NT3 did not show local minima are a problem
    # there is a problem with optimize which can generate Deprecation Warning
    # even when the starting point is 2d and has the correct shape
    columns = Xd.columns
    # trying to track down bizarrely duplicated columns ...
    from collections import Counter
    columncount = Counter(columns)
    logging.debug("columns: {}".format(columns))
    logging.debug(columncount)
    result_data = defaultdict(list)
    for i in range(gprpred.shape[0]):
        # TODO: pick a float between 0 and 1 for each value
        # rescale using bounds as range, i.e. r * (upper - lower) + lower
        start_val = Xd.iloc[i].as_matrix().reshape(-1,1)
        #start_val = np.array(start_val).reshape(-1, 1)
        #start_val = Xs[yidxmin].reshape(-1,  1)
    # Fit the GPR model
        result = sp.optimize.minimize(gpr.predict, start_val, method='L-BFGS-B', bounds=bounds)
        rx = result.x
        pred = gpr.predict(result.x)
        for col, val in zip(columns, rx):
            result_data[col].append(val)
        # pred is an ndarray with shape (1,) so unpack it
        result_data['gpr_optimum'].append(pred[0])
    for k , v in result_data.items():
        logging.debug("{} {}".format(k, len(v)))
    # the dictionary will need to be decoded by a ParameterSet object
    result_data = pd.DataFrame(result_data)
    #result_data = pd.concat((result_data, gprpred), axis=1)
    result_idx = result_data.gpr_optimum.idxmin()
    result_star = result_data.iloc[result_idx]
    result_star = result_star[columns]
    opt = {col : val for col, val in zip(columns, result_star)}
    return opt if not verbose else opt, gpr, Xd, gprpred, result_data


def report(gpr):
    for k, ker in gpr.kernel_.get_params().items():
        try:
            print(k)
            print(ker)
            print(ker.theta)
            print(ker.correlation)
        except:
            pass

def find_by_name(params):
    """Finds keys which have 'name' parameters, and associated name
    
    Currently only Projection kernels have names
    Nested kernel can be obtained from parameter "{}__kernel".format(key)
    """
    found = {}
    for k, v in params.items():
        split = k.split("__", 1)
        name = split[1] if len(split) > 1 else ""
        if name == 'name':
            found[split[0]] = v
        #print(k, split, v)
    return found

def name_report(gpr, factors):
    params = gpr.kernel_.get_params()
    found = find_by_name(params)
    print("*"*50)
    for key, name in found.items():
        if name in factors:  
            try:
                print(name)
                print(params["{}__kernel".format(key)].correlation)
                print("*"*50)
            except:
                pass
            
            
if __name__ == "__main__":
    pg = p1b1_parameter_grid()
    ps = p1b1_parameter_set()
    
    print(pg)
    print(ps)
    
    p1b1csv = "P1B1_data.csv"
    
    p1b1_data = pd.read_csv(p1b1csv)
    
    valid = p1b1_data.validation_loss.notnull()
    p1b1_data = p1b1_data[valid]
    
    print(p1b1_data.describe())
       
# =============================================================================
# After inspecting the data, it seems that the overwhelming majority are < 1
# but there are some really big ones in there
# =============================================================================
    p1b1_data = p1b1_data[p1b1_data.validation_loss < 1]

# =============================================================================
# To work with a subset of the 1046 points remaining after the above:
# =============================================================================
    subset = [i for i in range(len(p1b1_data)) if i % 4 == 0]
    p1b1_data = p1b1_data.iloc[subset]
    p1b1_data = p1b1_data[p1b1_data.model == 'ae']

    data_columns = [
             'drop',
             'optimizer',
             'warmup_lr',
             'activation',
             'residual',
             'batch_size',
             'epochs',
             'dense',
             'latent_dim',
             'reduce_lr',
             'model',
             'learning_rate',
             'run_id',
             'training_loss',
             'validation_loss',
             'runtime_hours',
             'run_id.1',
             'training_loss.1',
             'validation_loss.1',
             'runtime_hours.1'
             ]
    X_columns = [
             'drop',
             'batch_size',
             'epochs',
             'learning_rate'
             ]
    factors =[
             'optimizer',
             'warmup_lr',
             'activation',
             'residual',
             'dense',
             'latent_dim',
             'reduce_lr',
             'model'
             ]
    
    # when restricted to model == 'ae':
    factors.remove('model')
    
    # try a smaller problmen
    #factors = ['dense', 'model', 'warmup_lr', 'reduce_lr']
    assert all(x in data_columns for x in X_columns), "invalid column"

# =============================================================================
# Fit Gaussian Process Regression, optimize the model, return recommended
# parameter dictionary
# =============================================================================
    d_ec, gpr_ec, Xd, gprpred_ec, result_data_ec = \
        gpr_tensor_search_EC(p1b1_data,
                             X_columns + factors, 
                             TARGET,
                             factors=factors)
    logging.debug(d_ec)
    print(d_ec)
# =============================================================================
# The parameters need to be decoded to remove dummies, also validated
# to enforce bounds, and the correct data types(int or float)
# =============================================================================
    dd = ps.decode_dummies(d_ec)
    print(dd)
    
    # report the hyperparameters as fit by the model
    print(gpr_ec.kernel_.get_params())
#    for factor, k in factor_kernel.items():
#        print("Correlations ({})\n{}".format(factor, k.correlation))
# =============================================================================
# Fit Gaussian Process Regression, optimize the model, return recommended
# parameter dictionary
# =============================================================================
    d_mc, gpr_mc, Xd, gprpred_mc, result_data_mc = \
        gpr_tensor_search_MC(p1b1_data,
                             X_columns + factors,
                             gpr_ec,
                             TARGET,
                             factors=factors)
    logging.debug(d_mc)
    print(d_mc)
# =============================================================================
# The parameters need to be decoded to remove dummies, also validated
# to enforce bounds, and the correct data types(int or float)
# =============================================================================
    dd = ps.decode_dummies(d_mc)
    print(dd)
    
    # report the hyperparameters as fit by the model
    print(gpr_mc.kernel_.get_params())
#    for factor, k in factor_kernel.items():
#        print("Correlations ({})\n{}".format(factor, k.correlation))
# =============================================================================
# Fit Gaussian Process Regression, optimize the model, return recommended
# parameter dictionary
# =============================================================================
    d_uc, gpr_uc, Xd, gprpred_uc, result_data_uc = \
        gpr_tensor_search_UC(p1b1_data,
                             X_columns + factors,
                             gpr_mc,
                             TARGET,
                             factors=factors)
    logging.debug(d_uc)
    print(d_uc)
# =============================================================================
# The parameters need to be decoded to remove dummies, also validated
# to enforce bounds, and the correct data types(int or float)
# =============================================================================
    dd = ps.decode_dummies(d_uc)
    print(dd)
    
    # report the hyperparameters as fit by the model
    print(gpr_uc.kernel_.get_params())
#    for factor, k in factor_kernel.items():
#        print("Correlations ({})\n{}".format(factor, k.correlation))