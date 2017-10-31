#!/usr/bin/env python2
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function
"""
Created on Thu Sep  7 11:10:38 2017

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
         "P3_common" : ['..', '..', 'Benchmarks', 'Pilot3', 'common'],
         "P3B1" : ['..', '..', 'Benchmarks', 'Pilot3', 'P3B1']
        }

for path in paths.values():
    lib_path = os.path.abspath(os.path.join(*[file_path]+path))
    sys.path.append(lib_path)

# TODO: change the name to run_data 
# TODO: remove this unless run.x.x.x.json logs written by keras are being used
# import nt3_run_data as nt3d
import p3b1_baseline_keras2 as p3b1k2
import p3b1
import parameter_set as prs
import CategoricalKernel as ck
#import CategoricalKernel_cython as ck


from collections import defaultdict
from math import pi

#from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, ParameterSampler

import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats.distributions import expon

# data are correctly reshaped but warning is present anyway, 
#so suppress them all (bug in sklearn.optimize reported)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 


import logging
logging.basicConfig(filename='P3B1.log',level=logging.DEBUG)

# =============================================================================
# CONFIGURATION done here for now
# =============================================================================

# if True, parameter dictionaries will be sent to nt3_baseline_keras2
run_keras = False

# if True, plots will be generated
plots = True

# use simple categorical or factor kernel
use_simple_categorical_kernel = False

# Location of saved output
output_dir = os.path.join(file_path, 'save')
OUTPUT_SUBDIRECTORY = "experiment_0"

# initial data will be read from here if present,
# cached here if not
cache_subdirectory = ""
CACHE_DIR = os.path.join(file_path, 'save', cache_subdirectory)
CACHE_FILE = os.path.join(CACHE_DIR, "P3B1_cache_0.csv")

# if there are no data in the cache, 
# five the mumber of points to draw for the initial sample
INITIAL_SAMPLE = 30

# location of solr if used for results
SOLR_URL = "http://localhost:8983/solr"

config_file_path = os.path.join(*[file_path]+paths["P3B1"])
CONFIG_FILE = os.path.join(config_file_path, 'p3b1_default_model.txt')
# read in global default parameter configuration
DEFAULT_PARAMS = p3b1.read_config_file(CONFIG_FILE)

# don't include points with ridiculously large losses, as defined here
# when fitting GPR model
MAX_LOSS = 5

PREFIX_SEP = "|"

# the target is validation_loss, could be training_loss or runtime_hours
TARGET = 'validation_loss'

# =============================================================================
# end of configuration
# =============================================================================




# =============================================================================
# ParameterSet used in mlrMBO implementation for reference ...
# =============================================================================

# =============================================================================
# param.set <- makeParamSet(
#   makeNumericParam("learning_rate", lower= 0.00001, upper= 0.1 ),
#   makeNumericParam("dropout", lower= 0, upper= 0.9 ),
#   makeDiscreteParam("activation", 
#     values= c( "softmax","elu","softplus","softsign",
#                "relu", "tanh","sigmoid","hard_sigmoid",
#                "linear") ),
#   makeDiscreteParam("optimizer", 
#     values = c("adam", "sgd", "rmsprop","adagrad",
#                "adadelta")),
#   makeDiscreteParam("shared_nnet_spec", 
#     values= c( "400", "500", "600", "700" 
#                #"800", "900", "1000", "1100",  "1200", 
#                #"400,400", "500,500", "600,600", "700,700", 
#                #"800,800", "900,900", "1000,1000", "1100,1100", 
#                #"1200,1200" 
# 	       ) ),
#   makeDiscreteParam("ind_nnet_spec",
#     values= c( "400:400:400", "600:600:600" 
#                #"800:800:800", "1000:1000:1000",
#                #"1200:1200:1200",
#                #"400,400:400,400:400,400", "600,600:600,600:600,600", 
#                #"800,800:800,800:800,800", "1000,1000:1000,1000:1000,1000",
#                #"1200,1200:1200,1200:1200,1200",
#                #"800,400:800,400:800,400",
#                #"1200,400:1200,400:1200,400",
#                #"1200,800,400:1200,800,400:1200,800,400"
# 	       )),
#   makeDiscreteParam("batch_size", values = c(16,32,64,128,256)),
#   makeIntegerParam("epochs", lower = 5, upper = 50)
# )
# =============================================================================

# =============================================================================
# Global variables for ParameterGrid and ParameterSet
# =============================================================================
batch_size = [16, 32, 64, 128, 256]
#activation = ["softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
#optimizer = ["adam", "sgd", "rmsprop", "adagrad", "adadelta","adamax","nadam"]
shared_nnet_spec = ["400", "500", "600", "700", "1200"]
ind_nnet_spec = ["400:400:400", "600:600:600", "1200, 1200:1200, 1200:1200, 1200"]  
      
# =============================================================================
# ParameterGrid used for initial keras runs
# =============================================================================
def p3b1_parameter_grid():
    """Utility function to encapsulate ParameterGrid definition"""
    
    gdict = {"batch_size" : batch_size,
             "epochs" : [5, 25, 50],
             "dropout" : [0.05, 0.20, 0.50],
             "learning_rate" : [0.01, 0.05, 0.1],
             "shared_nnet_spec" : shared_nnet_spec,
             "ind_nnet_spec" : ind_nnet_spec
            }
    
    pg = ParameterGrid(gdict)
    return pg

# =============================================================================
# ParameterSet used for focus search after model fit
# =============================================================================
def p3b1_parameter_set(): 
    """Utility function to encapsulate ParameterSet definition"""
    
    ps = prs.ParameterSet()
    
    # switching batch_size to NumericList to enforce integer validation
    #ps.add(prs.DiscreteParameter("batch_size", batch_size))
    ps.add(prs.NumericListParameter("batch_size", batch_size))
    ps.add(prs.IntegerParameter("epochs", 5, 50))
    #ps.add(prs.DiscreteParameter("activation", activation))
    #ps.add(prs.DiscreteParameter("optimizer", optimizer))
    ps.add(prs.NumericParameter("dropout", 0.0, 0.9))
    ps.add(prs.NumericParameter("learning_rate", 0.00001, 0.1))
    ps.add(prs.DiscreteParameter("shared_nnet_spec", shared_nnet_spec))
    ps.add(prs.DiscreteParameter("ind_nnet_spec", ind_nnet_spec))
    
    return ps

# =============================================================================
# Object oriented repackaging of functionality
# =============================================================================
class Run_P3B1(object):
    """Runs p3b1_baseline_keras2 on given parameters, accumulates results
    
    run(): calls keras, which also generates run.***.json log files
    in save/output_subdirectory, optionally sends results to solr
    """
    # class variable for convenience id
    _run_id = 0
    
    def __init__(self,
                 default_params=DEFAULT_PARAMS,
                 output_subdirectory=OUTPUT_SUBDIRECTORY,
                 solr_url=None
                 ):
        """
        default_params:  typically read from p3b1_default_model.txt
        output_subdirectory: passed to keras as 'save' parameter
            output will be saved in 'save/output_subdirectory'
        solr_url: if present, passed to keras as 'solr', triggers callback
        which submits json results to solr
        """
        self.default_params = default_params
        self.output_subdirectory = output_subdirectory
        self.solr_url = solr_url
        self.data = defaultdict(list)
        self._df = None
        
    def run(self, params, run_id=None):
        if run_id is None:
            run_id = Run_P3B1._run_id
        run_params = self._parameter_update(run_id, params)
        self.data['run_id'].append(run_id)
        # TODO: switch print statements over to log files
        logging.debug("*"*80)
        logging.debug("* Parameters: ")
        for k, v in params.items():
            logging.debug("{:25}{}".format(k, v))
            self.data[k].append(v)
        logging.debug("*"*80)
        avg_loss = p3b1k2.do_n_fold(run_params)
        logging.debug("Average Loss: {}".format(avg_loss))
        self.data['validation_loss'].append(avg_loss)
        Run_P3B1._run_id += 1
        return avg_loss
        
    def _parameter_update(self, run_id, params):
        run_params = self.default_params.copy()
        run_params.update(params)
        # TODO: use os.path.join
        run_params['save'] = 'save/{}'.format(self.output_subdirectory)
        if self.solr_url:
            run_params['solr_root'] = self.solr_url
        run_params['run_id'] = "{}".format(run_id)
        # temporary workaround for P3B1 using a list of integers
        # force integer value, don't yet enforce list membership
        run_params['batch_size'] = int(run_params['batch_size'])
        return run_params

    @classmethod
    def run_id_start(cls, start_value):
        """Specify a starting value for run_id, for example when re-starting
        
        Avoid overwriting previous log files by keeping track of run_id
        """
        cls._run_id = start_value
        
    def _dataframe(self):
        """Converts the accumulated parameter values and results into a dataframe"""
        data = self.data
        run_id = data['run_id']
        df = pd.DataFrame(data, index=run_id)
        self.data = defaultdict(list)
        # TODO: decide if this is worth keeping track of everywhere
        df.drop('run_id', axis=1, inplace=True)
        if self._df is not None:
            self._df = pd.concat((self._df, df))
        else:
            self._df = df
        return self._df
    
    dataframe = property(_dataframe, None, None, "Lazy evaluation of dataframe")
    
    def __add__(self, other):
        # dataframe will ensure cached data are included in the dataframe
        # TODO: type checking, column checking, ...
        # TODO: consider import copy, copy.copy(self)
        run_sum = Run_P3B1(self.default_params, self.output_subdirectory, self.solr_url)
        if isinstance(other, Run_P3B1):
            df = pd.concat((self.dataframe, other.datframe))
            run_sum._df = df
        elif isinstance(other, pd.DataFrame):
            df = pd.concat((self.dataframe, other))
            run_sum._df = df
        elif isinstance(other, (defaultdict, dict)):
            assert all(isinstance(v, list) for v in other.values()), "Expecting a dict of lists"
            run_sum.data = other
            run_sum._df = self.dataframe
            df = run_sum.dataframe
        else:
            logging.error("incompatible types")
            assert False, "Add: type unsupported by Run_P3B1"
        return run_sum
            
# =============================================================================
#     def __add__(self, other):
#         assert set(self.data.keys() == set(other.data.keys())), "Inconsistent data"
#         data = defaultdict(list)
#         data0 = self.data
#         data1 = other.data
#         for k, v in data0.keys():
#             data[k] = data0[k] + data1[k]
#         run_sum = Run_P3B1(self.default_params, self.output_subdirectory, self.solr_url)
#         run_sum.data = data
#         return run_sum
# =============================================================================
 
        
        
# =============================================================================
# Evaluate average validation loss in keras
# =============================================================================
def run_p3b1(params):
    """Run p3b1_baseline_keras2
    
    params: dictionary of parameter values
    Returns: average loss
    """
    # TODO: switch print statements over to log files
    logging.debug("*"*80)
    logging.debug("* Parameters: ")
    for k, v in params.items():
        logging.debug("{:25}{}".format(k, v))
    logging.debug("*"*80)
    avg_loss = p3b1k2.do_n_fold(params)
    logging.debug("Average Loss: {}".format(avg_loss))
    return avg_loss

# =============================================================================
# Supply default values for parameters not included in parameter grid/set
# =============================================================================
# TODO: current setup assumes configuration through global variables,
# although it does allow overrides.  Find a less sleazy way to set this up
def param_update(run_id, params, default_params=DEFAULT_PARAMS, output_subdirectory=OUTPUT_SUBDIRECTORY):
    run_params = default_params.copy()
    run_params.update(params)
    run_params['save'] = 'save/{}'.format(output_subdirectory)
    #run_params['solr_root'] = "http://localhost:8983/solr"
    run_params['run_id'] = "{}".format(run_id)
    return run_params

# =============================================================================
# Generate initial data from a parameter grid, or cached from an earlier run
# =============================================================================
def initialize_data(cache_file=CACHE_FILE, sample=INITIAL_SAMPLE):
    """Create new data, or read from cache if present
    
    If no cache file is found, draws samples from a parameter grid
    Runs p3b1_baseline_keras2, writes results to cache
    (call with cache_file=None to omit saving results)
    Returns: pandas dataframe containing parameter values and average loss
    """
    # coerce data into correct types in dataframe
    float64 = 'float64'
    int64 =  'int64'
    category = 'category'
    pdtypes = {'batch_size': int64,
               'dropout': float64,
               'epochs': int64,
               'learning_rate': float64,
               'shared_nnet_spec' : category,
               'ind_nnet_spec' : category,
               'runtime_hours' : float64,
               'training_loss' : float64,
               'validation_loss' : float64
               }
    # TODO: consider .astype for _nnet_spec
    # s_cat = s.astype("category", categories=["b","c","d"], ordered=False)
    try:
        # read the cached data, if present
        logging.debug("Reading data from cache: {}".format(cache_file))
        df = pd.read_csv(cache_file, dtype=pdtypes)
        #df['shared_nnet_spec'] = df['shared_nnet_spec'].astype("category", categories=shared_nnet_spec)
        #df['ind_nnet_spec'] = df['ind_nnet_spec'].astype("category", categories=ind_nnet_spec)
    except:
        # create the initial data
        logging.debug("Generating data from keras, sampling {} parameters".format(sample))
        data = defaultdict(list)
        param_grid = p3b1_parameter_grid()
        for run_id, i in enumerate(np.random.choice(len(param_grid), sample, replace=False)):
            params = param_grid[i]
            run_params = param_update(run_id, params)
            avg_loss = run_p3b1(run_params)
            for k, v in params.items():
                data[k].append(v)
            data['validation_loss'].append(avg_loss)
        df = pd.DataFrame(data) #, dtype=pdtypes)
        # write to cache for next time
        if cache_file:
            df.to_csv(cache_file)
    return df
  
# =============================================================================
# Gaussian Process Regression
# =============================================================================
def gpr(data_df, X_columns, target=TARGET, factors=['shared_nnet_spec', 'ind_nnet_spec']):
    """Gaussian Process Regression model
    
    data_df:    dataframe with data to model
    X_columns:  list of parameter names to use as predictors
    target:     name of field to predict
    factors:    list of parameters to be treated as categorical
                creates full-rank indicator variables using pandas get_dummies
    """
    # TODO: move all the print statements over to log
    X = data_df[X_columns]
    y = data_df[target]
    
    # compare to GPR minimum below...
    yidxmin = y.idxmin()
    
    y_star = y.iloc[yidxmin]
    X_star = X.iloc[yidxmin]
    
    logging.debug("Best observed value")
    logging.debug("Index: {}".format(yidxmin))
    logging.debug(y_star)
    logging.debug("Parameter values for best observed:")
    logging.debug(X_star)
    
    # Create auxiliary dataframe with dummy-coded indicators 
    if factors:
        #data_with_dummies = prs.DataFrameWithDummies(X, dummies=['shared_nnet_spec', 'ind_nnet_spec'])
        #Xd = data_with_dummies.dataframe
        Xd = pd.get_dummies(X, columns=factors, prefix_sep=PREFIX_SEP) if factors else df
    else:
        Xd = X
        
    continuous_columns = []
    factor_columns = defaultdict(list)
    
    for i, name in enumerate(Xd.columns):
        n0 = name.split("|")[0]
        if n0 in factors:
            factor_columns[n0].append(i)
        else:
            continuous_columns.append(i)
    
    n_continuous = len(continuous_columns)
    
    # fairly generic Radial Basis Function kernel with scaling
    # fairly broad bounds on the hyperparameters (which are fit automatically) 
    #k = ConstantKernel(1.0, (0.001, 1000.0))
    kernel = ck.ProjectionKernel(RBF([1.0]*n_continuous, (0.001, 1000.0)), continuous_columns, name="continuous")
    
    if use_simple_categorical_kernel:
        kernel = ck.ProjectionKernel(RBF([1.0]*n_continuous, (0.001, 1000.0)), continuous_columns, name="continuous")
        for factor, columns in factor_columns.iteritems():
            dim = len(columns)
            kernel *= ck.ProjectionKernel(ck.SimpleCategoricalKernel(dim), columns, name=factor)
    else:
        factor_kernel = {}
        for factor, columns in factor_columns.iteritems():
            dim = len(columns)
            fk = ck.FactorKernel(dim, [2*pi+pi/4.]*(dim*(dim-1)/2))
            factor_kernel[factor] = fk
            kernel *= ck.ProjectionKernel(fk, columns, name=factor)

    
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   alpha=0.001,
                                   normalize_y=True,
                                   n_restarts_optimizer=20)
    #gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, normalize_y=True)
    
    gpr.fit(Xd, y)    
    # n.b. could scale y but for now handle y with normalize_y in GPR

    gpr.fit(Xd, y)
    
    logging.debug("Fit Gaussian Process Regression:\n", gpr.kernel_.get_params())
    
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
    #if plots:
    #    gprpred.plot(title="GPR predictions on training data")
    
    # Obtain bounds after standardization
    lower_bound = Xd.min(axis=0)
    upper_bound = Xd.max(axis=0)
    
    bounds = [(lower, upper) for lower, upper in zip(lower_bound, upper_bound)]
    
    # using location of best actual value (yidxmin) as starting point
    # could try alternative starting points and return the best
    # preliminary tests on NT3 did not show local minima are a problem
    # there is a problem with optimize which can generate Deprecation Warning
    # even when the starting point is 2d and has the correct shape
    start_val = X.iloc[yidxmin].as_matrix().reshape(-1,  1)
    # Fit the GPR model
    result = sp.optimize.minimize(gpr.predict, start_val, method='L-BFGS-B', bounds=bounds)
    rx = result.x
    
    columns = Xd.columns
    d = {col : val for col, val in zip(columns, rx)}
    
    # the dictionary will need to be decoded by a ParameterSet object
    return d


def gpr_search(data_df, X_columns, target=TARGET, 
               factors=['shared_nnet_spec', 'ind_nnet_spec'],
               sample=100, verbose=True):
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
    
    # compare to GPR minimum below...
    yidxmin = y.idxmin()
    
    y_star = y.iloc[yidxmin]
    X_star = X.iloc[yidxmin]
    
    logging.debug("Best observed value")
    logging.debug("Index: {}".format(yidxmin))
    logging.debug(y_star)
    logging.debug("Parameter values for best observed:")
    logging.debug(X_star)
    
    # Create auxiliary dataframe with dummy-coded indicators 
    if factors:
        #data_with_dummies = prs.DataFrameWithDummies(X, dummies=['shared_nnet_spec', 'ind_nnet_spec'])
        #Xd = data_with_dummies.dataframe
        Xd = pd.get_dummies(X, columns=factors, prefix_sep=PREFIX_SEP) if factors else df
    else:
        Xd = X
        
    continuous_columns = []
    factor_columns = defaultdict(list)
    
    for i, name in enumerate(Xd.columns):
        n0 = name.split("|")[0]
        if n0 in factors:
            factor_columns[n0].append(i)
        else:
            continuous_columns.append(i)
    
    n_continuous = len(continuous_columns)
    
    # fairly generic Radial Basis Function kernel with scaling
    # fairly broad bounds on the hyperparameters (which are fit automatically) 
    #k = ConstantKernel(1.0, (0.001, 1000.0))
    kernel = ck.ProjectionKernel(RBF([1.0]*n_continuous, (0.001, 1000.0)), continuous_columns, name="continuous")
    if use_simple_categorical_kernel:
        for factor, columns in factor_columns.iteritems():
            dim = len(columns)
            kernel += ck.ProjectionKernel(ck.SimpleCategoricalKernel(dim), columns, name=factor)    
    else:
        factor_kernel = {}
        for factor, columns in factor_columns.iteritems():
            dim = len(columns)
            fk = ck.FactorKernel(dim, [2*pi+pi/4.]*(dim*(dim-1)/2))
            factor_kernel[factor] = fk
            kernel *= ck.ProjectionKernel(fk, columns, name=factor)

    kernel += ck.ConstantKernel()
    
    gpr = GaussianProcessRegressor(kernel=kernel,
                                   alpha=0.001, 
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

# =============================================================================
# Construct a representation of the parameter set
# =============================================================================
ps = p3b1_parameter_set()

# =============================================================================
# Get the data
# =============================================================================
data = initialize_data()

logging.debug("Data for GPR")
logging.debug(data.columns)
logging.debug(data.describe())

# columns to use as predictors in the model
# non-numeric columns will be recoded as as indicator variables for GPR
X_col = ['batch_size', 'dropout', 'epochs', 'learning_rate',
         'shared_nnet_spec', 'ind_nnet_spec']
# columns omitted from X: ['run_id', 'runtime_hours', 'training_loss']
assert all(x in data.columns for x in X_col), "invalid column"

# =============================================================================
# Fit Gaussian Process Regression, optimize the model, return recommended
# parameter dictionary
# =============================================================================
d, gpr, Xd, gprpred, result_data = gpr_search(data, X_col, TARGET, factors=['shared_nnet_spec', 'ind_nnet_spec'])
logging.debug(d)
# =============================================================================
# The parameters need to be decoded to remove dummies, also validated
# to enforce bounds, and the correct data types(int or float)
# =============================================================================
dd = ps.decode_dummies(d)

# =============================================================================
# Begin amassing a bunch of candidates, starting with the GPR recommendation
# =============================================================================

if run_keras:
    run_params = []
        
    runner = Run_P3B1() # default initialization controlled by Configuration
    runner.run_id_start(data.shape[0])
    loss = runner.run(dd)
    df = runner.dataframe
    df.describe()
    
    for i in range(3):
        focus1 = ps.focus(dd)
        for j in range(3):
            params = focus1.draw()
            run_params.append(params)
            focus2 = focus1.focus(dd)
            for k in range(5):
                params = focus2.draw()
                run_params.append(params)
    
    for params in run_params:
        loss = runner.run(params)
    df = runner.dataframe
    df.describe()          

# =============================================================================
# # =============================================================================
# # Construct new parameter sets focussed at various degrees on the best so far
# # =============================================================================
# # more draws at higher levels will explore more broadly
# # more layers will narrow the focus tightly around the initial point
# for i in range(1):
#     focus1 = ps.focus(d)
#     for j in range(1):
#         params = focus1.draw()
#         run_params.append(params)
#         focus2 = focus1.focus(d)
#         for k in range(1):
#             params = focus2.draw()
#             run_params.append(params)
#             
# print("="*80, "\nDefault Parameters:\n", DEFAULT_PARAMS)
# avg_losses = []
# data = defaultdict(list)
# for i, params in enumerate(run_params):
#     run = param_update(i, params)
#     avg_loss = run_p3b1(run)
#     avg_losses.append(avg_loss)
#     for k, v in params.items():
#         data[k].append(v)
#     data['validation_loss'].append(avg_loss)
# 
# new_df = pd.DataFrame(data) #, dtype=pdtypes)
# print(new_df.describe())
# cache_dir = os.path.join(file_path, 'save')
# new_cache_file = os.path.join(cache_dir, "P3B1_cache_1.csv")
# new_df.to_csv(new_cache_file)
# 
# # =============================================================================
# # Now gather up the results from output_subdirectory and repeat
# # =============================================================================
# #new_p3b1_data = nt3d.P3B1RunData(output_dir=output_dir,
# #                           subdirectory=OUTPUT_SUBDIRECTORY,
# #                           pdtypes=pdtypes)
# #new_p3b1_data.add_all()
# ##new_data = nt3_data.dataframe
# #new_data_dict = new_p3b1_data.data   
# 
# 
# run_params_0 = run_params
# # data can be recovered with pd.read_csv(new_cache_file)
# run_params = []
# for i in range(3):
#     focus1 = ps.focus(d)
#     for j in range(3):
#         params = focus1.draw()
#         run_params.append(params)
#         focus2 = focus1.focus(d)
#         for k in range(3):
#             params = focus2.draw()
#             run_params.append(params)
#             
# print("="*80, "\nDefault Parameters:\n", DEFAULT_PARAMS)
# avg_losses = []
# data = defaultdict(list)
# for i, params in enumerate(run_params):
#     run_params = param_update(i+1100, params)
#     avg_loss = run_p3b1(run_params)
#     avg_losses.append(avg_loss)
#     for k, v in params.items():
#         data[k].append(v)
#     data['validation_loss'].append(avg_loss)
# 
# new_df = pd.DataFrame(data) #, dtype=pdtypes)
# print(new_df.describe())
# #cache_dir = os.path.join(file_path, 'save')
# new_cache_file = os.path.join(cache_dir, "P3B1_cache_3.csv")
# new_df.to_csv(new_cache_file)
# 
# =============================================================================
