#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Oct 18 14:36:59 2017

@author: johnbauer
"""
from __future__ import print_function

# =============================================================================
# Assumes GPR_Qualitative_Kernel is located in Scratch/GPR_Qualitative_Kernel
# at the same level as Benchmarks:
#
# GitHub
#    Scratch
#        GPR_Qualitative_Kernel
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

paths = {"common" : ['..', '..', '..', 'Benchmarks', 'common'],
         "P1_common" : ['..', '..', '..', 'Benchmarks', 'Pilot1', 'common'],
         "P1B1" : ['..', '..', '..', 'Benchmarks', 'Pilot1', 'P1B1'],
         "parent" : ['..']
        }

for path in paths.values():
    lib_path = os.path.abspath(os.path.join(*[file_path]+path))
    sys.path.append(lib_path)


import logging
logging.basicConfig(filename='P1B1.log', level=logging.DEBUG)

import run_data
import p1b1_baseline_keras2 as p1b1k2
import p1b1
#import run_data
import parameter_set as prs
#import qualitative_kernels as qk
from gpr_model import GPR_Model, report



#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel
#from kernels import RBF, ConstantKernel
from sklearn.model_selection import ParameterGrid  #, ParameterSampler

import pandas as pd
import numpy as np
#import scipy as sp
#from scipy.stats.distributions import expon

# data are correctly reshaped but warning is present anyway, 
#so suppress them all (bug in sklearn.optimize reported)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# =============================================================================
# CONFIGURATION done here for now
# =============================================================================

# if True, parameter dictionaries will be sent to p1b1_baseline_keras2
run_keras = False

# Location of saved output
output_dir = os.path.join(file_path, 'save')

config_file_path = os.path.join(*[file_path]+paths["P1B1"])
CONFIG_FILE = os.path.join(config_file_path, 'p1b1_default_model.txt')
# read in global default parameter configuration
DEFAULT_PARAMS = p1b1.read_config_file(CONFIG_FILE)

# don't include points with ridiculously large losses, as defined here
# when fitting GPR model
MAX_LOSS = 5

# the target is validation_loss, could be training_loss or runtime_hours
TARGET = 'validation_loss'

# =============================================================================
# From P1B1_param_set.R
# =============================================================================
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
dense = [
 [1800, 600],
 [978, 978, 978, 978, 978, 978],
 [978, 978, 978, 978, 978],
 [978, 978, 978, 978],
 [978, 978, 978],
 [978, 978]]
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
# ParameterGrid used for iteration, enumeration, and grid search
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
 
    # batch_size is NumericList to enforce integer validation
    ps["activation"] = prs.DiscreteParameter(activation)
    ps["batch_size"] = prs.NumericListParameter(batch_size)
    ps["dense"] = prs.DiscreteParameter(dense)
    ps["drop"] = prs.NumericParameter(0.0, 0.9)
    ps["epochs"] = prs.IntegerParameter(10, 20) #100, 200)
    ps["latent_dim"] = prs.NumericListParameter(latent_dim)
    ps["learning_rate"] = prs.NumericParameter(0.00001, 0.1)
    ps["model"] = prs.DiscreteParameter(model)
    ps["optimizer"] = prs.DiscreteParameter(optimizer)
    ps["residual"] = prs.DiscreteParameter(residual)
    ps["reduce_lr"] = prs.DiscreteParameter(reduce_lr)
    ps["warmup_lr"] = prs.DiscreteParameter(warmup_lr)
    
    return ps

# =============================================================================
# Always call param_update before sending parameter dictionary to keras
# =============================================================================
def param_update(params, default_params, run_id, output_subdirectory='exp'):
    """Last-minute ammendations to the parameters.  
    
    Many parameters arguably belong in, but are missing from 
    p1b1_default_model.txt: 
        alpha_dropout, logfile, verbose, shuffle, datatype, cp, tb, tsne

    ChainMap in Python 3 would be a good replacement"""
    run_params = default_params.copy()
    run_params.update(params)
    run_params['save'] = 'save/{}'.format(output_subdirectory)
    #run_params['solr_root'] = "http://localhost:8983/solr"
    run_params['run_id'] = run_id # "run.{}.json".format(run_id)
    # TODO: should these be in default_params?
    # If they are supplied in either params or default_params, those values
    # will take precedence over the values below
    run_params['alpha_dropout'] = run_params.get('alpha_dropout', 0)
    run_params['use_landmark_genes'] = run_params.get('use_landmark_genes', True)
    run_params['logfile'] = run_params.get('logfile', 'placeholder.log')
    run_params['verbose'] = run_params.get('verbose', True)
    run_params['shuffle'] = run_params.get('shuffle', True)
    run_params['datatype'] = run_params.get('datatype', np.float64) #'f32' # DEFAULT_DATATYPE
    run_params['cp'] = run_params.get('cp', False)
    run_params['tb'] = run_params.get('tb', False)
    run_params['tsne'] = run_params.get('tsne', False)
    
    # TODO: fix restrict_model so it works properly
    model = run_params.get('model', False)
    if not model:
        # current hack should have put 'ae' or 'vae' in default_params
        # but just make sure something is there...
        run_params['model'] = default_params.get('model', 'vae')
    
    return run_params

# =============================================================================
# Python implementation of Focus algorithm from mlrMBO
# =============================================================================
def focus_search(params,
                 default_params,
                 output_subdirectory,
                 run_params=[],
                 n_recommend=1,
                 degree=1):
    for i in range(degree):
        focus = ps.focus(params)
    # low-rent strategy for generating a unique id: use length of run_params
    for j in range(n_recommend):
        params = param_update(focus.draw(), default_params, len(run_params), output_subdirectory)
        run_params.append(params)
    return run_params
                
                
if __name__ == "__main__":
    # parameter grid would be used to generate initial data
    # not currently used because data are read from cached .csv file
    pg = p1b1_parameter_grid()
    ps = p1b1_parameter_set()
    
    print(pg)
    print(ps)
    
    p1b1csv = "P1B1_data.csv"
    
    p1b1_run_data = run_data.P1B1RunData(output_dir, subdirectory="opt")
    p1b1_run_data.from_csv(p1b1csv)
    print("Testing .from_csv")
    print(p1b1_run_data.dataframe.describe())
    
    #p1b1_data = pd.read_csv(p1b1csv)
    p1b1_data = p1b1_run_data.dataframe
    valid = p1b1_data.validation_loss.notnull()
    p1b1_data = p1b1_data[valid]
    
    print(p1b1_data.describe())
       
# =============================================================================
# After inspecting the data, it seems that the overwhelming majority are < 1
# but there are some really big ones in there
# =============================================================================
    p1b1_data = p1b1_data[p1b1_data.validation_loss < 1]

# =============================================================================
# TODO: use a train/test split, cross-validate model, compute score
# =============================================================================

# =============================================================================
# To work with a subset of the 1046 points remaining after the above:
# =============================================================================
    subset = [i for i in range(len(p1b1_data)) if i % 5 == 0]
    p1b1_data = p1b1_data.iloc[subset]


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
             'dense',
             'latent_dim',
             'reduce_lr',
             'model'
             ]
    
# =============================================================================
#  SEE NOTE in ParameterSet.R:
#    Current best val_corr: 0.96 for ae, 0.86 for vae
#    We are more interested in vae results
# =============================================================================
    restrict_model = 'vae' # None or False or 'vae' or 'ae'
    if restrict_model in ('ae', 'vae'):
        p1b1_data = p1b1_data[p1b1_data.model == restrict_model]
        factors.remove('model')
    
    # try a smaller problem
    #factors = ['dense', 'model', 'warmup_lr', 'reduce_lr']
    assert all(x in data_columns for x in X_columns), "invalid column"
    
    gpr_model = GPR_Model(p1b1_data, X_columns, TARGET, factors)
    gpr_model.fit_EC()
    gpr_model.fit_MC()
    gpr_model.fit_UC()
    # equivalent to : gpr_model.fit()
    
    print("\nExchangeable Correlations")
    report(gpr_model.gpr_ec)
    print("\nMultiplicative Correlations")
    report(gpr_model.gpr_mc)
    print("\nUnrestrictive Correlations")
    report(gpr_model.gpr_uc)
    
    print("\nExchangeable Correlations")
    print(gpr_model.name_report(gpr_model.gpr_ec))
    print("\nMultiplicative Correlations")
    print(gpr_model.name_report(gpr_model.gpr_mc))
    print("\nUnrestrictive Correlations")
    print(gpr_model.name_report(gpr_model.gpr_uc))
    
  
    lcb_rec = gpr_model.LCB_recommend(param_set=ps, max_recommend=3)
    opt_rec, x_rec = gpr_model.optimize_recommend(param_set=ps,
                                                  return_data=True)
    
    # optimize_recommend finds all local minima; all points returned
    # could have converged to the same point, and be very close to each other.
    # Recommendations are clustered with Affinity Propagation
    # and the 'most-representative' results returned.  The number of
    # clusters is determined automatically.

    # Use the default model read in from Benchmarks/P1B1
    default_params = DEFAULT_PARAMS
    # Read the local default model if changes are made
    #default_params = p1b1.read_config_file("p1b1_default_model.txt")
    
    if restrict_model in ('ae', 'vae'):
        default_params.update({'model' : restrict_model})
        
    # TODO: last-minute additions to default_params could be done here, e.g.
    #default_params['logfile'] = 'logfile.txt'
    
    # randomize draws in the vicinity of LCB points, since the original
    # points have already been evaluated
    
    # send opt and lcb recommendations to different subdirectories
    # to facilitate strategy comparison

    run_params = []
    # len(run_parsms) is used to generate unique ids of the form run.0.json
    for param_dict in opt_rec:
        run_params.append(param_update(param_dict, default_params,
                                       len(run_params), "opt_0"))
        
    # note focus_search calls param_update
    for param_dict in lcb_rec:
        run_params = focus_search(param_dict, default_params, "lcb_0", run_params,
                                  n_recommend=1, degree=5)
    
    for params in run_params[:3]:
        print(params)

# TODO: Fix This!
# datatype=np.float32 gets keras to run, but can't be serialized to json
#    # write parameter dictionaries to json file, e.g. to be sent to swift-t
#    import json
#    with open("p1b1_recommend.json", "w") as jsonfile:
#        json.dump(run_params, jsonfile)
   
    if run_keras:
        for params in run_params:
            try:
                p1b1k2.run(params)
            except Exception as e:
                logging.error(repr(e))
                logging.error("\nKeras run failed for parameters:\n{}".format(params))
    
    if run_keras:
        data = run_data.P1B1RunData(output_dir, subdirectory="opt")
        data.add_run_id("*")
        print("Results from optimization")
        print(data.dataframe.validation_loss.describe())
        data.subdirectory = "lcb"
        data.add_run_id("*")
        print("Results from optimization and lower confidence bound")
        print(data.dataframe.validation_loss.describe())        
    