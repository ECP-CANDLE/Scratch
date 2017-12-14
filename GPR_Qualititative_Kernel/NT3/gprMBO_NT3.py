#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 12 17:10:28 2017

@author: johnbauer
"""

from __future__ import absolute_import
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
         "NT3" : ['..', '..', '..', 'Benchmarks', 'Pilot1', 'NT3'],
         "parent" : ['..']
        }

for path in paths.values():
    lib_path = os.path.abspath(os.path.join(*[file_path]+path))
    sys.path.append(lib_path)


import logging
logging.basicConfig(filename='NT3.log',level=logging.DEBUG)

import run_data
import nt3_baseline_keras2 as nt3b

import parameter_set as prs
#import qualitative_kernels as qk
from gpr_model import GPR_Model, report



#from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel
#from kernels import RBF, ConstantKernel
from sklearn.model_selection import ParameterGrid  #, ParameterSampler

import random
import pandas as pd
import numpy as np
#import scipy as sp
#from scipy.stats.distributions import expon

# data are correctly reshaped but warning is present anyway, 
# so suppress them all (bug in sklearn.optimize reported)
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# =============================================================================
# CONFIGURATION done here for now
# =============================================================================

# if True, parameter dictionaries will be sent to p1b1_baseline_keras2
run_keras = True

# Location of saved output
output_dir = os.path.join(file_path, 'save')

config_file_path = os.path.join(*[file_path]+paths["NT3"])
CONFIG_FILE = os.path.join(config_file_path, 'nt3_default_model.txt')
# read in global default parameter configuration
DEFAULT_PARAMS = nt3b.read_config_file(CONFIG_FILE)

# don't include points with ridiculously large losses, as defined here
# when fitting GPR model
MAX_LOSS = 5

# the target is validation_loss, could be training_loss or runtime_hours
TARGET = 'validation_loss'

# Number of parameter sets to explore from a cold start
N_INITIAL = 60

# for demonstration purposes, limit the number of keras runs
# set to any number larget than N_INTIAL to use all the data
MAX_DEMO = 1000

from collections import defaultdict

import nt3_baseline_keras2 as nt3b

import parameter_set as prs


import scipy as sp
from scipy.stats.distributions import expon

# =============================================================================
# data are correctly reshaped but warning is present any, so suppress them all
# =============================================================================
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

# =============================================================================
# # =============================================================================
# # CONFIGURATION stuff done here for now
# # =============================================================================
# 
# file_path = os.path.dirname(os.path.realpath(__file__))
# config_file = os.path.join(file_path, 'nt3_default_model.txt')
# output_dir = os.path.join(file_path, 'save') 
# subdirectory = "exp_4" #'experiment_0' 
# 
# # read in global default parametere configuration
# default_params = nt3b.read_config_file(config_file)
# 
# # don't fit models with ridiculously large losses, as defined here
# # there is at least one keras run with a validation loss > 8 
# MAX_LOSS = 2
# 
# # the target is validation_loss, could be training_loss or runtime_hours
# TARGET = 'validation_loss'
# =============================================================================


# =============================================================================
# ParameterSet modelled on nt3_param_set.R
# =============================================================================
batch_size = [16, 32, 64, 128, 256, 512]
activation = ["softmax", "elu", "softplus", "softsign", "relu", "tanh",
              "sigmoid", "hard_sigmoid", "linear"]
dense = [[500, 100, 50],
         [1000, 500, 100, 50],
         [2000, 1000, 500, 100, 50],
         [2000, 1000, 1000, 500, 100, 50],
         [2000, 1000, 1000, 1000, 500, 100, 50]]
optimizer = ["adam", "sgd", "rmsprop", "adagrad", "adadelta","adamax","nadam"]
conv = [[50, 50, 50, 50, 50, 1],
        [25, 25, 25, 25, 25, 1],
        [64, 32, 16, 32, 64, 1],
        [100, 100, 100, 100, 100, 1],
        [32, 20, 16, 32, 10, 1]]

ps = prs.ParameterSet()

ps["batch_size"] = prs.DiscreteParameter(batch_size)
ps["epochs"] = prs.IntegerParameter(5, 500)
ps["activation"] = prs.DiscreteParameter(activation)
ps["dense"] = prs.DiscreteParameter(dense)
ps["optimizer"] = prs.DiscreteParameter(optimizer)
ps["drop"] = prs.NumericParameter(0.0, 0.9)
ps["learning_rate"] = prs.NumericParameter(0.00001, 0.1)
ps["conv"] = prs.DiscreteParameter(conv)
# TODO: since dense and conv will be dummy-coded, ensure that all possible
# category values are present in the parameter set

print(ps)

# =============================================================================
# Add run_id and subdirectory of /save for logged output.
# Ensure that all parameters are populated with default values
# Any last-minute or ad hoc changes can be added here
# =============================================================================
def param_update(params, default_params, run_id, subdirectory='exp'):
    run_params = default_params.copy()
    run_params.update(params)
    run_params['save'] = 'save/{}'.format(subdirectory)
    #run_params['solr_root'] = "http://localhost:8983/solr"
    run_params['run_id'] = "{}".format(run_id)
    # TODO: find a better fix for this:
    # seems to be expecting a list, could edit default params instead
    ###run_params['metrics'] = ['accuracy']
    return run_params


# =============================================================================
# Use ParameterGrid or ParameterSampler if cached data are not being used
# to fit the GPR model
# =============================================================================
grid = {"activation" : activation,
         "batch_size" : batch_size,
         "dense" : dense,
         "drop" : [0.0, 0.90],
         "epochs" : [5, 500],
         "learning_rate" : [0.00001, 0.1],
         "optimizer" : optimizer,
         "conv" : conv
        }
# =============================================================================
# Can't do a grid search over the full dictionary:
# =============================================================================
param_grid = ParameterGrid(grid)
print(len(param_grid))
75600

def sample_iterator(param_grid, k):
    """ Draw a sample of size k from the grid.  If k exceeds the number
        of points in the grid, returns all the points in random order.
    """
    grid_size = len(param_grid)
    k = min(k, grid_size)
    sample = random.sample(range(grid_size), k)
    for i in sample:
        yield param_grid[i]

#for params in sample_iterator(param_grid, 5):
#    print(params)
            
# =============================================================================
# Ensure that extremes of continuous variables are represented in the training
# sample.  GPR will not extrapolate beyond the region it is trained on.  Also 
# ensure all discrete parameters are represented at least once.
# Each dict in the list will be iterated over separately.
# =============================================================================
boundary = [{"epochs": [5, 500],
            "drop" : [0.0, 0.9],
            "learning_rate" : [0.00001, 0.1]},
            {"activation" : activation },
            {"batch_size" : batch_size},
            {"dense" : dense},
            {"optimizer" : optimizer},
            {"conv" : conv}]

param_grid = ParameterGrid(boundary)

run_params = []
for boundary_params in param_grid:
    # get a complete random set of parameters
    p = ps.draw()
    # replace the continuous values with points from the boundary hypercube,
    # then each of the discrete parameter values
    p.update(boundary_params)
    p = param_update(p, DEFAULT_PARAMS, len(run_params),
                     subdirectory='experiment')
    run_params.append(p)
    
# Fill out the requested number of initial points at random
for i in range(N_INITIAL - len(run_params)):
    p = ps.draw()
    p = param_update(p, DEFAULT_PARAMS, len(run_params),
                     subdirectory='experiment')
    run_params.append(p)
    
for params in run_params[:MAX_DEMO]:
    print("*"*80)
    print("* Parameters: ")
    for k, v in params.items():
        print("{:25}{}".format(k, v))
    print("*"*80)
    if run_keras:
        # finally do some work!
        nt3b.run(params)
       
subdirectory="experiment"
nt3_data = run_data.NT3RunData(output_dir, subdirectory)
nt3_data.add_run_id("*")
df = nt3_data.dataframe
print(df.describe())

# Add in the cached data and fit the GPR model

nt3csv = "nt3_initial_data.csv"

nt3_data.from_csv(nt3csv)
df = nt3_data.dataframe
print(df.describe())

# fit the Gaussian Procww Regression model

subdirectory="experiment_1"

X_columns = ['drop',
             'epochs',
             'learning_rate']

factors =['optimizer',
          'batch_size',
          'activation',
          'dense',
          'conv']

gpr_model = GPR_Model(df, X_columns, TARGET, factors)
gpr_model.fit_EC()
gpr_model.fit_MC()
gpr_model.fit_UC()
# equivalent to : gpr_model.fit()
