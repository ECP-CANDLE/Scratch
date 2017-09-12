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
# =============================================================================

import os
import sys


def add_path(*args):
    """utility function for joining paths"""
    lib_path = os.path.abspath(os.path.join(*args))
    sys.path.append(lib_path)
    
file_path = os.path.dirname(os.path.realpath(__file__))

paths =  [['..', '..', 'Benchmarks', 'common'],
          ['..', '..', 'Benchmarks', 'Pilot1', 'common'],
          [ '..', '..', 'Benchmarks', 'Pilot1', 'NT3']]

for path in paths:
    p = [file_path]
    p.extend(path)
    add_path(*p)

from collections import defaultdict

import nt3_run_data as nt3d
import nt3_baseline_keras2 as nt3b

import parameter_set as prs

import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, ParameterSampler

import scipy as sp
from scipy.stats.distributions import expon

# =============================================================================
# data are correctly reshaped but warning is present any, so suppress them all
# =============================================================================
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 



# =============================================================================
# CONFIGURATION stuff done here for now
# =============================================================================

# if True, parameter dictionaries will be sent to nt3_baseline_keras2
run_keras = False

config_file = os.path.join(file_path, 'nt3_default_model.txt')
output_dir = os.path.join(file_path, 'save') 
output_subdirectory = 'experiment_0' 

# read in global default parametere configuration
default_params = nt3b.read_config_file(config_file)

# don't fit models with ridiculously large losses, as defined here
# there is at least one keras run with a validation loss > 8 
MAX_LOSS = 2

# the target is validation_loss, could be training_loss or runtime_hours
TARGET = 'validation_loss'

# =============================================================================
# ParameterSet in case we need it ...
# =============================================================================
batch_size = [16, 32, 64, 128, 256, 512]
#activation = ["softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
dense = [(500, 100, 50),
         (1000, 500, 100, 50),
         (2000, 1000, 500, 100, 50),
         (2000, 1000, 1000, 500, 100, 50),
         (2000, 1000, 1000, 1000, 500, 100, 50)]
#optimizer = ["adam", "sgd", "rmsprop", "adagrad", "adadelta","adamax","nadam"]
conv = [ (50, 50, 50, 50, 50, 1),
         (25, 25, 25, 25, 25, 1),
         (64, 32, 16, 32, 64, 1),
         (100, 100, 100, 100, 100, 1),
         (32, 20, 16, 32, 10, 1)]

ps = prs.ParameterSet()

ps.add(prs.DiscreteParameter("batch_size", batch_size))
ps.add(prs.IntegerParameter("epochs", 5, 100))
#ps.add(prs.DiscreteParameter("activation", activation))
ps.add(prs.DiscreteParameter("dense", dense))
#ps.add(prs.DiscreteParameter("optimizer", optimizer))
ps.add(prs.NumericParameter("drop", 0.0, 0.9))
ps.add(prs.NumericParameter("learning_rate", 0.00001, 0.1))
ps.add(prs.DiscreteParameter("conv", conv))
# TODO: since dense and conv will be dummy-coded, ensure that all possible
# category values are present in the parameter set

# =============================================================================
# DATA
# =============================================================================

# =============================================================================
# Create a ParameterSet, construct initial parameter grid,
# generate parameter dicitionaries,
# run keras on each,
# recover the output from JSON logs or solr,
# ... retrieve the result from earlier runs
# =============================================================================

# =============================================================================
# TODO: move this to nt3_run_data
# =============================================================================


# coerce data into correct types in dataframe
float64 = 'float64'
int64 =  'int64'
pdtypes = {'batch_size': int64,
           'drop': float64,
           'epochs': int64,
           'learning_rate': float64,
           'run_id': int64,
           'runtime_hours' : float64,
           'training_loss' : float64,
           'validation_loss' : float64
           }

def get_nt3_data(output_dir=output_dir,
             subdirectory=output_subdirectory,
             dtype=pdtypes,
             from_csv=False):
    """Assumes json log files from a previous run are available

    If json logs are in output_subdirectory, call with from_csv=False
    Otherwise it will read cached data from a previous run from a csv file
    """

    if from_csv:
        data = pd.read_csv("nt3_initial_data.csv", dtype=dtype)
    else:
        nt3_data = nt3d.NT3RunData(output_dir=output_dir,
                                   subdirectory=output_subdirectory)
        nt3_data.add_all()
        data = nt3_data.dataframe
    
    data = data[data.validation_loss < MAX_LOSS]
    return data

data = get_nt3_data(from_csv=True)

#print(data.columns)

# columns to use as predictors in the model
# non-numeric columns will be recoded as as indicator variables for GPR
X_col = ['batch_size', 'conv', 'dense', 'drop', 'epochs', 'learning_rate']
# columns omitted from X: ['run_id', 'runtime_hours', 'training_loss']
assert all(x in data.columns for x in X_col), "invalid column"

# preliminary input for modelling
X = data[X_col]
y = data[TARGET]


# =============================================================================
# rfr = RandomForestRegressor(oob_score=True)
# rfr.fit(X, y)
# 
# pred = pd.DataFrame({"pred" : rfr.predict(X),
#                      "oob_pred" : rfr.oob_prediction_,
#                      "y" : y })
# 
# importance = [z for z in zip(data.columns, rfr.feature_importances_)]
# importance = sorted(importance, key=lambda x : x[1], reverse=True)
# print("Feature Importances")
# print("{:20}\t{:5}".format("Parameter:", "Value:"))
# for param, imp in importance[:10]:
#     print("{:20}\t{:5.3f}".format(param, imp))
# =============================================================================

yidxmin = y.idxmin()

y_star = y.iloc[yidxmin]
X_star = X.iloc[yidxmin]

print("Best observed value")
print("Index: {}".format(yidxmin))
print(y_star)
print("Parameter values for best observed:")
print(X_star)

# =============================================================================
# Get estimate of std error for predictions from model estimators
# =============================================================================
# n.b. lambda is a keyword so lmbda vector of values
def LCB(forest, X, number_to_sample, use_min=False):
    est = forest.estimators_
    forestpreds = { "pred_{}".format(i) : est.predict(X) for i, est in enumerate(est)}
    forestpreds = pd.DataFrame(forestpreds)
    preds = pd.DataFrame({"pred" : forest.predict(X),
                          "stds" : forestpreds.std(axis=1)})
    if use_min:
        preds['pred'] = forestpreds.min(axis=1)    
    lmbda = (ParameterSampler({ "lm" : expon()}, n_iter=number_to_sample))
    lcb = pd.DataFrame({ "lcb_{}".format(i) : preds.pred - li["lm"] * preds.stds for i, li in enumerate(lmbda) })
    # TODO: include X in lcb, to look up parameters from selected values
    return lcb


# =============================================================================
# Preliminary attempt at Gaussian Process Regresssion
# =============================================================================

# =============================================================================
# Create auxiliary dataframe with dummy-coded indicators for dense, conv...
# =============================================================================

# TODO: rework DataFrameWithDummies to implement fit, transform
data_with_dummies = prs.DataFrameWithDummies(X, dummies=['conv', 'dense'])
Xd = data_with_dummies.get_dataframe()

# standardized data
scaler = StandardScaler()
Xs = scaler.fit_transform(Xd)
# CAREFUL X and Xd are DataFrame objects but Xs is a numpy array
# could convert to dataframe here

# n.b. could scale y but for now handle y with normalize_y in GPR

k = ConstantKernel(1.0, (0.001, 1000.0)) * RBF(1.0, (0.01, 100.0))
gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, n_restarts_optimizer=20, normalize_y=True)
gpr.fit(Xs, y)

print(gpr.kernel_.get_params())

gprpred = pd.DataFrame({"pred" : gpr.predict(Xs)})
print("GPR predictions")
print(gprpred.describe())
gpidx = gprpred.idxmin()
y_gstar = y.iloc[gpidx]
print("observed value corresponding to best prediction")
print(y_gstar)
X_gstar = X.iloc[gpidx]
print("Parameter values for best prediction")
print(X_gstar)
gprpred.plot(title="GPR predictions")

# =============================================================================
# Xgs = pd.DataFrame(scaler.transform(Xgrid), columns=Xgrid.columns)
# grid_pred, grid_se = gpr.predict(Xgs, return_std=True)
# gprpredgrid = pd.DataFrame({"pred_gpr" : grid_pred, "pred_se" : grid_se})
# print("GPR predictions on larger grid")
# print(gprpredgrid.describe())
# 
# ggpidx = gprpredgrid.pred_gpr.idxmin()
# X_ggstar = Xgrid.iloc[ggpidx]
# pred_ggstar = gprpredgrid.pred_gpr.iloc[ggpidx] #gpr.predict(Xgs.iloc[ggpidx])
# print("Predicted GPR minimum")
# print(pred_ggstar)
# print("Parameter values at GPR-pred min")
# print(X_ggstar)
# gprpredgrid.plot(title="GPR predictions on expanded grid")
# =============================================================================


# =============================================================================
# Get estimate of std error for predictions from model
# =============================================================================
# n.b. lambda is a keyword so lmbda vector of values
def GPR_LCB(gpr, X, number_to_sample):
    # X should be scaled!!!
    pred, se = gpr.predict(X, return_std=True)
    preds = pd.DataFrame({"pred" : pred, "se" : se})
    lmbda = (ParameterSampler({ "lm" : expon()}, n_iter=number_to_sample))
    lcb = pd.DataFrame({ "lcb_{}".format(i) : preds.pred - li["lm"] * preds.se for i, li in enumerate(lmbda) })
    return lcb

gprLCB = GPR_LCB(gpr, Xs, 5)
print("GPR Lower Confidence Bounds")
gprLCB.describe()
gprLCB.plot(title="GPR Lower Confidence Bounds")

print("Parameter values of LCB selected best point")
print(X.iloc[gprLCB.idxmin()])

#file_path = os.path.dirname(os.path.realpath(__file__))
#config_file = os.path.join(file_path, 'nt3_default_model.txt')
#
#default_params = nt3b.read_config_file(config_file)

candidate_LCB = set()
for col in gprLCB.columns:
    gprsorted = gprLCB.sort_values(by=col)
    ids = gprsorted.head(15).index
    #print(col, ids)
    candidate_LCB.update(ids)
    
def param_update(params, default_params, run_id, output_subdirectory='exp'):
    run_params = default_params.copy()
    run_params.update(params)
    run_params['save'] = 'save/{}'.format(output_subdirectory)
    #run_params['solr_root'] = "http://localhost:8983/solr"
    run_params['run_id'] = "run.{}.json".format(run_id)
    return run_params

# =============================================================================
# Extract the candidates from Lower Confidence Bounds
# However, for GPR on NT3 the standard error is very uniform
# so these are could just as well be obtained as the points with the lowest loss
# For Random Forest Regression, evaluate on many points of a grid
# =============================================================================
# TODO: stop looking up index in original!  Does not generalize
# give them unique ids by specifying a start value
start_id = 5000
for i, c in enumerate(candidate_LCB):
    d = dict(X.iloc[c])
    # give them unique run_id values in case they actually get used...
    params = param_update(d, default_params, i+start_id, output_subdirectory)
    # TODO: translate categorical variables' parameters back to original names
    print("="*80)
    print("index: {:4d} loss: {}".format(c, y.iloc[c]))
    print(params)
    # since these already have known losses, don't send to keras
    #nt3b.run(params)

# this needs to be automated, probably use a dict or OrderedDict
# Xd requires 14 columns worth of bounds
# (alternatively, use a Scaler object from sklearn.preprocessing
lower_bound = Xs.min(axis=0)
upper_bound = Xs.max(axis=0)

bounds = [(lower, upper) for lower, upper in zip(lower_bound, upper_bound)]

# using location of best actual value (yidxmin) as starting point
start_val = Xs[yidxmin].reshape(-1,  1)
result = sp.optimize.minimize(gpr.predict, start_val, method='L-BFGS-B', bounds=bounds)
rx = scaler.inverse_transform(result.x)
columns = Xd.columns
d = {col : val for col, val in zip(columns, rx)}
d = ps.decode_dummies_dict(d)
# workaround for batch_size requiring int but using DiscreteParameter
d['batch_size'] = int(d.get('batch_size', 16))

# TODO: using a random draw from the parameter set as starting point
#start_val = ps.draw()
# TODO: set up parameter set with fit, transform, inverse_transform or comparable
#start_val = ps.transform(start_val)
#result = sp.optimize.minimize(gpr.predict, Xs[yidxmin].reshape(-1,  1), method='L-BFGS-B', bounds=bounds)
#rx = scaler.inverse_transform(result.x)
#columns = Xd.columns
#d = {col : val for col, val in zip(columns, rx)}
#d = ps.decode_dummies_dict(d)

# =============================================================================
# Begin amassing a bunch of candidates, starting with the GPR recommendation
# =============================================================================
run_params = []
params = param_update(d, default_params, len(run_params), output_subdirectory)
#print(params)
run_params.append(params)
#nt3b.run(params)

## use candidate_LCB as a convenient set of starting points
#results = []
#for c in candidate_LCB:
#    result = sp.optimize.minimize(gpr.predict, Xs[c].reshape(-1,  1), method='L-BFGS-B', bounds=bounds)
#    rx = scaler.inverse_transform(result.x)
#    columns = Xd.columns
#    d = {col : val for col, val in zip(columns, rx)}
#    results.append(ps.decode_dummies_dict(d))
#    d['batch_size'] = int(d.get('batch_size', 16))
#    print(results[-1])
    
# =============================================================================
# Construct new parameter sets focussed at various degrees on the best so far
# =============================================================================
# more draws at higher levels will explore more broadly
# more layers will narrow the focus tightly around the initial point
for i in range(3):
    focus = ps.focus(d)
    for j in range(3):
        params = param_update(focus.draw(), default_params, output_subdirectory)
        run_params.append(params)
        focus = ps.focus(d)
        for k in range(2):
            params = param_update(focus.draw(), default_params, output_subdirectory)
            run_params.append(params)
            
            
for params in run_params:
    print("*"*80)
    print("* Parameters: ")
    for k, v in params.items():
        print("{:25}{}".format(k, v))
    print("*"*80)
    if run_keras:
        # finally do some work!
        nt3b.run(params)

# =============================================================================
# Now gather up the results from output_subdirectory and repeat
# =============================================================================
nt3_data = nt3d.NT3RunData(output_dir=output_dir,
                           subdirectory=output_subdirectory,
                           pdtypes=pdtypes)
nt3_data.add_all()
#new_data = nt3_data.dataframe
new_data_dict = nt3_data.data   