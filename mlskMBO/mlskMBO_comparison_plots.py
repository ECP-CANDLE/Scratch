#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 25 13:15:04 2017

@author: johnbauer
"""

# =============================================================================
# HISTORY LOG hope the kernel doesn't crash again!
# =============================================================================
## ---(Thu Sep 14 10:50:58 2017)---
from __future__ import absolute_import
from __future__ import print_function
"""
Created on Thu Sep  7 11:10:38 2017

@author: johnbauer
"""

import os
from collections import defaultdict

import nt3_run_data as nt3d
import nt3_baseline_keras2 as nt3b

import parameter_set as prs

import pandas as pd

from sklearn.ensemble import RandomForestRegressor

from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ParameterGrid, ParameterSampler

import scipy as sp
from scipy.stats.distributions import expon

# =============================================================================
# data are correctly reshaped but warning is present any, so suppress them all
# =============================================================================
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 

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

print(ps)
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
data = pd.read_csv("nt3_initial_data.csv", dtype=pdtypes)
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

data = pd.read_csv("nt3_initial_data.csv", dtype=pdtypes)
data.shape
data = data[data.validation_loss < MAX_LOSS]
print(data.columns)

# columns to use as predictors in the model
# non-numeric columns will be recoded as as indicator variables for GPR
X_col = ['batch_size', 'conv', 'dense', 'drop', 'epochs', 'learning_rate']
# columns omitted from X: ['run_id', 'runtime_hours', 'training_loss']
assert all(x in data.columns for x in X_col), "invalid column"

# preliminary input for modelling
X = data[X_col]
y = data[TARGET]
MAX_LOSS = 2
TARGET = 'validation_loss'

file_path = "/Users/johnbauer/Benchmarks/Pilot1/NT3"
config_file = os.path.join(file_path, 'nt3_default_model.txt')
output_dir = os.path.join(file_path, 'save') 
output_subdirectory = "exp_x" #'experiment_0' 

# read in global default parametere configuration
default_params = nt3b.read_config_file(config_file)
X_col = ['batch_size', 'conv', 'dense', 'drop', 'epochs', 'learning_rate']
# columns omitted from X: ['run_id', 'runtime_hours', 'training_loss']
assert all(x in data.columns for x in X_col), "invalid column"

# preliminary input for modelling
X = data[X_col]
y = data[TARGET]

conv = X.conv.copy()
convset = set(c for c in conv)
conv = conv.replace(convset, range(len(convset)))
conv.describe()
dense = X.dense.copy()
denseset = set(d for d in dense)
dense = dense.replace(list(denseset), range(len(denseset)))
dense.describe()

X = pd.concat([X[['batch_size', 'drop', 'epochs', 'learning_rate']], dense, conv], axis=1)
X.describe()
rfr = RandomForestRegressor(oob_score=True)
# =============================================================================
# rfr.fit(X, y)
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
# def LCB(forest, X, number_to_sample, use_min=False):
#     est = forest.estimators_
#     forestpreds = { "pred_{}".format(i) : est.predict(X) for i, est in enumerate(est)}
#     forestpreds = pd.DataFrame(forestpreds)
#     preds = pd.DataFrame({"pred" : forest.predict(X),
#                           "stds" : forestpreds.std(axis=1)})
#     if use_min:
#         preds['pred'] = forestpreds.min(axis=1)    
#     lmbda = (ParameterSampler({ "lm" : expon()}, n_iter=number_to_sample))
#     lcb = pd.DataFrame({ "lcb_{}".format(i) : preds.pred - li["lm"] * preds.stds for i, li in enumerate(lmbda) })
#     # TODO: include X in lcb, to look up parameters from selected values
#     return lcb
# 
# lcb = LCB(rfr, X, 5)
# lcb.plot()
# =============================================================================
# =============================================================================
# forest = rfr
# est = forest.estimators_
# forestpreds = { "pred_{}".format(i) : est.predict(X) for i, est in enumerate(est)}
# forestpreds = pd.DataFrame(forestpreds)
# preds = pd.DataFrame({"pred" : forest.predict(X),
#                       "stds" : forestpreds.std(axis=1)})
# preds.plot()
# preds.pred.plot()
# =============================================================================
# =============================================================================
# batch_size = [16, 32, 64, 128, 256, 512]
# #activation = ["softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
# dense = [(500, 100, 50),
#          (1000, 500, 100, 50),
#          (2000, 1000, 500, 100, 50),
#          (2000, 1000, 1000, 500, 100, 50),
#          (2000, 1000, 1000, 1000, 500, 100, 50)]
# #optimizer = ["adam", "sgd", "rmsprop", "adagrad", "adadelta","adamax","nadam"]
# conv = [ (50, 50, 50, 50, 50, 1),
#          (25, 25, 25, 25, 25, 1),
#          (64, 32, 16, 32, 64, 1),
#          (100, 100, 100, 100, 100, 1),
#          (32, 20, 16, 32, 10, 1)]
# 
# ps = prs.ParameterSet()
# 
# ps.add(prs.DiscreteParameter("batch_size", batch_size))
# ps.add(prs.IntegerParameter("epochs", 5, 100))
# #ps.add(prs.DiscreteParameter("activation", activation))
# ps.add(prs.DiscreteParameter("dense", dense))
# #ps.add(prs.DiscreteParameter("optimizer", optimizer))
# ps.add(prs.NumericParameter("drop", 0.0, 0.9))
# ps.add(prs.NumericParameter("learning_rate", 0.00001, 0.1))
# ps.add(prs.DiscreteParameter("conv", conv))
# # TODO: since dense and conv will be dummy-coded, ensure that all possible
# # category values are present in the parameter set
# 
# print(ps)
# griddata = {}
# griddata = defaultdict(list)
# for p in ps:
#     for k, v in p.items():
#         griddata[k].append(v)
# =============================================================================
        
from sklearn.model_selection import ParameterGrid, ParameterSampler
batch_size = [16, 32, 64, 128, 256, 512]
#activation = ["softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
# =============================================================================
# dense = [(500, 100, 50),
#          (1000, 500, 100, 50),
#          (2000, 1000, 500, 100, 50),
#          (2000, 1000, 1000, 500, 100, 50),
#          (2000, 1000, 1000, 1000, 500, 100, 50)]
# #optimizer = ["adam", "sgd", "rmsprop", "adagrad", "adadelta","adamax","nadam"]
# conv = [ (50, 50, 50, 50, 50, 1),
#          (25, 25, 25, 25, 25, 1),
#          (64, 32, 16, 32, 64, 1),
#          (100, 100, 100, 100, 100, 1),
#          (32, 20, 16, 32, 10, 1)]
# =============================================================================
dense = range(5)
conv = range(5)
epochs = [5*i for i in range(1,21)]
drop = [i*0.05 for i in range(19)]
learning_rate = [(0.00001 * i + 0.1 * (20 - i)) / 20 for i in range(21)] 

gdict = { "batch_size" : batch_size,
         "dense" : dense,
         "conv" : conv,
         "epochs" : epochs,
         "drop" : drop,
         "learning_rate" : learning_rate }

pg = ParameterGrid(gdict)
        
gdata = defaultdict(list)
for p in pg:
    for k, v in p.items():
        gdata[k].append(v)
        
Xgrid = pd.DataFrame(gdata)
Xgrid.shape
print(Xgrid.describe())

# =============================================================================
# gridpred = pd.DataFrame({"pred" : rfr.predict(Xgrid)})
# print(gridpred.describe())
# gridpred.plot()
# =============================================================================
Xgridd = pd.get_dummies(Xgrid, columns=['dense', 'conv'])
Xgridd.describe()

Xd = pd.get_dummies(X, columns=['dense', 'conv'])
scaler = StandardScaler()
Xds = scaler.fit_transform(Xd)

Xgridds = scaler.transform(Xgridd)

forest =  RandomForestRegressor(oob_score=True)
forest.fit(Xds, y)
#est = forest.estimators_
#forestpreds = { "pred_{}".format(i) : est.predict(Xds) for i, est in enumerate(est)}
#forestpreds = pd.DataFrame(forestpreds)
preds = pd.DataFrame({"pred" : forest.predict(Xds)})  #,
#                      "stds" : forestpreds.std(axis=1)})

importance = [z for z in zip(Xd.columns, forest.feature_importances_)]
importance = sorted(importance, key=lambda x : x[1], reverse=True)
print("Feature Importances")
print("{:20}\t{:5}".format("Parameter:", "Value:"))
for param, imp in importance[:10]:
    print("{:20}\t{:5.3f}".format(param, imp))

preds.plot()
#preds.pred.plot()

rfrpred = pd.DataFrame({"rfr" : forest.predict(Xgridds)})
rfrpred.plot()

k = ConstantKernel(1.0, (0.001, 1000.0)) * RBF(1.0, (0.01, 100.0))
gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, n_restarts_optimizer=20, normalize_y=True)
gpr.fit(Xds, y)
gprpred = pd.DataFrame({"gpr" : gpr.predict(Xgridds)})
gprpred.plot()


gprpred_idxmin = gprpred.idxmin()

gprpred_star = gprpred.iloc[gprpred_idxmin]
Xgrid_star = Xgrid.iloc[gprpred_idxmin]

print("Best GPR predicted value")
print("Index: {}".format(gprpred_idxmin))
print(gprpred_star)
print("Parameter values for best observed:")
print(Xgrid_star)

rfrpred_idxmin = rfrpred.idxmin()

rfrpred_star = rfrpred.iloc[rfrpred_idxmin]
Xgrid_star = Xgrid.iloc[rfrpred_idxmin]

print("Best RFR predicted value")
print("Index: {}".format(rfrpred_idxmin))
print(rfrpred_star)
print("Parameter values for best observed:")
print(Xgrid_star)

gprpred_idxmin = gprpred.idxmin()

gprpred_star = gprpred.iloc[gprpred_idxmin]
Xgrid_star = Xgrid.iloc[gprpred_idxmin]

print("Best GPR predicted value")
print("Index: {}".format(gprpred_idxmin))
print(gprpred_star)
print("Parameter values for best observed:")
print(Xgrid_star)

nadir_gpr = gprpred[gprpred.gpr == float(gprpred.min())]
print("GPR grid points at minimum: {:5d}".format(nadir_gpr.shape))
nadir_rfr = rfrpred[rfrpred.rfr == float(rfrpred.min())]
print("RFR grid points at minimum: {:5d}".format(nadir_rfr.shape))


