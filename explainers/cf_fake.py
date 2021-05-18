
# CF FAKE PY
# Test that loading TF, etc., work as intended

print("CF_FAKE: Starting imports ...")

import tensorflow as tf
tf.get_logger().setLevel(40) # suppress deprecation messages
tf.compat.v1.disable_v2_behavior() # disable TF2 behaviour as alibi code still relies on TF1 constructs
from tensorflow.keras.models import Model, load_model
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from time import time
from alibi.explainers import CounterFactual, CounterFactualProto
#print('TF version: ', tf.__version__)
#print('Eager execution enabled: ', tf.executing_eagerly()) # False
import pickle

print("CF_FAKE: Imports OK.")

def run(i):
    msg = "CF_FAKE.run(%s): OK." % i
    print("python: " + msg)
    #sys.stdout.flush()



    model_nt3 = tf.keras.models.load_model('/gpfs/alpine/med106/scratch/jain/Scratch/explainers/nt3.autosave.model')
    with open('/gpfs/alpine/med106/scratch/jain/Scratch/explainers/nt3.autosave.data.pkl', 'rb') as pickle_file:
            X_train,Y_train,X_test,Y_test = pickle.load(pickle_file)

    print ("opened files")
    shape_cf = (1,) + X_train.shape[1:]
    print(shape_cf)
    target_proba = 0.9
    tol = 0.1 # want counterfactuals with p(class)>0.90
    target_class = 'other' # any class other than will do
    max_iter = 1000
    lam_init = 1e-1
    max_lam_steps = 20
    learning_rate_init = 0.1
    feature_range = (0,1)


    cf = CounterFactual(model_nt3, shape=shape_cf, target_proba=target_proba, tol=tol,
                        target_class=target_class, max_iter=max_iter, lam_init=lam_init,
                        max_lam_steps=max_lam_steps, learning_rate_init=learning_rate_init,
                        feature_range=feature_range)
        



    shape = X_train[0].shape[0]
    results=[]
    X = np.concatenate([X_train,X_test])
    x_sample=X[i:i+1]
    print(x_sample.shape)
    start = time()
    explanation = cf.explain(x_sample)
    #print('Counterfactual prediction: {}, {}'.format(explanation.cf['class'], explanation.cf['proba']))
    #print("Actual prediction: {}".format(model_nt3.predict(x_sample)))
    results.append([explanation.cf['X'],explanation.cf['class'], explanation.cf['proba']])
    print("saving i=", i)
    filename = "save.p" + str(i)
    pickle.dump(results, open(filename, "wb"))
    results=[]


   # print (results) 
    #filename = "save.p" + i
    #pickle.dump(results, open(filename, "wb")    
    #return msg result
    #return "return: " + msg
