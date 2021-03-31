
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
from time import time
from alibi.explainers import CounterFactual, CounterFactualProto
#print('TF version: ', tf.__version__)
#print('Eager execution enabled: ', tf.executing_eagerly()) # False
import pickle


print("CF_FAKE: Imports OK.")

def run():
    print("CF_FAKE.run(): OK.")
    return "CF_FAKE RETURN OK"
