# -*- coding: utf-8 -*-

# Assuming Scratch/Runtime_Anomalies and Benchmarks are cloned to directories
# at thessame level:
import p1b1_paths

# otherwise, omit the above but ensure the following are on the PYTHONPATH:

import p1b1_baseline_keras2 as p1b1k2
import json
import numpy as np

# Load the parameter dictionaries:
param_dicts = json.load(open('suspect.json', 'r'))

# Now submit each parameter dictionary to keras:
    
for param_dict in param_dicts:
    # Change the value of ‘datatype’ in each dictionary to a Python type:
    param_dict['datatype'] = np.float32
    # Create a list of int from json string representation
    param_dict['dense'] = eval(param_dict['dense'])
    # Last-minute changes to parameters could be made here
    # e.g. set save location:
    #param_dict['save'] = 'save/directory'
    #param_dict['logfile'] = 'logfile.log'
    # If uaing solr, set 'solr_root' here
    #etc., etc.
    p1b1k2.run(param_dict)
    
# json log will be written to the location specified in 'save', i.e.
# param_dict['save'] = 'save/anomalous'
    
# Then examine the final validation_loss, runtime_hours, etc. in each
# json log file.  The parameters will be near the beginning.