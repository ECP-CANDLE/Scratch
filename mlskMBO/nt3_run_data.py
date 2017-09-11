#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:49:27 2017

@author: johnbauer
"""

import pandas as pd
import os
import glob
import json
from collections import defaultdict, Counter

# =============================================================================
# Need to incorporate ParameterSet and default dictionary in order to
# construct dictionaries from the values suggested by RFR or GPR...
# also do sampling, need to add Naive Bayes, update probs etc.
# inputs: need to start with a "schema" of sorts:
# read in default value dictionary
# =============================================================================
   
# pick a file with .format(run_id),  or use  .format("*") with glob 
RUN_FILE_PATTERN = "run.{}.json"
OUTPUT_DIR = "/Users/johnbauer/Benchmarks/Pilot1/NT3/save"

# for reference, here is param_dict.keys():
PARAM_DICT_KEYS = ['data_url', 'train_data', 'test_data', 'model_name', 'conv',
                   'dense', 'activation', 'out_act', 'loss', 'optimizer',
                   'metrics', 'epochs', 'batch_size', 'learning_rate',
                   'drop', 'classes', 'pool', 'save', 'run_id',
                   'trainable_params', 'non_trainable_params', 'total_params',
                   'training_loss', 'validation_loss']
# these are the keys actually being retreived
PARAM_KEYS = ['conv', 'dense', 'epochs', 'batch_size', 'learning_rate',
              'drop', 'classes', 'pool', 'run_id',
              'training_loss', 'validation_loss', 'runtime_hours']
# coerce to the desired type
PARAM_DTYPES = {'batch_size' : 'int64', 'classes' : 'int64', 'drop' : 'float64',
                  'epochs' : 'int64', #'run_id' : 'int64',
                  'learning_rate' : 'float64'}
PARAM_CATEGORICAL = ['conv', 'dense'] #, 'pool']
# TODO: conv, dense, pool are string representations of lists of integers, not yet being parsed   
  
class NT3RunData:
    """Scrape NT3 data from run.###.json files"""
    def __init__(self,
                 output_dir=OUTPUT_DIR, 
                 subdirectory="",
                 run_file_pattern=RUN_FILE_PATTERN,
                 param_keys=PARAM_KEYS,
                 pdtypes=PARAM_DTYPES,
                 dummies=PARAM_CATEGORICAL):
        
        self.param_keys = param_keys # parameters to send to pandas dataframe
        self.run_file_path = os.path.join(output_dir, subdirectory, run_file_pattern)
        self.data = defaultdict(list)
        # internal use only, use dataframe property to access
        self._df = None
        self.pdtypes = pdtypes
        self.dummies = dummies
        self.dummy_names = defaultdict(list)
        self.original_names = {}
        self.new_names = {}
        
    def run_file(self, run_id):
        """run_id ="*" to get pattern for all, otherwise a single number"""
        return self.run_file_path.format(run_id)
        
    def add_file(self, filename):
        param_dict = NT3RunJSONLog(filename).parse_file()
        self.add(param_dict)
        
    def add_all(self):
        for run_file in glob.iglob(self.run_file("*")):
            self.add_file(run_file)
        
    def add(self, run_dict):
        """Accumulate a list of values for each parameter"""
        for key in self.param_keys:
            self.data[key].append(run_dict.get(key, None))
            
    def _get_dataframe(self):
        if self._df:
            df = self._df
        else:
            df = pd.DataFrame.from_dict(self.data)
            # TODO: discard data dict if unneeded
            #self.data = None
            # training_loss and validation_loss are already float64
            pdtypes = self.pdtypes
            dummies = self.dummies
            df = df.astype(pdtypes) if pdtypes else df
            df = pd.get_dummies(df, columns=dummies) if dummies else df
            names = df.columns
            dummy_names = self.dummy_names
            rename = {}
            original = {}
            index = Counter()
            # TODO: this is not bulletproof, but mistakes are not likely
            # if parameters have reasonable names not ending in _0, _1, etc.
            for name in names:
                name_ = name.split("_")
                rootname = name_[0]
                if rootname in dummies:
                    dummy_names[rootname].append("_".join(name_[1:]))
                    newname = "{}_{}".format(rootname, index[rootname])
                    index[rootname] += 1
                    original[newname] = name
                    rename[name] = newname
            self.dummy_names = dummy_names
            self.original_names = original
            self.new_names = rename
            # TODO: less drastic resolution for name conflicts before rename
            for name in rename.values():
                assert name not in names, "Name conflict: {}".format(name)
            # n.b. original.keys() == rename.values() and vice-versa 
            df.rename(columns=rename, inplace=True)
            self.df = df
        return df
    
    dataframe = property(_get_dataframe, None, None, "Lazy evaluation of dataframe")
    
    # it seems worth thinking about a ParameterDict class that is aware of the issues...
    def original_names(self, param_dict):
        """restore original parameter names
        
        but beware: multi-valued lists are returned for dummy coded variables
        """
        renamed = {}
        #multivalued = defaultdict(list)
        for key in self.dummies.keys():
            # set up list of correct length to hold indexed values
#            values = []
#            for i in range(len(self.dummies[key])):
#                values.append(None)
            renamed[key] = [None] * len(self.dummies[key])
        for key in param_dict:
            tokens = key.split("_")
            root_name = tokens[0]
            if root_name in self.dummies:
                index = int(tokens[1])
                # working without a net:  not checking bounds
                # by virtue of construction, index must be valid...
                renamed[root_name][index] = param_dict[key]
                # NB use self.dummy_names[root_name][index] to retrieve 
                # original categorical *value* for this index
                # assume dictionary will be post-processed before
                # being handed over to nt3.run(...)
            else:
                renamed[key] = param_dict[key]
        return renamed

# TODO: figure out why parameters is not showing up in solr, but is in the log
class JSONLog:
    """Utility class to mediated between JSON log file and parameter dictionary"""
    def __init__(self, run_file, param_dict={}):
        """run_file: typically run.###.json
           param_dict: default values, typically empty"""
        try:
            with open(run_file, "r") as f:
                run_json = json.load(f)
        except:
            print("Problem decoding {}".format(run_file))
            run_json = []
        
        assert isinstance(run_json, list), "Expecting a list of dictionaries"
        assert all(isinstance(d, dict) for d in run_json), "Expecting only dictionaries"
        
        self.json = run_json
        self.param_dict = param_dict

    def get(self, key, cmd='set'):
        """ Returns the last value found.
        
            cmd: solr command, in ('add', 'set', None) """
        assert cmd in (None, 'add', 'set'), "Unknown solr command"
        
        var = None
        for d in self.json:
            temp = d.get(key, var)
            if isinstance(temp, dict):
                var = temp.get(cmd, var)
            else:
                var = temp
        self.param_dict[key] = var
        return var

    def parse_file(self):
        assert False, "Abstract method, must be overridden in subclass"
        
class NT3RunJSONLog(JSONLog):
    def parse_file(self):        
        params = self.get('parameters')
        print("Parameters: ")
        print(params)
        # proably harmless to leave parameters alone...
        # ... but if parameters contains a parameter named 'parameters' 
        # it would not show up
        del self.param_dict['parameters']
        param_dict = self._parse_parameters(params)
        self.param_dict.update(param_dict)        
        print("Parameter dictionary: ")
        print(param_dict)        
        self.get('run_id')
        self.get('training_loss') #, 'set')
        self.get('validation_loss') #, 'set')
        self.get('runtime_hours') #), 'set')
        print("Full Parameter dictionary: ")
        print(self.param_dict)          
        return self.param_dict
    
    # utility method 
    def _parse_parameters(self, params):
        """JSON 'parameters' holds a list of strings, each a ":"-separated pair"""
        param_dict = {}
        for param in params:
            param = param.split(":")
            key = param[0]
            param = ":".join(param[1:]) # re-join e.g. ['ftp', '//ftp.mcs...']
            param = param.strip()
            # TODO: parse out the lists, such as conv or dense or pool
            param_dict[key] = param
        return param_dict

if __name__ == "__main__":
    
    output_subdirectory = 'experiment_0'
    
    #output_dir = os.path.join(NT3RunData.OUTPUT_DIR, output_subdirectory)
    #output_dir = "/Users/johnbauer/Benchmarks/Pilot1/NT3"
    #nt3_data = NT3RunData(output_dir=output_dir, subdirectory=output_subdirectory)
    
    nt3_data = NT3RunData(subdirectory=output_subdirectory)

    nt3_data.add_all()
    data = nt3_data.dataframe
    #data = pd.DataFrame(nt3_data.data)
    
    print(nt3_data.dummy_names)
    print(nt3_data.new_names)
    print(nt3_data.original_names)
    
    print(data.describe())
    print(data.head())
    print(data.columns)
    print(data.dtypes)
