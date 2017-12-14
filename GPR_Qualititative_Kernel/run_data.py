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
from collections import defaultdict  #, Counter

   
# pick a file with .format(run_id),  or use  .format("*") with glob 
RUN_JSON_PATTERN = "run.{}.json"


# =============================================================================
# edit these for testing on a local machine
# =============================================================================
NT3_OUTPUT_DIR = "/Users/johnbauer/Benchmarks/Pilot1/NT3/save"
NT3_SUBDIRECTORY = 'experiment_0'

P3B1_OUTPUT_DIR = "/Users/johnbauer/Benchmarks/Pilot3/P3B1/save"
P3B1_SUBDIRECTORY = ''

#P1B1_OUTPUT_DIR = "/Users/johnbauer/Benchmarks/Pilot1/P1B1/save"
P1B1_OUTPUT_DIR = "/Users/johnbauer/Documents/GitHub/Scratch/GPR_Qualititative_Kernel/P1B1/save"
P1B1_SUBDIRECTORY = "opt"

# =============================================================================
# Create a subclass to read JSON logs from a project, such as NT3 or P1B3
# See subclasses below for examples
# Each subclass packages suitable default values with which to initialize
# the RunData object, detailing the keys expected in the parameters list
# and JSON keys such as 'validation_loss' to be retrieved from the jSON log. 
# RunData will then configure a JSON_Log object for each file in the 
# target directory/subdirectory and append the requested values into a
# dataframe with one row for each value of run_id.
# =============================================================================
class RunData(object):
    """Scrape data from run.###.json files"""
    # used when creating dummy variables for categorical variables
    # should not be found in parameter names
    # if changed
    PREFIX_SEP = "|"
    def __init__(self,
                 output_dir, 
                 subdirectory,
                 run_json_pattern,
                 json_keys,
                 param_keys,
                 pdtypes):
        
        self._output_dir = output_dir
        self._subdirectory = subdirectory
        self.run_json_pattern = run_json_pattern
        self.json_keys = json_keys
        self.param_keys = param_keys # parameters to send to pandas dataframe
        self._run_file_path = os.path.join(output_dir, subdirectory, run_json_pattern)
        # data are cached until dataframe property requested
        self._data = defaultdict(list)
        # internal use only, use dataframe property to access
        self._df = None
        self.pdtypes = pdtypes
 
    @property
    def output_dir(self):
        return self._output_dir
    
    @output_dir.setter
    def output_dir(self, output_dir):
        self._output_dir = output_dir
        self._run_file_path = os.path.join(output_dir, self.subdir, self.run_json_pattern)        


    @property
    def subdirectory(self):
        return self._subdirectory
    
    @subdirectory.setter
    def subdirectory(self, subdir):
        self._subdirectory = subdir
        self._run_file_path = os.path.join(self.output_dir, subdir, self.run_json_pattern)        
    
    def run_file(self, run_id):
        """run_id ="*" to get pattern for all, otherwise a single number"""
        return self._run_file_path.format(run_id)
        
    def add_dict(self, run_dict):
        """Accumulate a list of values for each parameter"""
        for key in self.param_keys:
            self._data[key].append(run_dict.get(key, None))

    def add_json_log(self, filename):
        json_log = JSONLog(filename, self.json_keys)
        param_dict = json_log.parse_file()
        self.add_dict(param_dict)
        
    # TODO: optional directory/subdirectory
    # allow calling for multiple subdirectories
    # Workaround: use pd.concat((file1, file2, ...), axis=0)
    # to combine data files
    def add_run_id(self, run_id="*"):
        for json_log in glob.iglob(self.run_file(run_id)):
            self.add_json_log(json_log)
        
    def _get_dataframe(self):
        """utility function for lazy evaluation of data set from dict buffer"""
        # retrieve the cached dataframe, if any
        df0 = self._df
        #if df0 is not None: print(df0.describe())
        # flush the buffered data read in from log files
        if self._data:
            df1 =  pd.DataFrame.from_dict(self._data)
            pdtypes = self.pdtypes
            df1 = df1.astype(pdtypes) if pdtypes else df1
            self._data = defaultdict(list)
        else:
            df1 = None
        return self._merge_data(df0, df1)
        
    def _merge_data(self, df0, df1):
        if df0 is not None and df1 is not None:
            df =  pd.concat([df0, df1])
        elif df0 is not None:
            df = df0
        elif df1 is not None:
            df = df1
        else:
            df = None
        self._df = df
        return df

    dataframe = property(_get_dataframe, None, None, "Lazy evaluation of dataframe")

    def to_csv(self, csvfile):
        """Cache data as .csv"""
        df = self.dataframe
        df.to_csv(csvfile, index=False)
        
    def from_csv(self, csvfile):
        """Concatenate cached data to this object's data (if any)"""
        # this will flush any buffered data
        df0 = self.dataframe
        df1 = pd.read_csv(csvfile)
        return self._merge_data(df0, df1)

    def add_dataframe(self, df):
        """Concatenate this object's data with another DataFrame """
        # this will flush any buffered data
        df0 = self.dataframe
        assert isinstance(df, pd.DataFrame), "Expecting a pandas DataFrame"
        return self._merge_data(df0, df)        
        
# =============================================================================
# TODO: finish migrating these into subdirectories
# =============================================================================
class NT3RunData(RunData):
    """Scrape NT3 data from run.###.json files"""
    # these are the keys present in the JSON log/solr to be retrieved, 
    # except 'parameters', which gets special treatment
    JSON_KEYS = ['run_id', 'training_loss', 'validation_loss', 'runtime_hours']
    # these are the keys actually being retreived as a dataframe
    # possible values can be found in the 'parameters' list in JSON logs
    # or solr files; each item is a string parameter_name: parameter_value
    PARAM_KEYS = ['conv', 'dense', 'epochs', 'batch_size', 'learning_rate',
                  'drop', 'classes', 'pool', 'run_id',
                  'activation', 'optimizer',
                  'training_loss', 'validation_loss', 'runtime_hours']
    # coerce to the correct type in the dataframe
    PARAM_DTYPES = {'batch_size' : 'int64',
                    'classes' : 'int64',
                    'drop' : 'float64',
                    'epochs' : 'int64',
                    'learning_rate' : 'float64'
                    }
    # these will be dummy-coded
    PARAM_CATEGORICAL = ['dense', 'conv']
    
    def __init__(self,
                 output_dir=NT3_OUTPUT_DIR, 
                 subdirectory="",
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=JSON_KEYS,
                 param_keys=PARAM_KEYS,
                 pdtypes=PARAM_DTYPES):
        """output_dir/subdirectory gives the location of the json log files 
        to be parsed.  
        
        Default values can be overridden, normally just call with directory"""
        super(NT3RunData, self).__init__(
                 output_dir=output_dir, 
                 subdirectory=subdirectory,
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=json_keys,
                 param_keys=param_keys,
                 pdtypes=pdtypes)

# =============================================================================
# Setting up a new subclass
# Open one of the JSON files (e.g. run.0.json)
# Determine which value(s) should be collected (e.g. 'validation_loss')
# These are listed as JSON_KEYS
# Examine the list after 'parameters'
# Each item is a string of the form parameter_name: parameter_value
# Each will be split on ":" and added to the data dictionary
# with key = parameter_name and value = parameter_value
# =============================================================================

class P1B1RunData(RunData):
    """Scrape P1B1 data from run.###.json files"""
    # these are the keys present in the JSON log/solr to be retrieved, 
    # except 'parameters', which gets special treatment
    JSON_KEYS = ['run_id', 'training_loss', 'validation_loss', 'runtime_hours']
    # these are the keys actually being retreived as a dataframe
    # possible values can be found in the 'parameters' list in JSON logs
    # or solr files; each item is a string parameter_name: parameter_value
    PARAM_KEYS = ['drop',
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
                  'runtime_hours']
    # coerce to the correct type in the dataframe
    PARAM_DTYPES = {'batch_size' : 'int64',
                    'drop' : 'float64',
                    'epochs' : 'int64',
                    'learning_rate' : 'float64'
                    }
    # these will be dummy-coded
    PARAM_CATEGORICAL = ['dense', 'optimizer', 'warmup_lr', 'reduce_lr',
                         'activation', 'model']
    
    def __init__(self,
                 output_dir=P1B1_OUTPUT_DIR, 
                 subdirectory="",
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=JSON_KEYS,
                 param_keys=PARAM_KEYS,
                 pdtypes=PARAM_DTYPES):
        """output_dir/subdirectory gives the location of the json log files 
        to be parsed.  
        
        Default values can be overridden, normally just call with directory"""
        super(P1B1RunData, self).__init__(
                 output_dir=output_dir, 
                 subdirectory=subdirectory,
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=json_keys,
                 param_keys=param_keys,
                 pdtypes=pdtypes)
        
class P3B1RunData(RunData):
    """Scrape P3B1 data from run.###.json files"""
    # these are the keys present in the JSON log/solr to be retrieved, 
    # except 'parameters', which gets special treatment
    JSON_KEYS = ['run_id', 'training_loss', 'validation_loss', 'runtime_hours']
    # these are the keys actually being retreived as a dataframe
    # possible values can be found in the 'parameters' list in JSON logs
    # or solr files; each item is a string parameter_name: parameter_value
    PARAM_KEYS = ['epochs', 'batch_size', 'learning_rate',
                  'dropout', 'shared_nnet_spec', 'ind_nnet_spec',
                  'activation', 'optimizer',
                  'run_id', 'training_loss', 'validation_loss', 'runtime_hours']
    # coerce to the correct type in the dataframe
    PARAM_DTYPES = {'batch_size' : 'int64',
                    'dropout' : 'float64', 
                    'epochs' : 'int64',
                    'learning_rate' : 'float64'}
    # these will be dummy-coded
    PARAM_CATEGORICAL = ['shared_nnet_spec', 'ind_nnet_spec',
                         'activation', 'optimizer']
    
    def __init__(self,
                 output_dir=P3B1_OUTPUT_DIR, 
                 subdirectory="",
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=JSON_KEYS,
                 param_keys=PARAM_KEYS,
                 pdtypes=PARAM_DTYPES):
        """output_dir/subdirectory gives the location of the json log files 
        to be parsed.  
        
        Default values can be overridden, normally just call with directory"""
        super(P3B1RunData, self).__init__(
                 output_dir=output_dir, 
                 subdirectory=subdirectory,
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=json_keys,
                 param_keys=param_keys,
                 pdtypes=pdtypes)
        
# =============================================================================
#  TODO: figure out why parameters is not showing up in solr, but is in the log
#  for now open the file and extract json here, 
#  if parameters start to show up in solr rework this 
#  so the json.load happens outside, just pass in the json
# =============================================================================
class JSONLog:
    """Utility class to mediate between JSON log file and parameter dictionary"""
    def __init__(self, run_file, keys=[], parameters="parameters"):
        """run_file: log file name, typically 'run.#.json'
           keys: to be retrieved from JSON files
           parameters: typically 'parameters', expecting a list of strings
           which receive special treatment, each gets parsed into 
           {key:value} pairs and added to the dictionary"""
        try:
            with open(run_file, "r") as f:
                run_json = json.load(f)
        except:
            print("Problem decoding {}".format(run_file))
            run_json = []
        
        assert isinstance(run_json, list), "Expecting a list of dictionaries"
        assert all(isinstance(d, dict) for d in run_json), "Expecting only dictionaries"
        
        self.json = run_json
        self.param_dict = {}
        self.keys = keys
        self.parameters = parameters

    def get(self, key, cmd='set'):
        """ Returns the last value found in either JSON log file or solr.
        
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
        """Get values for the requested keys, with special handling for 'parameters' """
        if self.parameters:
            params = self.get(self.parameters)
            del self.param_dict[self.parameters]
            self._parse_parameters(params)
        for key in self.keys:
            self.get(key)
        return self.param_dict
        
    # utility method 
    def _parse_parameters(self, params):
        """JSON 'parameters' holds a list of strings, each a ":"-separated pair"""
        if params is None:
            return self.param_dict
        for param in params:
            param = param.split(":")
            key = param[0]
            param = ":".join(param[1:]) # re-join e.g. ['ftp', '//ftp.mcs...']
            param = param.strip()
            # TODO: find a better way to handle the list/tuple issue
            # bonus handling because dense, conv have values which may
            # be (string representations of) lists, but elsewhere these
            # may be converted to tuple... need to be consistent
            # or dummy coding will malfuncion and double-count these
            # TODO: get rid of this if it's really no longer needed
            #param = param.replace("[", "(")
            #param = param.replace("]", ")")
            self.param_dict[key] = param
        return self.param_dict
    

if __name__ == "__main__":
 
    output_dir = P1B1_OUTPUT_DIR
    output_subdirectory = P1B1_SUBDIRECTORY
    
    #output_dir = os.path.join(NT3RunData.OUTPUT_DIR, output_subdirectory)
    #output_dir = "/Users/johnbauer/Benchmarks/Pilot1/NT3"
    #nt3_data = NT3RunData(output_dir=output_dir, subdirectory=output_subdirectory)
    
    p1b1_data = P1B1RunData(output_dir=output_dir, subdirectory=output_subdirectory)

    p1b1_data.add_run_id()
    data = p1b1_data.dataframe
    #data = pd.DataFrame(nt3_data.data)
    
   # print(nt3_data.dummy_names)
    #print(nt3_data.new_names)
    #print(nt3_data.original_names)
    
    print(data.describe())
    print(data.head())
    print(data.columns)
    print(data.dtypes)
    
    output_subdirectory = "lcb"
    
    p1b1_data.subdirectory = output_subdirectory
    p1b1_data.add_run_id()
    data = p1b1_data.dataframe

    
#    p1b1_data_lcb = P1B1RunData(output_dir=output_dir, subdirectory=output_subdirectory)
#
#    p1b1_data_lcb.add_run_id()
#    data_lcb = p1b1_data_lcb.dataframe
#    
#    data = pd.concat([data, data_lcb])
    
    print(data.describe())
    print(data.head())
    print(data.columns)
    print(data.dtypes)
    
# =============================================================================
#     TODO: Alternatively, set .subdirectory, .add_run_id(), .dataframe
# =============================================================================

    output_dir = NT3_OUTPUT_DIR
    output_subdirectory = NT3_SUBDIRECTORY
    
    #output_dir = os.path.join(NT3RunData.OUTPUT_DIR, output_subdirectory)
    #output_dir = "/Users/johnbauer/Benchmarks/Pilot1/NT3"
    #nt3_data = NT3RunData(output_dir=output_dir, subdirectory=output_subdirectory)
    
    nt3_data = NT3RunData(output_dir=output_dir, subdirectory=output_subdirectory)

    nt3_data.add_run_id()
    data = nt3_data.dataframe
    #data = pd.DataFrame(nt3_data.data)
    
   # print(nt3_data.dummy_names)
    #print(nt3_data.new_names)
    #print(nt3_data.original_names)
    
    print(data.describe())
    print(data.head())
    print(data.columns)
    print(data.dtypes)


    output_dir = P3B1_OUTPUT_DIR
    output_subdirectory = P3B1_SUBDIRECTORY
    
    p3b1_data = P3B1RunData(output_dir=output_dir, subdirectory=output_subdirectory)

    p3b1_data.add_run_id()
    data = p3b1_data.dataframe
    #data = pd.DataFrame(p3b1_data.data)
    
    #print(p3b1_data.dummy_names)
    #print(p3b1_data.new_names)
    #print(p3b1_data.original_names)
    
    print(data.describe())
    print(data.head())
    print(data.columns)
    print(data.dtypes)    
