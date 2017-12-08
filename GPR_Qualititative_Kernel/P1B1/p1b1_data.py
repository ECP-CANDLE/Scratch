#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 31 15:49:27 2017

@author: johnbauer
"""
#import pandas as pd
import os
import glob
import json
import csv
from collections import defaultdict


# =============================================================================
# edit these directorys for the a local machine
# =============================================================================

#P1B1_SAVE_DIR = "/homes/jain/P1B1_titan/fixed360_good"
P1B1_SAVE_DIR = "/Users/johnbauer/Benchmarks/Pilot1/P1B1/save/fixed360_good"
CSV_FILE = "P1B1_data.csv"


# =============================================================================
# pick a file with .format(run_id),  or use  .format("*") with glob 
RUN_JSON_PATTERN = "run.{}.json"

NT3_SAVE_DIR = "/Users/johnbauer/Benchmarks/Pilot1/NT3/save"
P3B1_SAVE_DIR = "/Users/johnbauer/Benchmarks/Pilot3/P3B1/save"

# =============================================================================
# Create a subclass to read JSON logs from a project, such as NT3 or P1B3
# See subclasses below for examples
# Each subclass simply packages suitable values with which to initialize
# the RunData object.  RunData will then configure a JSON_Log object
# for each file in the target directory/subdirectory
# and append the requested values into a dataframe with one row for each 
# value of run_id
# =============================================================================
class RunData(object):
    """Scrape data from run.###.json files"""
    # used when creating dummy variables for categorical variables
    # should not be found in parameter names
    # if changed
    def __init__(self,
                 csv_file,
                 save_dir, 
                 subdirectory,
                 run_json_pattern,
                 json_keys,
                 param_keys):
        self.csv_file = csv_file
        self.save_dir = save_dir
        self.json_keys = json_keys
        self.param_keys = param_keys # parameters to send to pandas dataframe
        self.run_json_pattern = run_json_pattern
        self.data = defaultdict(list)
        # internal use only, use dataframe property to access

#    def run_file(self, run_id):
#        """run_id ="*" to get pattern for all, otherwise a single number"""
#        return self.run_file_path.format(run_id)
        
    # TODO: optional directory/subdirectory
    # allow calling for multiple subdirectories
    # Workaround: use pd.concat((file1, file2, ...), axis=1)
    # to combine data files
    def add_all(self):
        # TODO: instantiate a DictWriter, grab each dict from add_file,
        # writer to csv
        with open(self.csv_file, "w") as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=self.param_keys + self.json_keys)
            writer.writeheader()
            save_dir_pattern = os.path.join(self.save_dir, self.run_json_pattern).format("*")
            sub_dir_pattern = os.path.join(self.save_dir, "run_{}/output/", self.run_json_pattern).format("*", "*")
            print(save_dir_pattern)
            print(sub_dir_pattern)
            for run_file in glob.iglob(save_dir_pattern):
                print(run_file)
                param_dict = self.add_file(run_file)
                writer.writerow(self.for_csv(param_dict))
            for run_file in glob.iglob(sub_dir_pattern):
                print(run_file)
                param_dict = self.add_file(run_file)
                writer.writerow(self.for_csv(param_dict))
                
    def add(self, run_dict):
        """Accumulate a list of values for each parameter"""
        for key in self.param_keys:
            self.data[key].append(run_dict.get(key, None))
            
    def for_csv(self, run_dict):
        csv_dict = {}
        for key in self.param_keys + self.json_keys:
            csv_dict[key] = run_dict.get(key, None)
        return csv_dict

    def add_file(self, filename):
        param_dict = JSONLog(filename, self.json_keys).parse_file()
        self.add(param_dict)
        return param_dict
    
    
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
                  'training_loss', 'validation_loss', 'runtime_hours']    
    def __init__(self,
                 save_dir=NT3_SAVE_DIR, 
                 subdirectory="",
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=JSON_KEYS,
                 param_keys=PARAM_KEYS):
        """save_dir/subdirectory gives the location of the json log files 
        to be parsed.  
        
        Default values can be overridden, normally just call with directory"""
        super(NT3RunData, self).__init__(
                 save_dir=save_dir, 
                 subdirectory=subdirectory,
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=json_keys,
                 param_keys=param_keys)

# =============================================================================
# Setting up a new subclass
# Open one of the JSON files (e.g. run.0.json)
# Determine which value(s) should be collected (e.g. 'validation_loss')
# These are listed as JSON_KEYS
# Examine the list after 'parameters'
# Each item is a string of the form parameter_name: parameter_value
# Each will be split on ":" and added to the data dictionary
# with key = parameter_name and value = parameter_value
# List these as PARAM_KEYS
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
    
    def __init__(self,
                 csv_file=CSV_FILE,
                 save_dir=P1B1_SAVE_DIR, 
                 subdirectory="",
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=JSON_KEYS,
                 param_keys=PARAM_KEYS):
        """save_dir/subdirectory gives the location of the json log files 
        to be parsed.  
        
        Default values can be overridden, normally just call with directory"""
        super(P1B1RunData, self).__init__(
                 csv_file=csv_file,
                 save_dir=save_dir, 
                 subdirectory=subdirectory,
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=json_keys,
                 param_keys=param_keys)
        
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
    
    def __init__(self,
                 csv_file=CSV_FILE,
                 save_dir=P3B1_SAVE_DIR, 
                 subdirectory="",
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=JSON_KEYS,
                 param_keys=PARAM_KEYS):
        """save_dir/subdirectory gives the location of the json log files 
        to be parsed.  
        
        Default values can be overridden, normally just call with directory"""
        super(P3B1RunData, self).__init__(
                 csv_file=csv_file,
                 save_dir=save_dir, 
                 subdirectory=subdirectory,
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=json_keys,
                 param_keys=param_keys)
        
# TODO: figure out why parameters is not showing up in solr, but is in the log
# for now open the file and extract json here, 
# if parameters start to show up in solr rework this 
# so the json.load happens outside, just pass in the json
class JSONLog:
    """Utility class to mediate between JSON log file and parameter dictionary"""
    def __init__(self, run_file, keys=[], parameters="parameters"):
        """run_file: log file name, typically 'run.#.json'
           keys: to be retrieved from JSON files
           parameters: typically 'parameters', expecting a list of strings
           which receive special treatment, each gets parsed into {key:value}"""
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
        for param in params:
            param = param.split(":")
            key = param[0]
            param = ":".join(param[1:]) # re-join e.g. ['ftp', '//ftp.mcs...']
            param = param.strip()
            self.param_dict[key] = param
        return self.param_dict
    

if __name__ == "__main__":
    
    csv_file=CSV_FILE
    save_dir = P1B1_SAVE_DIR
    
    p1b1_data = P1B1RunData(csv_file=csv_file, save_dir=save_dir)
    p1b1_data.add_all()



