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

   
# pick a file with .format(run_id),  or use  .format("*") with glob 
RUN_JSON_PATTERN = "run.{}.json"


# =============================================================================
# edit these for testing on a local machine
# =============================================================================
NT3_OUTPUT_DIR = "/Users/johnbauer/Benchmarks/Pilot1/NT3/save"
NT3_SUBDIRECTORY = 'experiment_0'

P3B1_OUTPUT_DIR = "/Users/johnbauer/Benchmarks/Pilot3/P3B1/save"
P3B1_SUBDIRECTORY = ''

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
                 pdtypes,
                 dummies):
        
        self.json_keys = json_keys
        self.param_keys = param_keys # parameters to send to pandas dataframe
        self.run_file_path = os.path.join(output_dir, subdirectory, run_json_pattern)
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
        
    def add_all(self):
        for run_file in glob.iglob(self.run_file("*")):
            self.add_file(run_file)
        
    def add(self, run_dict):
        """Accumulate a list of values for each parameter"""
        for key in self.param_keys:
            self.data[key].append(run_dict.get(key, None))

    def add_file(self, filename):
        param_dict = JSONLog(filename, self.json_keys).parse_file()
        self.add(param_dict)
            
    def _get_dataframe(self):
        msg = "Categorical variable names must not contain {}".format(RunData.PREFIX_SEP)
        assert all(RunData.PREFIX_SEP not in dname for dname in self.dummies), msg
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
            df = pd.get_dummies(df, columns=dummies, prefix_sep=RunData.PREFIX_SEP) if dummies else df
            names = df.columns
            dummy_names = self.dummy_names
            rename = {}
            original = {}
            index = Counter()
            for rootname in dummies:
                for name in names:
                    if rootname == name[:len(rootname)]:
                        catname = name[len(rootname):]
                        if RunData.PREFIX_SEP in catname:
                            catname = catname.split(RunData.PREFIX_SEP)
                            # n.b. this is impossible given the assertion
                            # TODO: remove one or the other
                            catname = RunData.PREFIX_SEP.join(catname[1:])
                            dummy_names[rootname].append(catname)
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
            #df.rename(columns=rename, inplace=True)
            self._df = df
        return df

# =============================================================================
#     def _get_dataframe_old(self):
#         if self._df:
#             df = self._df
#         else:
#             df = pd.DataFrame.from_dict(self.data)
#             # TODO: discard data dict if unneeded
#             #self.data = None
#             # training_loss and validation_loss are already float64
#             pdtypes = self.pdtypes
#             dummies = self.dummies
#             df = df.astype(pdtypes) if pdtypes else df
#             df = pd.get_dummies(df, columns=dummies, prefix_sep="|") if dummies else df
#             names = df.columns
#             dummy_names = self.dummy_names
#             rename = {}
#             original = {}
#             index = Counter()
#             # TODO: this is not bulletproof, but mistakes are not likely
#             # if parameters have reasonable names not ending in _0, _1, etc.
#             for name in names:
#                 name_ = name.split("|")
#                 rootname = name_[0]
#                 if rootname in dummies:
#                     dummy_names[rootname].append("|".join(name_[1:]))
#                     newname = "{}_{}".format(rootname, index[rootname])
#                     index[rootname] += 1
#                     original[newname] = name
#                     rename[name] = newname
#             self.dummy_names = dummy_names
#             self.original_names = original
#             self.new_names = rename
#             # TODO: less drastic resolution for name conflicts before rename
#             for name in rename.values():
#                 assert name not in names, "Name conflict: {}".format(name)
#             # n.b. original.keys() == rename.values() and vice-versa 
#             #df.rename(columns=rename, inplace=True)
#             self.df = df
#         return df
# =============================================================================
    
    dataframe = property(_get_dataframe, None, None, "Lazy evaluation of dataframe")
    
    # it seems worth thinking about a ParameterDict class that is aware of the issues...
    def original_names(self, param_dict):
        """restore original parameter names
        
        but beware: multi-valued lists are returned for dummy coded variables
        """
        renamed = {}
        #multivalued = defaultdict(list)
        for key, values in self.dummies.items():
            # set up list of correct length to hold indexed values
            renamed[key] = [None] * len(values)
        for key in param_dict:
            tokens = key.split(RunData.PREFIX_SEP)
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

class NT3RunData(RunData):
    """Scrape NT3 data from run.###.json files"""
    # these are the keys present in the JSON log/solr to be retrieved, 
    # except 'parameters', which gets special treatment
    JSON_KEYS = ['run_id', 'training_loss', 'validation_loss', 'runtime_hours']
    # these are the keys actually being retreived as a dataframe
    PARAM_KEYS = ['conv', 'dense', 'epochs', 'batch_size', 'learning_rate',
                  'drop', 'classes', 'pool', 'run_id',
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
                 pdtypes=PARAM_DTYPES,
                 dummies=PARAM_CATEGORICAL):
        """output_dir/subdirectory gives the location of the json log files 
        to be parsed.  
        
        Default values can be overridden, normally just call with directory"""
        super(NT3RunData, self).__init__(
                 output_dir=output_dir, 
                 subdirectory=subdirectory,
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=json_keys,
                 param_keys=param_keys,
                 pdtypes=pdtypes,
                 dummies=dummies)


class P3B1RunData(RunData):
    """Scrape P3B1 data from run.###.json files"""
    # these are the keys present in the JSON log/solr to be retrieved, 
    # except 'parameters', which gets special treatment
    JSON_KEYS = ['run_id', 'training_loss', 'validation_loss', 'runtime_hours']
    # these are the keys actually being retreived as a dataframe
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
                 pdtypes=PARAM_DTYPES,
                 dummies=PARAM_CATEGORICAL):
        """output_dir/subdirectory gives the location of the json log files 
        to be parsed.  
        
        Default values can be overridden, normally just call with directory"""
        super(P3B1RunData, self).__init__(
                 output_dir=output_dir, 
                 subdirectory=subdirectory,
                 run_json_pattern=RUN_JSON_PATTERN,
                 json_keys=json_keys,
                 param_keys=param_keys,
                 pdtypes=pdtypes,
                 dummies=dummies)
        
# TODO: figure out why parameters is not showing up in solr, but is in the log
# for now open the file and extract json here, 
# if parameters start to show up in solr rework this 
# so the json.load happens outside, just pass in the json
class JSONLog:
    """Utility class to mediated between JSON log file and parameter dictionary"""
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
    
    output_dir = NT3_OUTPUT_DIR
    output_subdirectory = NT3_SUBDIRECTORY
    
    #output_dir = os.path.join(NT3RunData.OUTPUT_DIR, output_subdirectory)
    #output_dir = "/Users/johnbauer/Benchmarks/Pilot1/NT3"
    #nt3_data = NT3RunData(output_dir=output_dir, subdirectory=output_subdirectory)
    
    nt3_data = NT3RunData(output_dir=output_dir, subdirectory=output_subdirectory)

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


    output_dir = P3B1_OUTPUT_DIR
    output_subdirectory = P3B1_SUBDIRECTORY
    
    p3b1_data = P3B1RunData(output_dir=output_dir, subdirectory=output_subdirectory)

    p3b1_data.add_all()
    data = p3b1_data.dataframe
    #data = pd.DataFrame(p3b1_data.data)
    
    print(p3b1_data.dummy_names)
    print(p3b1_data.new_names)
    print(p3b1_data.original_names)
    
    print(data.describe())
    print(data.head())
    print(data.columns)
    print(data.dtypes)    
