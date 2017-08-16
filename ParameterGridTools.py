#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 08:56:53 2017

@author: johnbauer
"""

import random

class GridIterator(object):
    """ Accepts an arbitrary number of iterables, iterates through the
        Cartesian product of their values in lexicographic order.  For example:
       
        for params in GridIterator([0,1],[0,1,2]):
            print (params)
           
        [0, 0]
        [0, 1]
        [0, 2]
        [1, 0]
        [1, 1]
        [1, 2]       
    """
    def __init__(self, *args):
        # assert args is a list of iterables ...
        assert all(len(arg) for arg in args), "Expecting length > 0 for all parameters"
        self.args = args
    def __iter__(self):
        if len(self.args) > 1:
            for param in self.args[0]:
                for param_list in GridIterator(*self.args[1:]):
                    yield [param] + param_list
        else:
            for param in self.args[0]:
                yield [param]
                
    def __getitem__(self, index):
        params = []
        q = index
        for arg in reversed(self.args):
            m = len(arg)
            q, r = divmod(q, m)
            params.append(arg[r])
        return params
    
    def sample_iterator(self, k):
        """ Draw a sample of size k from the grid.  If k exceeds the number
            of points in the grid, returns all the points in random order.
        """
        grid_size = 1
        for arg in self.args:
            grid_size *= len(arg)
        k = min(k, grid_size)
        sample = random.sample(range(grid_size), k)
        for i in sample:
            yield self[i]
        
 
class ParamDictionaryGridIterator(GridIterator):
    """ Iterates through the Cartesian product of values listed for each parameter
        Returns a dictionary of parameter values for each combination. For example:
            
        for param_dict in ParamDictionaryIterator(day='MF', hour=[6,12], time=['am', 'pm']):
            print(param_dict)

        {'day': 'M', 'hour': 6, 'time': 'am'}
        {'day': 'M', 'hour': 6, 'time': 'pm'}
        {'day': 'M', 'hour': 12, 'time': 'am'}
        {'day': 'M', 'hour': 12, 'time': 'pm'}
        {'day': 'F', 'hour': 6, 'time': 'am'}
        {'day': 'F', 'hour': 6, 'time': 'pm'}
        {'day': 'F', 'hour': 12, 'time': 'am'}
        {'day': 'F', 'hour': 12, 'time': 'pm'}    
    """
    def __init__(self, **kwargs):
        """ Each keyword argument should be an iterable.  
            parameters with a fixed value must be given as a list,
            e.g. size=[7] or method=['linear']
        """
        self.keys = list(kwargs.keys())
        super(ParamDictionaryGridIterator, self).__init__(*kwargs.values())
        #self.param_iterator = GridIterator(*kwargs.values())
        
    def __iter__(self):
        for params in super(ParamDictionaryGridIterator, self).__iter__():
            yield { k:p for k, p in zip(self.keys, params) }
            
    def __getitem__(self, index):
        """ Paramater dictionary may be constructed from its sequence number if enumerated
            
            pgi = ParamDictionaryGridIterator(a=[0,1],b=[0,1])
            for i, param_dict in enumerate(pgi):
                print(param_dict, " = ", pgi[i])
                
            {'a': 0, 'b': 0}  =  {'b': 0, 'a': 0}
            {'a': 0, 'b': 1}  =  {'b': 1, 'a': 0}
            {'a': 1, 'b': 0}  =  {'b': 0, 'a': 1}
            {'a': 1, 'b': 1}  =  {'b': 1, 'a': 1}           
        """
        param_dict = {}
        q = index
        for key, arg in zip(reversed(self.keys), reversed(self.args)):
            m = len(arg)
            q, r = divmod(q, m)
            param_dict[key] = arg[r]
        return param_dict            
            
class ParamDictionaryGridEnumerator(ParamDictionaryGridIterator):
    """Generates parameter dictionaries for each point in the Cartesian product,
       sequentially numbered by run_id beginning with start_number"""
    def __init__(self, run_id="run_id", start_number=0, **kwargs):
        self.run_id = run_id
        self.id_number = start_number
        super(ParamDictionaryGridEnumerator, self).__init__(**kwargs)
        
    def _get_id(self, param_dict):
        # Utility method to ensure unique run_id
        param_dict[self.run_id] = self.id_number            
        self.id_number += 1
        return param_dict
    
    def __iter__(self):
        for param_dict in super(ParamDictionaryGridEnumerator,self).__iter__():
            yield self._get_id(param_dict)
            
    def sample_iterator(self, k):
        for param_dict in super(ParamDictionaryGridEnumerator, self).sample_iterator(k):
            yield self._get_id(param_dict)
