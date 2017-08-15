#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 08:56:53 2017

@author: johnbauer
"""

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
            
class ParamDictionaryGridEnumerator(ParamDictionaryGridIterator):
    """Generates parameter dictionaries for each point in the Cartesian product,
       sequentially numbered by run_id beginning with start_number"""
    def __init__(self, run_id="run_id", start_number=0, **kwargs):
        self.run_id = run_id
        self.start_number = start_number
        super(ParamDictionaryGridEnumerator, self).__init__(**kwargs)
    def __iter__(self):
        run_id = self.start_number
        for param_dict in super(ParamDictionaryGridEnumerator,self).__iter__():
            param_dict[self.run_id] = run_id            
            run_id += 1
            yield param_dict
