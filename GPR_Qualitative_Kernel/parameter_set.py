#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 14:13:17 2017

@author: johnbauer
"""

# import ParamGrid # when/if iteration is contemplated
from collections import Counter, OrderedDict, defaultdict
import random
from math import floor, ceil
import pandas as pd
from sklearn.model_selection import ParameterGrid, ParameterSampler
from scipy.stats.distributions import expon

# =============================================================================
# Enable  and simplify random drawing suitable for Bayesian optimization
# =============================================================================
class ParameterSet(object):
    def __init__(self):
        self._parameters = OrderedDict()
        
    def __setitem__(self, key, parameter):
        if parameter.key is None:
            parameter.key = key
        assert parameter.key == key, "parameter key must match ParameterSet key"
        #self._parameters[key] = parameter
        assert isinstance(parameter, Parameter), "Please provide a Parameter object"
        self._parameters[parameter.key] = parameter
        
    def __getitem__(self, key):
        return self._parameters[key]
    
    def __delitem__(self, key):
        del self._parameters[key]
    
    def keys(self):
        return self._parameters.keys()
    
    def add(self, parameter):
        assert isinstance(parameter, Parameter), "Please provide a Parameter object"
        self[parameter.key] = parameter
        
    def draw(self):
        point = {}
        for param_obj in self._parameters.values():
            point[param_obj.key] = param_obj.draw()
        return point
    
    def bounds(self, names=[]):
        """List of bounds suitable for use in scipy optimize
        
        Defaults to all parameters.  If a list is given, all should be 
        valid parameter names."""
        
        assert all(name in self._parameters.keys() for name in names), "Invalid parameter name"
        
        names = names if names else self._parameters.keys()
        return [self[name].bound for name in names]
        
    def focus(self, point):
        focussed = ParameterSet()
        for parameter, value in point.items():
            if parameter in self._parameters.keys():
                #focussed.add(self._parameters[parameter].focus(value))
                focussed[parameter] = self[parameter].focus(value)
            # it's probably OK if there are unknown parameters in point
            # TODO: ...but if not, this is the place to deal with it
        for key in self._parameters.keys():
            if key not in focussed._parameters.keys():
                focussed.add(self._parameters[key])
        return focussed
    
    def __str__(self):
        return "\n".join(str(param_obj) for param_obj in self._parameters.values())

    def encode_dummies(self, vector):
        assert False, "Placeholder, will this be needed? What inputs?"
    
    def decode_dummies(self, dummies):
        """Dictionary has names of the form 'key|categoryvalue' 
        
        Each parameter picks out the  corresponding variables
        """
        assert isinstance(dummies, dict), "Expecting a dictionary"
        decoded = {}
        for param in self._parameters.values():
            decoded.update(param.decode_dummies(dummies))
        return decoded
#        decoded = {}
#        for key, value in dummies.items():
#            decoded.update(self[key].decode_dummies())
#        return decoded
    
# =============================================================================
# Abstract class providing behavior modelled on mlrMBO, ... especially focus
# =============================================================================
class Parameter(object):
    def __init__(self, key=None):
        self.key = key
        #assert False, "Abstract superclass, instantiate Discrete or Int or Float..."
        
    def draw(self):
        assert False, "Abstract method; subclasses must override"
        return 0.0
    
    # TODO: make bounds a property with get/set
    # not currently being used
    def bounds(self, lower, upper):
        self.lower = lower
        self.upper = upper
        
    def focus(self, value):
        assert False, "Abstract method; subclasses must override"
        return None
    
    # overridden by DiscreteParameter to undo dummy coding
    # more efficient to do in ParameterSet
    # also IntgerParameter should enforce int(value)
    def decode_dummies(self, dummies):
        """Picks out the variable corresponding to this parameter
        
        Also enforces validation
        """
        assert isinstance(dummies, dict), "Expecting a dictionary"
        value = dummies.get(self.key, None)
        if value is None:
            return {}
        else:
            return {self.key : self.validate(value)}

    def validate(self, value):
        """Override in subclass to provide a valid value"""
        return value
                
class FixedParameter(Parameter):
    def __init__(self, key, value):
        super(FixedParameter, self).__init__(key)
        self.value = value
        
    def draw(self):
        return self.value
    
    def focus(self, value):
        # TODO: should value be required to be the same as self.value?
        # should value be ignored?
        #assert value == self.value, "Inconsistent values for Fixed parameter"
        return FixedParameter(self.key, self.value)
    
    def __str__(self):
        return "FixedParameter[{}], {}".format(self.key, self.value)

    def validate(self, value):
        """Fixed paramter always has the same value"""
        return self.value
                
class DiscreteParameter(Parameter):
    def __init__(self, values, key=None, prefix_sep="|"):
        """Enumerates a list of values
        key:            parameter name
        prefix_sep:     keys which contain prefix_sep are presumed to have
                        been created by pandas get_dummies(), and will be 
                        packed into a single column by decode_dummies()
        """
        assert isinstance(values, (list, tuple)), "Expecting a list of values"
        #assert len(key.split("_")) == 1, "Design limitation: dummy root names cannot include '_'"
        super(DiscreteParameter, self).__init__(key)
        self.values = values
        self.prefix_sep = prefix_sep
#        names = OrderedDict()
#        # n.b. names uses the string representation of each value in the key
#        # anticipate the names which will be created by get_dummies()
#        # to be used by decode_dummies()
#        for val in self.values:
#            names["{}{}{}".format(key, prefix_sep, val)] = val
        self._names = None
        
    def draw(self):
        """select values with uniform probability..."""
        return self.values[random.randrange(len(self.values))]
    
    def focus(self, value):
        # TODO: should value be required to be in values?
        values = self.values[:] # slice copy
        if value in values:
            values.remove(value)
        # remove one class at random
        if values:
            values.pop(random.randrange(len(values)))
        # put back the focus value -- note it moves to end of list
        values.append(value)
        return DiscreteParameter(values, key=self.key)
    
    def __str__(self):
        return "DiscreteParameter[{}], {}".format(self.key, self.values)
    
    # TODO: make this into __getitem__ ???
    def get_categoryvalue(self, name):
        return self[name]   # self._names.get(name, "")
    
    def _set_names(self):
        names = OrderedDict()
        # n.b. names uses the string representation of each value in the key
        # anticipate the names which will be created by get_dummies()
        # to be used by decode_dummies()
        for val in self.values:
            names["{}{}{}".format(self.key, self.prefix_sep, val)] = val
        self._names = names

    def __getitem__(self, name):
        if self._names is None:
            self._set_names()
        return self._names.get(name, "")        
    
    # probably not used anytime soon
    def get_dummies_as_list(self, value):
        dummies = [0.0] * len(self.values)
        if value in self.values:
            dummies[self.values.index(value)] = 1.0
        return dummies
    
    # TODO: get rid of _as_dict, _list variants soon, pretty sure they're dead
    # don't know if this will be needed or used yet...
    def get_dummies_as_dict(self, value):
        dummies = self.get_dummies_as_list(value)
#        # lazy evaluation of names done here
#        if not self._names:  
#            root_name = self.key
#            self._names = ["{}_{}".format(root_name, val) for val in self.values]
        return {name : dval for name, dval in zip(self._names.keys(), dummies)}
    
    # another one probably never to be used...
    def decode_dummies_list(self, dummies):
        """Dictionary with original categorical value from list of values"""
        # assume dummies is list, len(dummies) == len(self.values)
        # equivalent to numpy argmax
        # TODO: consider checking to see if max is 0.0 (not found)
        # perhaps return empty dictionary... update would still make sense
        index = pd.Series(dummies).idxmax()
        return {self.key : self.values[index]}
    
    def decode_dummies(self, dummies):
        """dummies is a dict having names of the form 'key|categoryvalue' 
        
        Typically dummies parameter would have been created by modelling data
        after discrete parameters have been dummy-coded using | as separator
        Picks out the variables corresponding to this parameter
        Accumulates their values
        returns the category having the largest value
        """
        assert isinstance(dummies, dict), "Expecting a dictionary"
        values = {}
        for dname, dvalue in dummies.items():
            #print("Parameter[{}] | decode {}:{}".format(self.key, dname, dvalue))
            if dname == self.key:
                # no dummies were created, e.g. NT3 batch_size
                return { self.key : self.validate(dvalue) }
            root = dname[:len(self.key)]
            if self.key == root:
                values[dname] = dvalue
        if values:
            #print("Parameter[{}] | values {}".format(self.key, values))
            max_item = max(values.items(), key=lambda item: item[1])
            #print("Parameter[{}] | values[{}] = {}".format(self.key, max_item[0], max_item[1]))
            #print("---------[{}] | category value = {}".format(self.key, self.get_categoryvalue(max_item[0])))
            return {self.key : self[max_item[0]]}
        else:
            #return {}
            #print("returning [ {}:{} ]".format(self.key, dummies.get(self.key)))
            return {self.key : dummies.get(self.key, "")}
        
# =============================================================================
# def BooleanParameter(DiscreteParameter):
#     def __init__(self, key):
#         super(BooleanParameter, self).__init__(key, [False, True], "|")         
# =============================================================================
        
# =============================================================================
#     def decode_dummies(self, dummies):
#         """dummies is a dict having names of the form 'key|categoryvalue' 
#         
#         Picks out the variables corresponding to this parameter
#         Accumulates their values
#         returns the category having the largest value
#         """
#         assert isinstance(dummies, dict), "Expecting a dictionary"
#         values = {}
#         for key, value in dummies.items():
#             root = key[:len(self.key)]
#             if root == self.key:
#                 # categoryvalue = key[len(self.key):]
#                 categoryvalue = self._names.get(key, None)
#                 if categoryvalue is not None:
#                     #print("Decode[{}]: {} = {}".format(key, categoryvalue, value))
#                     # ad hoc solution for batch_size, which has lists as values
#                     if isinstance(categoryvalue, list):
#                         # avoid unhashable type error
#                         categoryvalue = tuple(categoryvalue)
#                     values[categoryvalue] = value
#         if values:
#             max_item = max(values.items(), key=lambda item: item[1])
#             return {self.key : max_item[0]}
#         else:
#             return {self.key : dummies.get(self.key, "")}
# =============================================================================
    
# =============================================================================
#     def decode_dummies(self, dummies):
#         """Dictionary has names of the form 'key_categoryvalue' 
#         
#         Picks out the variables corresponding to this parameter
#         Accumulates their values
#         returns the category having the largest value
#         """
#         assert isinstance(dummies, dict), "Expecting a dictionary"
#         values = {}
#         for key, value in dummies.items():
#             tokens = key.split("_")
#             #print("decode[{}]: {}".format(key, tokens))
#             root = tokens[0]
#             if root == self.key:
#                 #cat_value = "_".join(tokens[1:])
#                 categoryvalue = self._names[key]
#                 o
#         max_item = max(values.items(), key=lambda item: item[1])
#         return {self.key : max_item[0]}
# =============================================================================
class NumericListParameter(DiscreteParameter):
    def __init__(self, values, key=None, prefix_sep="|", validation_type=int):
        """Additional validation when using lists of numeric values
        
        Selects nearest value during validation, enforces type
        validation_type:    callable, int or float
        """
        valid_values = [validation_type(val) for val in values]
        super(NumericListParameter, self).__init__(valid_values, key, prefix_sep)
        self.validation_type = validation_type

    def validate(self, value):
        """returns the nearest of the listed values, which have already been validated"""
        #valid_val = self.validation_type(value)
        valid_val = float(value) # more careful handling of rounding
        difference = pd.Series([abs(valid_val - val) for val in self.values])
        nearest = difference.idxmin()
        return self.values[nearest]

    def __str__(self):
        return "NumericListParameter[{}], {}".format(self.key, self.values)
    

class NumericParameter(Parameter):
    def __init__(self, lower, upper, key=None):
        assert lower <= upper, "lower cannot exceed upper"
        super(NumericParameter, self).__init__(key=key)
        self.lower = lower
        self.upper = upper
        self.range = upper - lower
        # self.distribution = "Normal" ...

    def validate(self, value):
        ret_val = float(value)
        ret_val = self.upper if ret_val > self.upper else ret_val
        ret_val = self.lower if ret_val < self.lower else ret_val
        return ret_val
        
    def draw(self):
        """Returns a point sampled uniformly from the interval [lower, upper]"""
        return self.range * random.random() + self.lower
        
    def focus(self, value):
        #assert self.lower <= value <= self.upper, "value is out of bounds"
        #print("- Numeric Parameter: focus value = {}, lower = {}, upper = {}".format(value, self.lower, self.upper))
        value = self.validate(value)
        newrange = abs(self.range) / 4.0
        upper = self.validate(value + newrange)
        lower = self.validate(value - newrange)
        #upper = min(self.upper, value + newrange)
        #lower = max(self.lower, value - newrange)
        #print("+ Numeric Parameter: focus value = {}, lower = {}, upper = {}".format(value, lower, upper))
        return NumericParameter(lower, upper, key=self.key)

    def __str__(self):
        return "NumericParameter[{}], {}, {}".format(self.key, self.lower, self.upper)

    def decode_dummies(self, dummies):
        """Picks out the variable corresponding to this parameter
        """
        assert isinstance(dummies, dict), "Expecting a dictionary"
        value = dummies.get(self.key, None)
        if value is None:
            return {}
        else:
            ret_val = self.validate(value)
            return {self.key : ret_val}
    
class IntegerParameter(Parameter):
    def __init__(self, lower, upper, key=None):
        assert lower <= upper, "lower cannot exceed upper"
        super(IntegerParameter, self).__init__(key)
        self.lower = int(lower)
        self.upper = int(upper)
        self.range = upper - lower
        # self.distribution = "Normal" ...

    def validate(self, value):
        ret_val = int(round(float(value)))
        ret_val = self.upper if ret_val > self.upper else ret_val
        ret_val = self.lower if ret_val < self.lower else ret_val
        return ret_val

    def draw(self):
        """Returns a point sampled uniformly from the interval [lower, upper]"""
        return random.randrange(self.range) + self.lower
        
    def focus(self, value):
        #assert self.lower <= value <= self.upper, "value is out of bounds"
        value = self.validate(value)
        newrange = self.range / 4.0
        upper = min(self.upper, ceil(value + newrange))
        lower = max(self.lower, floor(value - newrange))
        return IntegerParameter(lower, upper, key=self.key)
        
    def __str__(self):
        return "IntegerParameter[{}], {}, {}".format(self.key, self.lower, self.upper)
    
    def decode_dummies(self, dummies):
        """Picks out the variable corresponding to this parameter
        """
        assert isinstance(dummies, dict), "Expecting a dictionary"
        value = dummies.get(self.key, None)
        if value is None:
            return {}
        else:
            ret_val = self.validate(value)
            #print("{} called with value {}, returning value {}".format(self, value, ret_val))
            return {self.key : ret_val}


# =============================================================================
# Handle dummy coding, translate dummies back into categorical, etc.
# Should this really be a subclass of DataFrame?
# THIS IS PROBABLY GOING AWAY
# =============================================================================
class DataFrameWithDummies(object):
    """Creates dummy coded variables in df, and associated Parameter objects
    """
    PREFIX_SEP = "|"

    def __init__(self, df, dummies=[]):
        assert isinstance(df, pd.DataFrame)
        assert all(DataFrameWithDummies.PREFIX_SEP not in name for name in df.columns), "{} is not allowed in parameter names".format(DataFrameWithDummies.PREFIX_SEP)
        
        #self.pdtypes = pdtypes
        self.dummies = dummies
        #df = df.astype(pdtypes) if pdtypes else df
        df = pd.get_dummies(df, columns=dummies, prefix_sep=DataFrameWithDummies.PREFIX_SEP) if dummies else df
        names = df.columns
        dummy_values = defaultdict(list)
        # TODO: use dummy_names instead of dummy_values directly in get_dummies
        # (or rip this out, as it's replicated when the DiscreteParameter 
        # objects are initialized)
        # TODO: test that dummy_names match those created by DiscreteParameter
        # dummy names and the corresponding category value
        # {'param_catval' : catval}
        dummy_names = OrderedDict()

        #rename = {}
        #original = {}
        #index = Counter()
        # TODO: this is not bulletproof, but mistakes are not likely
        # if parameters have reasonable names not ending in _0, _1, etc.
        # TODO: careful, parameter names like batch_size should be compared
        # by matching all the characters name[:len(dummyname}] == dummyname
        for name in names:
            name_ = name.split(DataFrameWithDummies.PREFIX_SEP)
            rootname = name_[0]
            if rootname in dummies:
                categoryvalue = DataFrameWithDummies.PREFIX_SEP.join(name_[1:])
                dummy_values[rootname].append(categoryvalue)
                dummy_names[name] = categoryvalue
                #newname = "{}_{}".format(rootname, index[rootname])
                #index[rootname] += 1
                #original[newname] = name
                #rename[name] = newname
        self._df = None

        self.dummy_values = dummy_values
        self.dummy_names = dummy_names

        #self.original_names = original
        #self.new_names = rename
        # TODO: less drastic resolution for name conflicts before rename
        #for name in rename.values():
        #    assert name not in names, "Name conflict: {}".format(name)
        # n.b. original.keys() == rename.values() and vice-versa 
        #df.rename(columns=rename, inplace=True)
        self._df = df
        self._dummy_parameters = None
        
    def _get_dataframe(self):
        return self._df
    
    dataframe = property(_get_dataframe, None, None, "Dataframe with dummy coded indicators for categorical variables")
        
    # TODO: consider moving more here, like most of what is in init
    # TODO: make this a full-fledged property
    def get_dummies(self):
        """Lazy creation of DiscreteParameter objects corresponding to dummies 
        """
        if not self._dummy_parameters:
            dummy_parameters = ParameterSet()
            for name, values in self.dummy_values.items():
                param = DiscreteParameter(name, values)
                dummy_parameters.add(param)
            self._dummy_parameters = dummy_parameters
        return self._dummy_parameters
    
# =============================================================================
# REFERENCE from nt3_param_set.R
# =============================================================================
nt3_paramset_R = """
# R code for reference:

param.set <- makeParamSet(
  makeDiscreteParam("batch_size", values = c(16, 32, 64, 128, 256, 512)),
  makeIntegerParam("epochs", lower = 5, upper = 500),
  makeDiscreteParam("activation", values = c("softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear")),
  makeDiscreteParam("dense", values = c("500 100 50", "1000 500 100 50", "2000 1000 500 100 50", "2000 1000 1000 500 100 50", "2000 1000 1000 1000 500 100 50")),
  makeDiscreteParam("optimizer", values = c("adam", "sgd", "rmsprop", "adagrad", "adadelta","adamax","nadam")),
  makeNumericParam("drop", lower = 0, upper = 0.9),
  makeNumericParam("learning_rate", lower = 0.00001, upper = 0.1),
  makeDiscreteParam("conv", values = c("50 50 50 50 50 1", "25 25 25 25 25 1", "64 32 16 32 64 1", "100 100 100 100 100 1", "32 20 16 32 10 1"))
  ## DEBUG PARAMETERS: DON'T USE THESE IN PRODUCTION RUN
  ## makeDiscreteParam("conv", values = c("32 20 16 32 10 1"))
)
"""

# =============================================================================
# Creates ParameterSet equivalent to the one used for mlrMBO
# =============================================================================
def NT3_ParameterSet():
    """Utility function to create NT3 Parameter Set corresponding to R version"""
    batch_size = [16, 32, 64, 128, 256, 512]
    activation = ["softmax", "elu", "softplus", "softsign", "relu", "tanh", "sigmoid", "hard_sigmoid", "linear"]
    dense = [[500, 100, 50],
             [1000, 500, 100, 50],
             [2000, 1000, 500, 100, 50],
             [2000, 1000, 1000, 500, 100, 50],
             [2000, 1000, 1000, 1000, 500, 100, 50]]
    optimizer = ["adam", "sgd", "rmsprop", "adagrad", "adadelta","adamax","nadam"]
    conv = [ [50, 50, 50, 50, 50, 1],
             [25, 25, 25, 25, 25, 1],
             [64, 32, 16, 32, 64, 1],
             [100, 100, 100, 100, 100, 1],
             [32, 20, 16, 32, 10, 1]]
    
    ps = ParameterSet()
    
    # switching from Discrete to NumericList to enforce integer type
    #ps.add(DiscreteParameter("batch_size", batch_size))
    ps.add(NumericListParameter("batch_size", batch_size))
    ps.add(IntegerParameter("epochs", 5, 100))
    ps.add(DiscreteParameter("activation", activation))
    ps.add(DiscreteParameter("dense", dense))
    ps.add(DiscreteParameter("optimizer", optimizer))
    ps.add(NumericParameter("drop", 0.0, 0.9))
    ps.add(NumericParameter("learning_rate", 0.00001, 0.1))
    ps.add(DiscreteParameter("conv", conv))
    
    return ps

# =============================================================================
# Test functionality
# =============================================================================
if __name__ == "__main__":

    ps = NT3_ParameterSet()
    print("NT3 Parameter Set:")
    print(ps)
    
    data = defaultdict(list)
    for i in range(5):
        params = ps.draw()
        params['validation_loss'] = 1/(1.0 + i)
        for param, value in params.items():
            if param in ("dense", "conv"):
                data[param].append(str(value))
            else:
                data[param].append(value)
    testdf = pd.DataFrame(data)
    dfwd = DataFrameWithDummies(testdf, dummies=['conv'])
    df = dfwd.dataframe
    for i in range(df.shape[0]):
        datum = dict(df.iloc[i])
        print("{:2d} {}".format(i, ps.decode_dummies(datum)))    
        
    
    testdict = {"batch_size" : 16,
                "conv|[50, 50, 50, 50, 50, 1]" : 0.03,
                "conv|[25, 25, 25, 25, 25, 1]" : 0.70, 
                "conv|[64, 32, 16, 32, 64, 1]" : 0.27,
                "dense|[1000, 500, 100, 50]" : 0.1,
                "dense|[2000, 1000, 500, 100, 50]" : 0.1,
                "dense|[2000, 1000, 1000, 500, 100, 50]" : 0.2,
                "drop" : 0.01}
    
    test_param_dict = ps.decode_dummies(testdict)
    print(test_param_dict)
    
# =============================================================================
#     testdata = {"batch_size" : [16, 16, 32],
#                 "conv" : ["[50, 50, 50, 50, 50, 1]",
#                           "[25, 25, 25, 25, 25, 1]", 
#                           "[64, 32, 16, 32, 64, 1]"],
#                 "drop" : [0.01, 0.02, 0.03]}
#     
#     testdf = pd.DataFrame(testdata)
#     dfwd = DataFrameWithDummies(testdf, dummies=['conv'])
#     data = dfwd.dataframe
#     for i in range(data.shape[0]):
#         datum = dict(data.iloc[i])
#         print("{:2d} {}".format(i, ps.decode_dummies(datum)))
# =============================================================================
    
    
#    ps.decode_dummies(testdf.iloc[0])
# =============================================================================
#     for i in range(5):
#         print(ps.draw())
#         
#     ps1 = ps.focus({ "batch_size" : 32, "learning_rate" : 0.1, "epochs" : 10, "drop" : 0.1})
#     print(ps1)
#     for i in range(10):
#         print(ps1.draw())
#         
#     ps2 = ps1.focus({ "batch_size" : 32, "learning_rate" : 0.1, "epochs" : 10, "drop" : 0.1})
#     print(ps2)
#     for i in range(10):
#         print(ps2.draw())
#         
#     ps2 = ps2.focus({ "batch_size" : 32, "learning_rate" : 0.1, "epochs" : 10, "drop" : 0.1})
#     print(ps2)
#     for i in range(10):
#         print(ps2.draw())
#             
#     for i in range(30):
#         print(ps.draw())
# =============================================================================
        
# =============================================================================
#     testdict = {'foo':[i for i in range(5)], 'a':[1,2,2,1,2], 'b':['[0 1]', '[1 0]', '[1 1]', '[0 1]', '[0 1]']}
#     testdf = pd.DataFrame(testdict)
#     dpsi = DataFrameWithDummies(testdf, dummies=['a', 'b'])
#     dummies = dpsi.get_dummies()
#     print(dummies)
#     
#     dfwd = DataFrameWithDummies(testdf, dummies=['a', 'b'])
#     dummies = dfwd.get_dummies()
#     print(dummies)
#         
#     da = dummies[0]
#     db = dummies[1]
#     da.get_dummies_as_dict('1')
#     da.get_dummies_as_dict('2')
#     da.get_dummies_as_dict('z')
#     da.get_dummies_as_list('z')
#     da.get_dummies_as_list('1')
#     da.get_dummies_as_list(1)
#     da.decode_dummies((0.0, 1.00))
#     da.decode_dummies((0.0, 0.00))
#     da.decode_dummies((1.0, 0.00))
#     db.decode_dummies([1,0,0])
#     db.decode_dummies([0,1,0])
#     db.decode_dummies([0,0.1,0])
#     db.decode_dummies([0,0.1,0.2])

#     db.decode_dummies([0,0.1,0.2,9])
# 
# =============================================================================