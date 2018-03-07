# -*- coding: utf-8 -*-

from collections import defaultdict

from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel
#from kernels import RBF, ConstantKernel
import kernels as kr
import qualitative_kernels as qk
import parameter_set as prs
from sklearn.model_selection import ParameterSampler
from sklearn.model_selection import train_test_split
from sklearn.cluster import AffinityPropagation

import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats.distributions import expon

import logging
logging.basicConfig(filename='GPR_Model.log',level=logging.DEBUG)

class Factor(object):
    def __init__(self, name, columns, values):
        self.name = name
        self.columns = columns
        self.values = values

    @property
    def kernel_(self, gpr_model):
        pass
    
class GPR_Model(object):
    """Given a dataframe, construct views for X, Y and dummy-coded factors.
    
    Currently, constructs a kernel which is the product of:
        a ConstantKernel
        an RBFKernel over the set of all continuous variables
        for each factor:
            create a set of dummy-coded variables
            put an ExchangeableCorrelation, MultiplicativeCorrelation,
            or UnrestrictiveCorrelation on the dummy-coded variates
            
    The .fit() method runs a Gaussian Process Regression on the data
    The resulting model may be numerically optimized
    Each point in the data is used to initialize the optimizer, and the
    resulting set of local minima are identified by Affinity Propagation
    clustering.
    Implementations of the Lower Confidence Bound and FocusSearch 
    algorithms as described in mlrMBO may be used to generate candidate
    parameter dictionaries.
    
    After calling fit(), the Gaussian Process Regression object may be 
    accessed as .gpr_  while .gpr_.kernel_ holds the kernel after fiiting.
    The factor_kernels_ property provides a dictionary interface by which
    an individual factor kernel may be accessed by name.
    """
    def __init__(self, data_df, X_columns, target, factors=[], 
                 prefix_sep="|"):
        
        dfc_set = set(data_df.columns)
        xcol_set = set(X_columns)
        factor_set = set(factors)
        
        assert target in dfc_set, "Target column must be in dataframe"
        assert xcol_set.issubset(dfc_set), "X_columns must be in dataframe's columns"
        assert factor_set.issubset(dfc_set), "Factors must be in dataframe's columns"
        #assert set(factors).issubset(set(X_columns)), "Factors should be listed in X_columns"
        
        self.data = data_df
        self.factors = factors
        
        xcol_set = xcol_set | factor_set    # set union
        xcol_set.discard(target)
        
        # n.b. set is not a hashable type so make it a list
        X = data_df[list(xcol_set)]
        y = data_df[target]
        
        # Create auxiliary dataframe with dummy-coded indicators 
        Xd = pd.get_dummies(X,
                            columns=factors,
                            prefix_sep=prefix_sep) if factors else X
            
        continuous_columns = []
        factor_columns = defaultdict(list)
        factor_values = defaultdict(list)
                
        # partition the column numbers of the dummy-coded data
        # keep track of untransformed i.e. 'continuous' columns
        # and columns belonging to each factor
        for i, name in enumerate(Xd.columns):
            n = name.split(prefix_sep)
            n0 = n[0]
            if n0 in factors:
                factor_columns[n0].append(i)
                factor_values[n0].append(prefix_sep.join(n[1:]))
            else:
                continuous_columns.append(i)
 
        factor_objects = {}
               
        for name, values in factor_values.items():
            #ps_factor.add(prs.DiscreteParameter(name, values))
            columns = factor_columns[name]
            # factor_columns is being obsoleted by factor_objects
            factor_objects[name] = Factor(name, columns, values)
    
        #self.n_continuous = len(continuous_columns)
        self.continuous_columns = continuous_columns
        self.factor_objects = factor_objects
        self.X = X
        self.Xd = Xd
        self.y = y
        
        # following sklearn convention, items available after .fit() 
        # end with an underscore _
        
        # cache gpr after fit
        self._gpr_ = None
        #self.gpr_ec = None
        #self.gpr_mc = None
        #self.gpr_uc = None
        
        # these are set when gpr_ is set
        
        self._factor_keys = {}
        self._factor_kernels_ = {}

    @staticmethod
    def find_by_name(params):
        """Finds keys which have 'name' parameters, and associated name
        
        Currently only Projection kernels have names
        Nested kernel can be obtained from parameter "{}__kernel".format(key)
        where key can be found by splitting the prequel to name
        params should be obtained from gpr.kernel_.get_params()
        """
        found = {}
        for k, v in params.items():
            split = k.split("__")
            last = split[-1] if len(split) > 1 else ""
            if last == 'name':
                # v will be the value of the "name" parameter
                found["__".join(split[:-1])] = v
            #print(k, split, v)
        return found
        
    @property
    def gpr_(self):
        return self._gpr_
    
    @gpr_.setter
    def gpr_(self, gpr):
        self._gpr_ = gpr
        self._factor_keys = {}
        self._factor_kernels_ = {}
    
    @property
    def factor_keys(self):
        if not self._factor_keys and self.gpr_ is not None:
            try:
                params = self.gpr_.kernel_.get_params()
                self._factor_keys = self.find_by_name(params)
            except:
                self._factor_keys = {}
        return self._factor_keys
            
    # actually since the "continuous" kernel is a Projection kernel
    # it will be found here too
    @property
    def factor_kernels_(self):
        if not self._factor_kernels_:
            if self.gpr_ is not None:
                params = self.gpr_.kernel_.get_params()
            else:
                params = {}
            for key, name in self.factor_keys.items():
                try:
                    #print("Factor Kernel[{}]: key = {}".format(name, key))
                    self._factor_kernels_[name] = params["{}__kernel".format(key)]
                except:
                    print("Couldn't find key = {}".format(key))
                    print("Params:\n", params)
        return self._factor_kernels_


    def _fit(self, model, theta, alpha, n_restarts_optimizer):
        
        # select the indicated callable
        select = model[0:].lower()
        if select == "u":
            model = qk.UnrestrictiveCorrelation
        elif select == "m":
            model = qk.MultiplicativeCorrelation
        else:
            model = qk.ExchangeableCorrelation
            
        kernels = [kr.ConstantKernel(1.0, (0.001, 1000.0))]
        
        # TODO: consider initializing with each variable's standard deviation
        kernel = qk.Projection(self.continuous_columns, name="continuous")
        kernels.append(kernel)

        for factor, columns in self.factor_columns.items():
            kernel = model(len(columns), zeta=theta)
            kernel = qk.Projection(columns, name=factor, kernel=kernel)
            kernels.append(kernel)
        
        kernel = qk.Tensor(kernels)
        
    # for now just use UC as tha model
    def fit_in_progress(self, theta=0.5, alpha=0.01, n_restarts_optimizer=20):
        
        # First model all qualitative factors with an Exchangeable Correlation
        
        # Overall multiplicative scale factor
        kernels = [kr.ConstantKernel(1.0, (0.001, 1000.0))]
        
        # Continuous variables
        # Projection kernel defaults to RBF kernel with s.d. 1.0
        kernel = qk.Projection(self.continuous_columns, name="continuous")
        
        kernels.append(kernel)

        for factor in self.factor_objects.values():
            columns = factor.columns
            kernel = qk.ExchangeableCorrelation(len(columns), zeta=theta)
            kernel = qk.Projection(columns, name=factor.name, kernel=kernel)
            kernels.append(kernel)
            
            # adding a WhiteKernel for noise variance is redundant
            # (its hyperparameter will be driven to 0)
            # use parameter alpha instead
        
        ec_kernel = qk.Tensor(kernels)      
        
        print("Exchangeable Correlation Kernel for Gaussian Process Regression")
        print(ec_kernel)
        logging.debug("Exchangeable Correlation Kernel for Gaussian Process Regression\n{}".format(ec_kernel))
        gpr = GaussianProcessRegressor(kernel=ec_kernel,
                                       alpha=alpha,
                                       normalize_y=True,
                                       n_restarts_optimizer=n_restarts_optimizer)
        #gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, normalize_y=True)
        self.gpr_ = gpr        
        gpr.fit(self.Xd, self.y)

        logging.debug("Fit Gaussian Process Regression:\n{}".format(gpr.kernel_.get_params())) 
        #logging.info("Exchangeable Correlation Kernel for Gaussian Process Regression\n{}".format(report))
                
        # Inspect the fitted kernels' correlations
        for name, k_ in self.factor_kernels_.items():
            print("Correlation for Factor[{}]:\n{}".format(name, k_.correlation))
            
        # =====================================================================
        # Now use the Exchangeable Correlation results to initialize
        # Muliplicative Correlations
        # =====================================================================

        kernels = [kr.ConstantKernel(1.0, (0.001, 1000.0))]
        try:
            theta = self.factor_kernels_["continuous"].theta
            print("Initializing length-scale for Multiplicative")
            print(theta)
        except:
            print("Couldn't find continuous kernel for Multiplicative")
            print(self.factor_kernels_)
            print("*"*80)
            theta = 1.0
        kernel = qk.Projection(self.continuous_columns, name="continuous",
                               kernel=kr.RBF(theta))
        
        for factor in self.factor_objects.values():
            columns = factor.columns
            try:
                eck_ = self.factor_kernels[factor.name]
                theta = eck_.multiplicative_correlation()
                print("Initialize {} for Multiplicative:".format(factor.name))
                print(theta)
            except:
                theta = 0.5
                print("Couldn't initialize {} for Multiplicative:".format(factor.name))
            kernel = qk.MultiplicativeCorrelation(len(columns), zeta=theta)
            kernel = qk.Projection(columns, name=factor.name, kernel=kernel)
            kernels.append(kernel)
                        
        mc_kernel = qk.Tensor(kernels)  
        
        print("Multiplicative Correlation Kernel for Gaussian Process Regression")
        print(mc_kernel)
        logging.debug("Multiplicative Correlation Kernel for Gaussian Process Regression\n{}".format(mc_kernel))
        gpr = GaussianProcessRegressor(kernel=mc_kernel,
                                       alpha=alpha,
                                       normalize_y=True,
                                       n_restarts_optimizer=n_restarts_optimizer)
        #gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, normalize_y=True)
        self.gpr_ = gpr        
        gpr.fit(self.Xd, self.y)  
        
# =============================================================================
# TODO: Unrestrictive FTW!
# =============================================================================
     # for now just use UC as tha model
    def fit(self, theta=0.5, alpha=0.01, n_restarts_optimizer=20):
        
        # First model all qualitative factors with an Exchangeable Correlation
        
        # Overall multiplicative scale factor
        kernels = [kr.ConstantKernel(1.0, (0.001, 1000.0))]
        
        # Continuous variables
        # Projection kernel defaults to RBF kernel with s.d. 1.0
        kernel = qk.Projection(self.continuous_columns, name="continuous")
        
        kernels.append(kernel)

        for factor in self.factor_objects.values():
            columns = factor.columns
            dim = len(columns)
            m = dim * (dim - 1) // 2
            theta = np.random.random_sample(m)
            uc = qk.UnrestrictiveCorrelation(dim, zeta=theta)
            kernel = qk.Projection(columns, name=factor.name, kernel=uc)
            kernels.append(kernel)
            
            # adding a WhiteKernel for noise variance is redundant
            # (its hyperparameter will be driven to 0)
            # use parameter alpha instead
        
        uc_kernel = qk.Tensor(kernels)      
        
        print("Unrestrictive Correlation Kernel for Gaussian Process Regression")
        print(uc_kernel)
        logging.debug("Unrestrictive Correlation Kernel for Gaussian Process Regression\n{}".format(uc_kernel))
        gpr = GaussianProcessRegressor(kernel=uc_kernel,
                                       alpha=alpha,
                                       normalize_y=True,
                                       n_restarts_optimizer=n_restarts_optimizer)
        #gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, normalize_y=True)
        self.gpr_ = gpr        
        gpr.fit(self.Xd, self.y)

        logging.debug("Fit Gaussian Process Regression:\n{}".format(gpr.kernel_.get_params())) 
        #logging.info("Exchangeable Correlation Kernel for Gaussian Process Regression\n{}".format(report))
                
        # Inspect the fitted kernels' correlations
        for name, k_ in self.factor_kernels_.items():
            try:
                print("Correlation for Factor[{}]:\n{}".format(name, k_.correlation))
            except:
                print("Factor[{}]:\n{}".format(name, k_))
            
               
    def dummy_data_to_dict(self, datum):
        columns = self.Xd.columns
        return {col : val for col, val in zip(columns, datum)}

    def decode_dummies(self, X, param_set):
        #columns = self.Xd.columns
        decoded = []
        for i in range(X.shape[0]):
            x = X.iloc[i]
            #d = {col : val for col, val in zip(columns, x)}
            #d = self.dummy_data_to_dict(x)
            d = dict(x)
            params = param_set.decode_dummies(d)
            decoded.append(params)
        return decoded

    def predict_penalized(self, gpr, gamma=1.0, delta=1.0):
        """Penalties are added to each factor to encourage feasible solutions.
        
        Distance from the unit sphere (2-norm = 1) and simplex (1-norm=1)
        are added as the penalty.  When both are near 0 the solution will have
        exactly one coordinate close to 1, all others close to 0.
        
        """
        def factor_penalty(X, columns):
            W = X[columns]
            return gamma*(np.linalg.norm(W, ord=2) - 1.0)**2 + \
                delta*(np.linalg.norm(W, ord=1) - 1)**2
        
        #TODO: see if this still works, was: X.reshape(1,-1)
        # n.b. gpr.predict accepts matrix input and returns an array
        # e.g. if X is n * p return an array with shape (p,)        
        return lambda X : gpr.predict(X.reshape(1,-1))[0] + \
            sum(factor_penalty(X, factor.columns) \
                for factor in self.factor_objects.values())
#               for columns in self.factor_columns.values())
            
        
    def optimize(self, gamma=1.0, delta=1.0, gpr=None, Xd=None):
        """gpr should be self.gpr_uc, _mc, or _ec..."""
        if gpr is None:
            gpr = self.gpr_ #self._get_gpr(gpr)
        if Xd is None:
            Xd = self.Xd
            
        columns = Xd.columns
            
        lower_bounds = Xd.min(axis=0)
        upper_bounds = Xd.max(axis=0)
        
        bounds = [(lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)]

        result_data = defaultdict(list)
        for i in range(Xd.shape[0]):
            #start_val = Xd.iloc[i].as_matrix().reshape(1,-1)
            #start_val = np.atleast_2d(Xd.iloc[i]).reshape(1,-1)
            start_val = Xd.iloc[i]
            # Fit the GPR model
            predict = lambda X : gpr.predict(X.reshape(1,-1))[0]
            result = sp.optimize.minimize(predict, start_val, method='L-BFGS-B', bounds=bounds)
            # the result will be a mixture of factor values
            # Now penalize points which are not feasible
            for epsilon in [1e-3, 1e-2, 5e-2, 1e-1]:
                predict = self.predict_penalized(gpr, epsilon, epsilon)
            result = sp.optimize.minimize(predict, result.x, method='L-BFGS-B', bounds=bounds)
            rx = result.x
            pred = gpr.predict(rx.reshape(1,-1))
            for col, val in zip(columns, rx):
                result_data[col].append(val)
            # pred is an ndarray with shape (1,) so unpack it
            result_data['gpr_optimum'].append(pred[0])
        for k , v in result_data.items():
            logging.debug("{} {}".format(k, len(v)))
        # the dictionary will need to be decoded by a ParameterSet object
        result_data = pd.DataFrame(result_data)
        return result_data

    def optimize_recommend(self, param_set,
                           max_recommend=3,
                           gamma=1.0, delta=1.0,
                           gpr=None, Xd=None,
                           return_data=False):
        """Optimizes GPR model, using each data point as initial value
        
        Clusters the result using Affinity Propagation, and returns
        the cluster representatives, choosing the number of clusters
        automatically. The results are decoded into parameter sets."""

        x = self.optimize(gamma=gamma, delta=delta, gpr=gpr, Xd=Xd)
        aff = AffinityPropagation()
        aff.fit(x)
        #x_rec = pd.DataFrame(aff.cluster_centers_, columns=x.columns)
        # select the lowest validation loss from each cluster
        x_rec = pd.concat([x, pd.DataFrame({'cluster_id' : aff.labels_})], axis=1)
        x_rec.sort_values(by=['cluster_id', 'gpr_optimum'], inplace=True)
        x_rec = x_rec.groupby('cluster_id').first()
        x_rec.sort_values(by=['gpr_optimum'], inplace=True)
        
        if max_recommend < 1:
            max_recommend = x.shape[0]
        x_rec = x.iloc[:max_recommend]
        #x_rec.index = range(len(x_rec))
        #x_rec = x_rec.drop(['gpr_optimum'], axis=1)
        paramdictlist = self.decode_dummies(x_rec, param_set)
        if return_data:
            return paramdictlist, x_rec
        else:
            return paramdictlist
    
    # TODO: this is probably obsolete since clustering should be superior
    def optimize_recommend_sorted(self, param_set, max_recommend=0,
                                  gamma=1.0, delta=1.0,
                                  gpr=None, Xd=None,
                                  return_data=False):
        """Optimizes GPR model, using each data point as initial value
        
        Returns one recommendation for each point, 
        however, these may all be the same if they all converge to the
        global minimum.  The results are decoded into parameter sets."""

        x = self.optimize(gamma=gamma, delta=delta, gpr=gpr, Xd=Xd)
        x.sort_values(by='gpr_optimum', inplace=True)
        if max_recommend < 1:
            max_recommend = x.shape[0]
        x_rec = x.iloc[:max_recommend]
        paramdictlist = self.decode_dummies(x_rec, param_set)
        if return_data:
            return paramdictlist, x_rec
        else:
            return paramdictlist

    def _get_gpr(self, gpr=None):
        """Return the best GPR model which has been fit so far
        
        EC < MC < UC"""
        if gpr is None:
            gpr = self.gpr_uc
        if gpr is None:
            gpr = self.gpr_mc
        if gpr is None:
            gpr = self.gpr_ec
        return gpr
            
    def LCB(self, n_sample, gpr=None, Xd=None):
        gpr = self.gpr_ #self._get_gpr(gpr)
        if Xd is None:
            Xd = self.Xd
        preds = gpr.predict(Xd, return_std=True)
        preds = pd.DataFrame({"prediction" : preds[0], "std_dev" : preds[1]})
        # n.b. lambda is a keyword so change vector of values to alpha
        alpha = ParameterSampler({ "alpha" : expon()}, n_iter=n_sample)
        lcb = pd.DataFrame({"lcb_{}".format(i) : \
                            preds.prediction - \
                            (li["alpha"] * preds.std_dev) \
                            for i, li in enumerate(alpha)})
        # TODO: include X in lcb, to look up parameters from selected values
        return lcb

    def LCB_recommend(self, param_set, max_recommend=0, n_sample=10,
                      gpr=None, Xd=None,
                      return_data=False):
        """Lower Confidence Bound recommendations for GPR model"""
        gpr = self.gpr_  #self._get_gpr(gpr)
        if Xd is None:
            Xd = self.Xd
        if max_recommend < 1:
            max_recommend = Xd.shape[0]
        lcb = self.LCB(n_sample=n_sample, gpr=gpr, Xd=None)
        lcb['minimum'] = lcb.min(axis=1)
        lcb.sort_values(by='minimum', inplace=True)
        Xdmin = Xd.iloc[lcb.index[:max_recommend]]
        recommend = self.decode_dummies(Xdmin, param_set)
#        recommend = []
#        for i in range(Xdmin.shape[0]):
#            x = Xdmin.iloc[i]
#            recommend.append(self.decode_dummies(x, param_set))
        if return_data:
            return recommend, Xdmin
        else:
            return recommend

    def name_report(self, gpr):
        """Report correlations for factors.
        
        Assumes the factor kernel can be found by searching Projection kernels
        by name."""
        report = []
        factors = self.factors
        params = gpr.kernel_.get_params()
        found = find_by_name(params)
        report.append("*"*50)
        for key, name in found.items():
            if name in factors:  
                try:
                    report.append(name)
                    report.append(str(params["{}__kernel".format(key)].correlation))
                    report.append("*"*50)
                except:
                    pass
        return "\n".join(report)




# =============================================================================
# SLUDGE PILE stop doing things this way!
# =============================================================================



class GPR_Model_old(object):
    """Given a dataframe, construct views for X, Y and dummy-coded factors.
    
    Currently, constructs a kernel which is the product of:
        a ConstantKernel
        an RBFKernel over the set of all continuous variables
        for each factor:
            create a set of dummy-coded variables
            put an ExchangeableCorrelation, MultiplicativeCorrelation,
            or UnrestrictiveCorrelation on the dummy-coded variates
            
        The .fit() method runs a Gaussian Process Regression on the data
        The resulting model may be numerically optimized
        Each point in the data is used to initialize the optimizer, and the
        resulting set of local minima are identified by Affinity Propagation
        clustering.
        Implementations of the Lower Confidence Bound and FocusSearch 
        algorithms as described in mlrMBO may be used to generate candidate
        parameter dictionaries.
    """
    def __init__(self, data_df, X_columns, target, factors=[], 
                 prefix_sep="|"):
        
        dfc_set = set(data_df.columns)
        xcol_set = set(X_columns)
        factor_set = set(factors)
        
        assert target in dfc_set, "Target column must be in dataframe"
        assert xcol_set.issubset(dfc_set), "X_columns must be in dataframe's columns"
        assert factor_set.issubset(dfc_set), "Factors must be in dataframe's columns"
        #assert set(factors).issubset(set(X_columns)), "Factors should be listed in X_columns"
        
        self.data = data_df
        self.factors = factors
        
        xcol_set = xcol_set | factor_set    # set union
        xcol_set.discard(target)
        
        # n.b. set is not a hashable type so make it a list
        X = data_df[list(xcol_set)]
        y = data_df[target]
        
        # Create auxiliary dataframe with dummy-coded indicators 
        Xd = pd.get_dummies(X,
                            columns=factors,
                            prefix_sep=prefix_sep) if factors else X
            
        continuous_columns = []
        factor_columns = defaultdict(list)
        factor_values = defaultdict(list)
        
        factor_objects = {}
        
        for i, name in enumerate(Xd.columns):
            n = name.split(prefix_sep)
            n0 = n[0]
            if n0 in factors:
                factor_columns[n0].append(i)
                factor_values[n0].append(prefix_sep.join(n[1:]))
            else:
                continuous_columns.append(i)
                
        # TODO: create a new parameter set, just for the factors
        ps_factor = prs.ParameterSet()
        for name, values in factor_values.items():
            #ps_factor.add(prs.DiscreteParameter(name, values))
            ps_factor[name] = prs.DiscreteParameter(values)
            columns = factor_columns[name]
            factor_objects[name] = Factor(name, columns, values)
    
        #self.n_continuous = len(continuous_columns)
        self.continuous_columns = continuous_columns
        self.factor_columns = factor_columns
        self.parameter_set = ps_factor
        self.X = X
        self.Xd = Xd
        self.y = y
        
        # TODO: consider leaving these till later, and using hasattr to check
        self.gpr_ec = None
        self.gpr_mc = None
        self.gpr_uc = None
        
    def _fit(self, theta, alpha, n_restarts_optimizer):
       
        kernels = [kr.ConstantKernel(1.0, (0.001, 1000.0))]
        
        # TODO: consider initializing with each variable's standard deviation
        kernel = qk.Projection(self.continuous_columns, name="continuous")
        kernels.append(kernel)
        # TODO: move this elsewhere!
    
        #model = "EC"
        model = qk.ExchangeableCorrelation

        for factor, columns in self.factor_columns.items():
            kernel = model(len(columns), zeta=theta)
            kernel = qk.Projection(columns, name=factor, kernel=kernel)
            kernels.append(kernel)
        
        kernel = qk.Tensor(kernels)
        
       
        
    def fit_EC(self, theta=0.1, alpha=0.01, n_restarts_optimizer=20):
       
        kernels = [kr.ConstantKernel(1.0, (0.001, 1000.0))]
        
        # TODO: consider initializing with each variable's standard deviation
        kernel = qk.Projection(self.continuous_columns, name="continuous")
        kernels.append(kernel)
        # TODO: move this elsewhere!
    
        #model = "EC"

        for factor, columns in self.factor_columns.items():
            kernel = qk.ExchangeableCorrelation(len(columns), zeta=theta)
            kernel = qk.Projection(columns, name=factor, kernel=kernel)
            kernels.append(kernel)
        
        kernel = qk.Tensor(kernels) # + kr.WhiteKernel()
        
        print("Exchangeable Correlation Kernel for Gaussian Process Regression")
        print(kernel)
        logging.debug("Exchangeable Correlation Kernel for Gaussian Process Regression\n{}".format(kernel))
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       alpha=alpha,
                                       normalize_y=True,
                                       n_restarts_optimizer=n_restarts_optimizer)
        #gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, normalize_y=True)
        
        gpr.fit(self.Xd, self.y)

        logging.debug("Fit Gaussian Process Regression:\n{}".format(gpr.kernel_.get_params())) 
        report = self.name_report(gpr)
        logging.info("Exchangeable Correlation Kernel for Gaussian Process Regression\n{}".format(report))
        
        self.gpr_ec = gpr

    def fit_MC(self, alpha=0.01, n_restarts_optimizer=20):
       
        kernels = [kr.ConstantKernel(1.0, (0.001, 1000.0))]
        
        # TODO: consider initializing with each variable's standard deviation
        kernel = qk.Projection(self.continuous_columns, name="continuous")
        kernels.append(kernel)
        # TODO: move this elsewhere!
    
        ec = self.gpr_ec
        params = ec.kernel_.get_params() if ec else {}
    
        found = find_by_name(params)
    
        ec_kernel = {}
        for k, name in found.items():
            if name in self.factors:
                # factor_columns already knows columns
                #columns = params.get("{}__columns".format(k), [])
                ec_kernel[name] = params.get("{}__kernel".format(k), None)  
                    
        for factor, columns in self.factor_columns.items():
            dim = len(columns)
            try:
                theta = ec_kernel[factor].multiplicative_correlation()
                print("Initializing factor '{}' with {}".format(factor, theta))
            except:
                print("Whoops! Factor {} not found".format(factor))
                theta = np.array([0.1] * len(columns))
            kernel = qk.MultiplicativeCorrelation(dim, zeta=theta)
            kernel = qk.Projection(columns, name=factor, kernel=kernel)

            kernels.append(kernel)
        
        kernel = qk.Tensor(kernels) # + kr.WhiteKernel()
        
        print("Multiplicative Correlation Kernel for Gaussian Process Regression")
        print(kernel)
        logging.debug("Multiplicative Correlation Kernel for Gaussian Process Regression\n{}".format(kernel))
        
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       alpha=alpha,
                                       normalize_y=True,
                                       n_restarts_optimizer=n_restarts_optimizer)        
        gpr.fit(self.Xd, self.y)

        logging.debug("Fit Gaussian Process Regression:\n{}".format(gpr.kernel_.get_params())) 
        report = self.name_report(gpr)
        logging.info("Multiplicative Correlation Kernel for Gaussian Process Regression\n{}".format(report))
        
        self.gpr_mc = gpr
        
    def fit_UC(self, theta=None, alpha=0.01, n_restarts_optimizer=20):
       
        kernels = [kr.ConstantKernel(1.0, (0.001, 1000.0))]
        
        # TODO: consider initializing with each variable's standard deviation
        kernel = qk.Projection(self.continuous_columns, name="continuous")
        kernels.append(kernel)
        # TODO: move this elsewhere!
    
        mc = self.gpr_mc
        
        params = mc.kernel_.get_params() if mc else {}
    
        found = find_by_name(params)
    
        mc_kernel = {}
        for k, name in found.items():
            if name in self.factors:
                # factor_columns already knows columns
                #columns = params.get("{}__columns".format(k), [])
                mc_kernel[name] = params.get("{}__kernel".format(k), None)  
                    
        for factor, columns in self.factor_columns.items():
            dim = len(columns)
            m = dim * (dim - 1) //2
            if np.iterable(theta) and (len(theta) == m):
                # use supplied theta
                pass
            else:
                try:
                    theta = mc_kernel[factor].unrestrictive_correlation()
                except:
                    print("Whoops! Factor {} not found".format(factor))
                    theta = np.array([0.1] * m)
            
            kernel = qk.UnrestrictiveCorrelation(dim, zeta=theta)
            kernel = qk.Projection(columns, name=factor, kernel=kernel)
            kernels.append(kernel)
        
        kernel = qk.Tensor(kernels) # + kr.WhiteKernel()
        
        print("Unrestrictive Correlation Kernel for Gaussian Process Regression")
        print(kernel)
        logging.debug("Unrestrictive Correlation Kernel for Gaussian Process Regression\n{}".format(kernel))
        
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       alpha=alpha,
                                       normalize_y=True,
                                       n_restarts_optimizer=n_restarts_optimizer)
        #gpr = GaussianProcessRegressor(kernel=k, alpha=0.001, normalize_y=True)
        
        gpr.fit(self.Xd, self.y)
        # n.b. could scale y but for now handle y with normalize_y in GPR
        
        logging.debug("Fit Gaussian Process Regression:\n{}".format(gpr.kernel_.get_params())) 
        report = self.name_report(gpr)
        logging.info("Unrestrictive Correlation Kernel for Gaussian Process Regression\n{}".format(report))
        self.gpr_uc = gpr

    def fit(self, theta=0.1, alpha=0.01, n_restarts_optimizer=20):
        self.fit_EC(theta=theta, alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)
        # use initial values provided by EC, then by MC
        self.fit_MC(alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)
        self.fit_UC(alpha=alpha, n_restarts_optimizer=n_restarts_optimizer)

    def dummy_data_to_dict(self, datum):
        columns = self.Xd.columns
        return {col : val for col, val in zip(columns, datum)}

    def decode_dummies(self, X, param_set):
        #columns = self.Xd.columns
        decoded = []
        for i in range(X.shape[0]):
            x = X.iloc[i]
            #d = {col : val for col, val in zip(columns, x)}
            #d = self.dummy_data_to_dict(x)
            d = dict(x)
            params = param_set.decode_dummies(d)
            decoded.append(params)
        return decoded

    def predict_penalized(self, gpr, gamma=1.0, delta=1.0):
        def factor_penalty(X, columns):
            W = X[columns]
            return gamma*(np.linalg.norm(W, ord=2) - 1.0)**2 + \
                delta*(np.linalg.norm(W, ord=1) - 1)**2
        

        # n.b. gpr.predict accepts matrix input and returns an array
        # e.g. if X is n * p return an array with shape (p,)        
        return lambda X : gpr.predict(X)[0] + \
            sum(factor_penalty(X, columns) \
                for columns in self.factor_columns.values())
        return lambda X : gpr.predict(X.reshape(1,-1))[0] + sum(factor_penalty(X.reshape(1,-1), columns) for columns in self.factor_columns.values())
        
    def optimize(self, gamma=1.0, delta=1.0, gpr=None, Xd=None):
        """gpr should be self.gpr_uc, _mc, or _ec..."""
        gpr = self._get_gpr(gpr)
        if Xd is None:
            Xd = self.Xd
            
        columns = Xd.columns
            
        lower_bounds = Xd.min(axis=0)
        upper_bounds = Xd.max(axis=0)
        
        bounds = [(lower, upper) for lower, upper in zip(lower_bounds, upper_bounds)]

        result_data = defaultdict(list)
        for i in range(Xd.shape[0]):
            #start_val = Xd.iloc[i].as_matrix().reshape(1,-1)
            #start_val = np.atleast_2d(Xd.iloc[i]).reshape(1,-1)
            start_val = Xd.iloc[i]
            # Fit the GPR model
            predict = lambda X : gpr.predict(X)[0]
            result = sp.optimize.minimize(predict, start_val, method='L-BFGS-B', bounds=bounds)
            # the result will be a mixture of factor values
            # Now penalize points which are not feasible
            for epsilon in [1e-3, 1e-2, 5e-2, 1e-1]:
                predict = self.predict_penalized(gpr, epsilon, epsilon)
            result = sp.optimize.minimize(predict, result.x, method='L-BFGS-B', bounds=bounds)
            rx = result.x
            pred = gpr.predict(result.x.reshape(1,-1))
            for col, val in zip(columns, rx):
                result_data[col].append(val)
            # pred is an ndarray with shape (1,) so unpack it
            result_data['gpr_optimum'].append(pred[0])
        for k , v in result_data.items():
            logging.debug("{} {}".format(k, len(v)))
        # the dictionary will need to be decoded by a ParameterSet object
        result_data = pd.DataFrame(result_data)
        return result_data

    def optimize_recommend(self, param_set,
                           max_recommend=3,
                           gamma=1.0, delta=1.0,
                           gpr=None, Xd=None,
                           return_data=False):
        """Optimizes GPR model, using each data point as initial value
        
        Clusters the result using Affinity Propagation, and returns
        the cluster representatives, choosing the number of clusters
        automatically. The results are decoded into parameter sets."""

        x = self.optimize(gamma=gamma, delta=delta, gpr=gpr, Xd=Xd)
        aff = AffinityPropagation()
        aff.fit(x)
        #x_rec = pd.DataFrame(aff.cluster_centers_, columns=x.columns)
        # select the lowest validation loss from each cluster
        x_rec = pd.concat([x, pd.DataFrame({'cluster_id' : aff.labels_})], axis=1)
<<<<<<< HEAD
        x_rec.sort_values(by=['cluster_id', 'gpr_optimum'], inplace=True)
        x_rec = x_rec.groupby('cluster_id').first()
        x_rec.sort_values(by=['gpr_optimum'], inplace=True)
        
        if max_recommend < 1:
            max_recommend = x.shape[0]
        x_rec = x.iloc[:max_recommend]
        #x_rec.index = range(len(x_rec))
        #x_rec = x_rec.drop(['gpr_optimum'], axis=1)
=======
        clus_rec = x_rec.groupby('cluster_id')
        x_rec = x_rec.iloc[clus_rec['gpr_optimum'].idxmin()]
        x_rec = x_rec.drop(['gpr_optimum','cluster_id'], axis=1)
>>>>>>> 28981818020bd81b77cd336914a79d8ef72a93f0
        paramdictlist = self.decode_dummies(x_rec, param_set)
        if return_data:
            return paramdictlist, x_rec
        else:
            return paramdictlist
    
    # TODO: this is probably obsolete since clustering should be superior
    def optimize_recommend_sorted(self, param_set, max_recommend=0,
                                  gamma=1.0, delta=1.0,
                                  gpr=None, Xd=None,
                                  return_data=False):
        """Optimizes GPR model, using each data point as initial value
        
        Returns one recommendation for each point, 
        however, these may all be the same if they all converge to the
        global minimum.  The results are decoded into parameter sets."""

        x = self.optimize(gamma=gamma, delta=delta, gpr=gpr, Xd=Xd)
        x.sort_values(by='gpr_optimum', inplace=True)
        if max_recommend < 1:
            max_recommend = x.shape[0]
        x_rec = x.iloc[:max_recommend]
        paramdictlist = self.decode_dummies(x_rec, param_set)
        if return_data:
            return paramdictlist, x_rec
        else:
            return paramdictlist

    def _get_gpr(self, gpr=None):
        """Return the best GPR model which has been fit so far
        
        EC < MC < UC"""
        if gpr is None:
            gpr = self.gpr_uc
        if gpr is None:
            gpr = self.gpr_mc
        if gpr is None:
            gpr = self.gpr_ec
        return gpr
            
    def LCB(self, n_sample, gpr=None, Xd=None):
        gpr = self._get_gpr(gpr)
        if Xd is None:
            Xd = self.Xd
        preds = gpr.predict(Xd, return_std=True)
        preds = pd.DataFrame({"prediction" : preds[0], "std_dev" : preds[1]})
        # n.b. lambda is a keyword so change vector of values to alpha
        alpha = ParameterSampler({ "alpha" : expon()}, n_iter=n_sample)
        lcb = pd.DataFrame({"lcb_{}".format(i) : \
                            preds.prediction - \
                            (li["alpha"] * preds.std_dev) \
                            for i, li in enumerate(alpha)})
        # TODO: include X in lcb, to look up parameters from selected values
        return lcb

    def LCB_recommend(self, param_set, max_recommend=0, n_sample=10,
                      gpr=None, Xd=None,
                      return_data=False):
        """Lower Confidence Bound recommendations for GPR model"""
        gpr = self._get_gpr(gpr)
        if Xd is None:
            Xd = self.Xd
        if max_recommend < 1:
            max_recommend = Xd.shape[0]
        lcb = self.LCB(n_sample=n_sample, gpr=gpr, Xd=None)
        lcb['minimum'] = lcb.min(axis=1)
        lcb.sort_values(by='minimum', inplace=True)
        Xdmin = Xd.iloc[lcb.index[:max_recommend]]
        recommend = self.decode_dummies(Xdmin, param_set)
#        recommend = []
#        for i in range(Xdmin.shape[0]):
#            x = Xdmin.iloc[i]
#            recommend.append(self.decode_dummies(x, param_set))
        if return_data:
            return recommend, Xdmin
        else:
            return recommend

    def name_report(self, gpr):
        """Report correlations for factors.
        
        Assumes the factor kernel can be found by searching Projection kernels
        by name."""
        report = []
        factors = self.factors
        params = gpr.kernel_.get_params()
        found = find_by_name(params)
        report.append("*"*50)
        for key, name in found.items():
            if name in factors:  
                try:
                    report.append(name)
                    report.append(str(params["{}__kernel".format(key)].correlation))
                    report.append("*"*50)
                except:
                    pass
        return "\n".join(report)

def report(gpr):
    for k, ker in gpr.kernel_.get_params().items():
        try:
            print(k)
            print(ker)
            print(ker.theta)
            print(ker.correlation)
        except:
            pass

def find_by_name(params):
    """Finds keys which have 'name' parameters, and associated name
    
    Currently only Projection kernels have names
    Nested kernel can be obtained from parameter "{}__kernel".format(key)
    params should be obtained from gpr.kernel_.get_params()
    """
    found = {}
    for k, v in params.items():
        split = k.split("__")
        name = split[-1] if len(split) > 1 else ""
        if name == 'name':
            found["__".join(split[:-1])] = v
        #print(k, split, v)
    return found

def name_report(gpr, factors):
    """Report correlations for factors.
    
    Assumes the factor kernel can be found by searching Projection kernels
    by name."""
    report = []
    params = gpr.kernel_.get_params()
    found = find_by_name(params)
    report.append("*"*50)
    for key, name in found.items():
        if name in factors:  
            try:
                report.append(name)
                report.append(str(params["{}__kernel".format(key)].correlation))
                report.append("*"*50)
            except:
                pass
    return "\n".join(report)
