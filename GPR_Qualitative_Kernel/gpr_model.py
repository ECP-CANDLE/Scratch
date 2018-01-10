# -*- coding: utf-8 -*-

from collections import defaultdict

from sklearn.gaussian_process import GaussianProcessRegressor
#from sklearn.gaussian_process.kernels import RBF, ConstantKernel
#from kernels import RBF, ConstantKernel
import kernels as kr
import qualitative_kernels as qk
import parameter_set as prs
from sklearn.model_selection import ParameterSampler
from sklearn.cluster import AffinityPropagation

import pandas as pd
import numpy as np
import scipy as sp
from scipy.stats.distributions import expon

import logging
logging.basicConfig(filename='GPR_Model.log',level=logging.DEBUG)

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
        
        kernel = qk.Tensor(kernels)
        
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
                theta = np.array([theta] * len(columns))
            kernel = qk.MultiplicativeCorrelation(dim, zeta=theta)
            kernel = qk.Projection(columns, name=factor, kernel=kernel)

            kernels.append(kernel)
        
        kernel = qk.Tensor(kernels)
        print("Multiplicative Correlation Kernel for Gaussian Process Regression")
        print(kernel)
        logging.debug("Multiplicative Correlation Kernel for Gaussian Process Regression\n{}".format(kernel))
        
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       alpha=alpha,
                                       normalize_y=True,
                                       n_restarts_optimizer=20)        
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
    
        uc_kernel = {}
        for k, name in found.items():
            if name in self.factors:
                # factor_columns already knows columns
                #columns = params.get("{}__columns".format(k), [])
                uc_kernel[name] = params.get("{}__kernel".format(k), None)  
                    
        for factor, columns in self.factor_columns.items():
            dim = len(columns)
            m = dim * (dim - 1) //2
            if np.iterable(theta) and (len(theta) == m):
                # use supplied theta
                pass
            else:
                try:
                    theta = uc_kernel[factor].unrestrictive_correlation()
                except:
                    print("Whoops! Factor {} not found".format(factor))
                    theta = np.array([0.1] * m)
            
            kernel = qk.UnrestrictiveCorrelation(dim, zeta=theta)
            kernel = qk.Projection(columns, name=factor, kernel=kernel)
            kernels.append(kernel)
        
        kernel = qk.Tensor(kernels)
        print("Multiplicative Correlation Kernel for Gaussian Process Regression")
        print(kernel)
        logging.debug("Unrestrictive Correlation Kernel for Gaussian Process Regression\n{}".format(kernel))
        
        gpr = GaussianProcessRegressor(kernel=kernel,
                                       alpha=alpha,
                                       normalize_y=True,
                                       n_restarts_optimizer=20)
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
            d = self.dummy_data_to_dict(x)
            params = param_set.decode_dummies(d)
            decoded.append(params)
        return decoded

    def predict_penalized(self, gpr, gamma=1.0, delta=1.0):
        def factor_penalty(X, columns):
            W = X[columns]
            return gamma*(np.linalg.norm(W, ord=2) - 1.0)**2 + delta*(np.linalg.norm(W, ord=1) - 1)**2
        
        return lambda X : gpr.predict(X.reshape(1,-1)) + sum(factor_penalty(X, columns) for columns in self.factor_columns.values())
        
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
            # TODO: pick a float between 0 and 1 for each value
            # rescale using bounds as range, i.e. r * (upper - lower) + lower
            start_val = Xd.iloc[i].as_matrix().reshape(1,-1)
            #start_val = np.array(start_val).reshape(-1, 1)
            #start_val = Xs[yidxmin].reshape(-1,  1)
            # Fit the GPR model
            predict = self.predict_penalized(gpr)
            result = sp.optimize.minimize(predict, start_val, method='L-BFGS-B', bounds=bounds)
            rx = result.x
            pred = gpr.predict(result.x)
            for col, val in zip(columns, rx):
                result_data[col].append(val)
            # pred is an ndarray with shape (1,) so unpack it
            result_data['gpr_optimum'].append(pred[0])
        for k , v in result_data.items():
            logging.debug("{} {}".format(k, len(v)))
        # the dictionary will need to be decoded by a ParameterSet object
        result_data = pd.DataFrame(result_data)
        return result_data

    def optimize_recommend(self, param_set, gamma=1.0, delta=1.0,
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
        clus_rec = x_rec.groupby('cluster_id')
        x_rec = x_rec.iloc[clus_rec.idxmin()['gpr_optimum']]
        x_rec = x_rec.drop(['gpr_optimum','cluster_id'], axis=1)
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
        split = k.split("__", 1)
        name = split[1] if len(split) > 1 else ""
        if name == 'name':
            found[split[0]] = v
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
