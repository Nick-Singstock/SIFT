# -*- coding: utf-8 -*-
"""
DRIFT ML method created and used for Chevrel Stability ML model creation.

More information on the machine learned descriptor and the ML method are available 
in the main text and SI of the corresponding manuscript. 

Inputs:
    data (dict): All individual entries and their corresponding features 
        and target property values.
    target_property (str): 
"""

import pandas as pd
import numpy as np
from scipy.stats import pearsonr as pearson
from scipy.stats import linregress
import operator
from Equation import Expression # available at: https://pypi.org/project/Equation/
import random
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

class drift():
    
    def __init__(self, data, target_prop, depth = 2, corr_function = 'rmse',
                 use_units = True, unit_data = None, keep_previous_set = True, 
                 normalize = True, norm_constants = None, verbose = True,
                 max_new_features = 500, max_complexity = 6, 
                 testing_split = 0.2, k_fold = (1,5)):
        
        # set inputs
        self.data_dic = data
        assert all([target_prop in v for k,v in data.items()]), 'Target property does not exist for all data entries.'
        self.target = target_prop
        assert depth > 0 and depth <= 6, 'Depth must be in range 0 < depth <= 6. This line can be removed for depth > 6.'
        self.depth = depth
        self.corrf = corr_function
        self.verbose = verbose
        self.keep_previous = keep_previous_set
        self.max_new = max_new_features
        self.max_complexity = max_complexity
        self.len_data = len(data)
        # get all feature names
        for k,v in data.items():
            all_features = v.keys()
            break
        self.features = [f for f in all_features if f != target_prop]
        # set units
        if not use_units:
            self.units = {f: '' for f in self.features}
        else:
            assert unit_data is not None, 'Unit dictionary must be input if use_units == True.'
            self.units = unit_data
        # normalize
        if normalize:
            self.normalize_data(norm_constants)
        # get train/test split
        self.split_train_test(k_fold, testing_split)
        
        self.setup_calc()
            
    def normalize_data(self, norms):
        if norms is None:
            def regularize(array, normalize = True):
                '''
                if normalize = True, returns values in [0,1], else returns values in [-1,1]
                '''
                amax = np.max(array)
                amin = np.min(array)
                reg = np.max(np.abs([amax, amin]))
                diff = amax - amin
                if normalize:
                    return (array-amin)/diff
                return array/reg
            
            for f in self.features:
                feat_vals = np.array([ v[f] for k,v in self.data_dic.items() ])
                reg_feats = regularize(feat_vals)
                for i, k in enumerate(self.data_dic):
                    self.data_dic[k][f] = reg_feats[i]
        else:
            for k,v in self.data_dic.items():
                for f in self.features:
                    self.data_dic[k][f] = v[f] / norms[f]
    
    def split_train_test(self, k_fold, test_split = 0.2):
        if k_fold is False or k_fold is None:
            test_len = int(self.len_data * test_split)
            self.test_set = random.sample(list(self.data_dic.keys()), test_len)
            self.train_set = [k for k in self.data_dic.keys() if k not in self.test_set]
        else: # do k-fold testing
            divisions = k_fold[1]
            divs = {str(i): [] for i in range(divisions)}
            for i,k in enumerate(self.data_dic):
                divs[str(i%divisions)].append(k)
            self.test_set = [k for k in self.data_dic.keys() if k in divs[str(k_fold[0])]]
            self.train_set = [k for k in self.data_dic.keys() if k not in self.test_set]

        self.test_data = {k:v for k,v in self.data_dic.items() if k in self.test_set}
        self.train_data = {k:v for k,v in self.data_dic.items() if k in self.train_set}

    def setup_calc(self):
        self.data_columns = [self.target] + self.features
        data_matrix = [ [v[kk] for kk in self.data_columns] for k,v in self.train_data.items() ]
        self.dataset = pd.DataFrame(data_matrix, columns = self.data_columns)
        self.complexity_dic = {k: 1 for k in self.features}
        self.keep_dic_set = {'0': self.features}
        
    '''
    define math functions
    '''
    def rsquared(self, x, y):
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return r_value**2
    
    def rmse(self, x, y):
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        y_pred = x * slope + intercept
        error = y - y_pred
        return np.sqrt(np.mean(error**2))
        
    def mae(self, x, y):
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        y_pred = x * slope + intercept
        error = y - y_pred
        return np.mean(np.abs(error))
    
    def max_error(self, x, y):
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        y_pred = x * slope + intercept
        error = y - y_pred
        return np.max(np.abs(error))
        
    def check_units(self, f1, f2, m):
        if self.units[f1] == self.units[f2]:
            if m in ['+','-']:
                return True, self.units[f1]
            if m in ['/']:
                return True, ''
        if m in ['*']: # does not allow 2 features with different units to be multiplied together 
            if self.units[f1] != '' and self.units[f2] != '':
                return False, 'None'
            else:
                return True, self.units[f1]+self.units[f2]
        if m in ['+','-','/']:
            return False, 'None'
    
    def get_corr(self, x, y):
        if self.corrf is 'pearson':
            cf = pearson
            return cf(x,y)[0]
        elif self.corrf is 'rsquared':
            cf = self.rsquared
        elif self.corrf is 'rmse':
            cf = self.rmse
        elif self.corrf is 'mae':
            cf = self.mae
        elif self.corrf is 'max_error':
            cf = self.max_error
        return cf(x, y)
    
    def math_2(self):
        return {'+': np.add,
                '-': np.subtract,
                '*': np.multiply,
                '/': np.divide}
    def math_1(self):
        return {}
        return {'^2': np.square}
    
    def get_feat_values(self, feats, dic):
        vals = []
        for f in feats:
            if f in dic:
                # feature already in dictionary
                vals.append(dic[f])
            else:
                # feature is composed of math of other features
                vals.append(self.evaluate(f, dic))
        return vals
    
    # primary function for feature math
    def evaluate(self, s, dic):
        counter = 0
        var_map = {}
        arg_order = []
        for k in dic.keys():
            if k in s:
                counter += 1
                s = s.replace(k, 'x'+str(counter))
                var_map['x'+str(counter)] = k
                arg_order.append('x'+str(counter))
        # vectorize data based upon string math expression, currently limited to 11 variables
        ex = Expression(s, argorder = arg_order.copy())
        
        assert counter > 0, 'Error: no features found in feature string: '+ s
        if counter == 1: return ex(dic[var_map[arg_order[0]]])
        if counter == 2: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]])
        if counter == 3: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]])
        if counter == 4: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]],
                                   dic[var_map[arg_order[3]]])
        if counter == 5: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]],
                                   dic[var_map[arg_order[3]]], dic[var_map[arg_order[4]]])
        if counter == 6: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]],
                                   dic[var_map[arg_order[3]]], dic[var_map[arg_order[4]]], dic[var_map[arg_order[5]]])
        if counter == 7: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]],
                                   dic[var_map[arg_order[3]]], dic[var_map[arg_order[4]]], dic[var_map[arg_order[5]]],
                                   dic[var_map[arg_order[6]]])
        if counter == 8: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]],
                                   dic[var_map[arg_order[3]]], dic[var_map[arg_order[4]]], dic[var_map[arg_order[5]]],
                                   dic[var_map[arg_order[6]]], dic[var_map[arg_order[7]]])
        if counter == 9: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]],
                                   dic[var_map[arg_order[3]]], dic[var_map[arg_order[4]]], dic[var_map[arg_order[5]]],
                                   dic[var_map[arg_order[6]]], dic[var_map[arg_order[7]]], dic[var_map[arg_order[8]]])
        if counter == 10: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]],
                                   dic[var_map[arg_order[3]]], dic[var_map[arg_order[4]]], dic[var_map[arg_order[5]]],
                                   dic[var_map[arg_order[6]]], dic[var_map[arg_order[7]]], dic[var_map[arg_order[8]]],
                                   dic[var_map[arg_order[9]]])
        if counter == 11: return ex(dic[var_map[arg_order[0]]], dic[var_map[arg_order[1]]], dic[var_map[arg_order[2]]],
                                   dic[var_map[arg_order[3]]], dic[var_map[arg_order[4]]], dic[var_map[arg_order[5]]],
                                   dic[var_map[arg_order[6]]], dic[var_map[arg_order[7]]], dic[var_map[arg_order[8]]],
                                   dic[var_map[arg_order[9]]], dic[var_map[arg_order[10]]])
        assert False, 'Number of features is too large (%i), currently supports up to 11 features.'%(counter)
        
    # primary function for growing new features 
    def grow_features(self, df_features, all_corr, units = True, keep_size = 20,
                      keep_init_set = True, exclude_same = True, itt = 1):
        corr_dic = {k:v for k,v in all_corr.items()}
        index = {t: df_features.columns.get_loc(t) for t in df_features.columns}
        
        counter = 0
        max_count = len(df_features.columns)
        if self.verbose: print('Generating Features')
        # scan through feature 1 
        for i1,f1 in enumerate(list(df_features.columns)):
            counter += 1
            xxx = counter / max_count
            if self.verbose: print("\rProgress {:2.1%}".format(xxx), end="\r")
            
            # vector of feature 1
            vec1 = df_features.iloc[:, index[f1]]
            # apply single-vector operators
            for m,fn in self.math_1().items():
                if units:
                    if self.units[f1] != '':
                        continue
                if m in ['^2']:
                    full = '('+f1+' '+m+')'
                else:
                    continue
                if full in all_corr:
                    continue
                if units:
                    # add units of new feature
                    self.units[full] = ''
                # vectorize new feature to test correlation but do not save to conserve memory
                vec = fn(vec1)
                corr = self.get_corr(vec, self.Y, return_coeffs = False)
                if str(corr) == 'nan':
                    continue
                corr_dic[full] = corr
                self.complexity_dic[full] = self.complexity_dic[f1]
            
            # scan through feature 2 and do feature math
            for i2,f2 in enumerate(list(df_features.columns)):
                # get vector for feature 2
                vec2 = df_features.iloc[:, index[f2]]
                # scan through math operations and perform math and test correlation with target
                for m,fn in self.math_2().items():
                    if m in ['+','*'] and i1 < i2: # exclude these from showing up twice
                        continue
                    if m == '/' and any([ xx==0 for xx in vec2 ]):
                        continue
                    if units: # check units between features
                        check, u = self.check_units(f1,f2,m)
                        if not check:
                            continue
                    full = '('+f1+' '+m+' '+f2+')'
                    if full in all_corr:
                        continue
                    if units:
                        # add units of new feature
                        self.units[full] = u
                    # vectorize new feature to test correlation but do not save to conserve memory
                    vec = fn(vec1, vec2)
                    # test correlation between new feature and target
                    corr = self.get_corr(vec, self.Y)
                    if str(corr) == 'nan':
                        continue
                    corr_dic[full] = corr
                    self.complexity_dic[full] = self.complexity_dic[f1] + self.complexity_dic[f2]
        
        # down select only the top-rated features based on keep_size
        dist = min([ keep_size, len(corr_dic) ])
        if self.verbose: print('\nReducing Features')
        red_corr_dic = {k:v for k,v in corr_dic.items() if self.complexity_dic[k] <= self.max_complexity}
        
        if self.corrf in ['pearson','rsquared',]:
            # higher is better
            keep_dic = dict(sorted(red_corr_dic.items(), key=operator.itemgetter(1), reverse=True)[:dist])
        elif self.corrf in ['rmse','mae','max_error']:
            # lower is better
            keep_dic = dict(sorted(red_corr_dic.items(), key=operator.itemgetter(1), reverse=True)[-dist:])
        # remove features with identical correlations
        if exclude_same:
            tmp = {}
            for k,v in keep_dic.items():
                if any([v2 == v for v2 in tmp.values()]):
                    continue
                tmp[k] = v
            keep_dic = tmp
        # retain initial feature set if requested so these can always be used for additional math
        if keep_init_set:
            data_columns = self.features + [k for k in keep_dic if k not in self.features]
        else:
            data_columns = [k for k in keep_dic]
        
        if self.keep_previous:
            data_columns = [k1 for k1 in self.keep_dic_set[str(itt-1)]] + [k for 
                            k in keep_dic if k not in self.keep_dic_set[str(itt-1)]]
        # make new data matrix based on best features available (old and new)
        data_matrix = []
        for k,v in self.train_data.items():
            # get_feat_values() vectorizes the new best features 
            data_matrix.append(self.get_feat_values(data_columns, v))
        df = pd.DataFrame(data_matrix, columns = data_columns)
        # return df with best features, dictionary of best feature correlations, and dictionary of all feature correlations
        return df, keep_dic, corr_dic

    def run(self):
        index_dic = {t: self.data_columns.index(t) for t in self.data_columns}
        # vectorize target
        self.Y = self.dataset.iloc[:,index_dic[self.target]]        
        # get intital correlation dictionary
        corr_dic = {k: self.get_corr(self.dataset.iloc[:,index_dic[k]], self.Y) for k in self.features}
        
        # first iteration performed based on keep_features set
        if self.verbose: print('\n---- Depth:  1 ----')
        df_new, keep_dic, corr_dic = self.grow_features(self.dataset.iloc[:, 1:], corr_dic, 
                                                        units = True, keep_size = self.max_new)
        best_corr = min(keep_dic.values())
        if self.verbose: print('Best Corr: %.4f' % best_corr)
        
        # perform following iterations based on best features from previous iteration
        self.keep_dic_set['1'] = keep_dic
        if self.depth > 1:
            for iii in range(self.depth-1):
                if self.verbose: print('\n---- Depth: ', str(iii+2),'----')
                df_new, keep_dic, corr_dic = self.grow_features(df_new, corr_dic, units = True, 
                                                           keep_size = self.max_new, itt = iii+2)
                if self.corrf in ['mae','rmse','max_error']:
                    best_corr = min(keep_dic.values())
                else:
                    best_corr = max(keep_dic.values())
                if self.verbose: print('Best Corr: %.4f' % best_corr)
                self.keep_dic_set[str(iii+2)] = keep_dic
        best_func = [k for k,v in keep_dic.items() if v == best_corr][0]
        if self.verbose:
            print('\n--- Descriptor: ---\n'+best_func + '\n'+self.corrf+': %.4f' % best_corr)
        self.total_descriptors = len(corr_dic)
        if self.verbose:
            print('\nTotal Descriptors Generated:', self.total_descriptors)
        self.analyze(best_func)
        
        return {'descriptor': best_func,
                'mae': {'train':self.train_mae,'test':self.test_mae},
                'rmse': {'train':self.train_rmse,'test':self.test_rmse},
                'correlation_function': {'train':self.train_corr,'test':self.test_corr},
                'slope': self.slope, 'intercept': self.intercept,
                'descriptor_units': self.units[best_func],
                'additional_data': {'total_descriptors': self.total_descriptors,
                                    'retained_descriptors': self.keep_dic_set,
                                    'training_set': self.train_set,
                                    'testing_set': self.test_set}}
    
    def analyze(self, descriptor):
        y_train = np.array([ v[self.target] for k,v in self.train_data.items() ])
        x_train = np.array([self.evaluate(descriptor, v) for k,v in self.train_data.items()])
        # get training set accuracy
        self.slope, self.intercept, r_value, p_value, std_err = linregress(x_train, y_train)
        y_pred = x_train * self.slope + self.intercept
        self.error_train = y_train - y_pred
        self.all_errors_train = {k: self.error_train[i] for i,k in enumerate(self.train_data.keys())}
        self.train_corr = self.get_corr(x_train, y_train)
        self.train_mae = np.mean(np.absolute(self.error_train))
        self.train_rmse = np.sqrt(np.mean(np.power(self.error_train, 2)))
        if self.verbose:
            print('Slope:', self.slope, '\tInt.:', self.intercept)
            print('\nTRAIN SET:\n')
            print('\tmae:', self.train_mae)
            print('\trmse:', self.train_rmse)
            print('\tmax error:', max(np.absolute(self.error_train)))
        # get test set accuracy 
        if len(self.test_data) > 0:
            test_Y = np.array([ v[self.target] for k,v in self.test_data.items() ])
            test_x = np.array([self.evaluate(descriptor, v) for k,v in self.test_data.items()])
            y_pred = test_x * self.slope + self.intercept
            self.test_error = test_Y - y_pred
            self.all_test_errors = {k: self.test_error[i] for i,k in enumerate(self.test_data.keys())}
            self.test_corr = self.get_corr(test_x, test_Y)
            self.test_mae = np.mean(np.absolute(self.test_error))
            self.test_rmse = np.sqrt(np.mean(np.power(self.test_error, 2)))
            if self.verbose:
                print('\nTEST SET:\n')
                print('\tmae:', self.test_mae)
                print('\trmse:', self.test_rmse)
                print('\tmax error:', max(np.absolute(self.test_error)))

