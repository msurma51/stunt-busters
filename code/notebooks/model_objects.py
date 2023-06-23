# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 05:51:44 2023

@author: surma
"""
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from itertools import product
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import ConfusionMatrixDisplay, RocCurveDisplay
from sklearn.metrics import f1_score, roc_auc_score

class LOWOCV():
    def __init__(self, data, features, target_col, num_weeks, classifier, clf_type = 'tree', 
                 scaler = StandardScaler(), params = {}):
        self.data = data
        self.features = features
        self.target_col = target_col
        self.num_weeks = num_weeks
        self.params = params
        self.clf = classifier
        self.clf_type = clf_type
        self.scaler = scaler
        self.probs = []
        self.preds = []
        self.train_aucs = []
        if self.clf_type == 'tree':    
            self.feature_importances = []
        elif self.clf_type == 'lm':
            self.coefficients = []
    def run_cv(self):
        self.probs = []
        self.preds = []
        if self.clf_type == 'tree':    
            self.feature_importances = []
        elif self.clf_type == 'lm':
            self.coefficients = []
        for week in range(1,self.num_weeks+1):
            X_train, X_test, y_train, y_test = self.train_test_split_towo(holdout_week = week)
            if len(self.params) > 0:
                self.clf.set_params(**self.params)
            if self.clf_type == 'tree':
                self.clf.fit(X_train,
                        y_train,
                        eval_set = [(X_test, y_test)],
                        verbose = False)
                self.preds.extend(self.clf.predict(X_test).tolist())
                self.probs.extend(self.clf.predict_proba(X_test)[:,1].tolist())
                self.train_aucs.append(roc_auc_score(y_train,self.clf.predict_proba(X_train)[:,1]))
                self.feature_importances.append(self.clf.feature_importances_)
            elif self.clf_type == 'lm':
                X_train_scaled = self.scaler.fit_transform(X_train)
                X_test_scaled = self.scaler.transform(X_test)
                self.clf.fit(X_train_scaled, y_train)
                self.preds.extend(self.clf.predict(X_test_scaled).tolist())
                self.probs.extend(self.clf.predict_proba(X_test_scaled)[:,1].tolist())
                self.train_aucs.append(roc_auc_score(y_train,self.clf.predict_proba(X_train_scaled)[:,1]))
                self.coefficients.append(self.clf.coef_[0])  
                            
    def train_test_split_towo(self, holdout_week):
        training_data = self.data[(self.data['week'] < holdout_week) |
                                  (self.data['week'].isin(range(holdout_week+1, self.num_weeks+1)))]
        test_data = self.data[self.data['week'] == holdout_week]
        X_train, y_train = training_data[self.features], training_data[self.target_col]
        X_test, y_test = test_data[self.features], test_data[self.target_col]
        return X_train, X_test, y_train, y_test 
    def get_scores_by_frame(self, frame_center):    
        f1_lambda = lambda x: f1_score(x['rush_win'], x['pred'])
        auc_lambda = lambda x: roc_auc_score(x['rush_win'], x['prob'])
        df = self.data.copy()
        df['pred'] = self.preds
        df['prob'] = self.probs
        scores_by_frame = df.groupby(frame_center)['pred'].count().rename('count').to_frame()
        scores_by_frame['pred'] = df.groupby(frame_center)['pred'].mean()
        scores_by_frame['prob'] = df.groupby(frame_center)['prob'].mean()
        scores_by_frame['f1_score'] = df.groupby(frame_center).apply(f1_lambda)
        scores_by_frame['auc_score'] = df.groupby(frame_center).apply(auc_lambda)
        return scores_by_frame
    def get_feature_importances_df(self):
        return pd.DataFrame(self.feature_importances, columns = self.features)
    def get_coefficients_df(self):
        return pd.DataFrame(self.coefficients, columns = self.features)
    def get_top_n_features(self,n,return_scores = False):
        if self.clf_type == 'tree':
            feature_scores_df = self.get_feature_importances_df()
        elif self.clf_type == 'lm':
            feature_scores_df = self.get_coefficients_df()
        top_feature_importances_sorted = feature_scores_df.mean().abs().sort_values(ascending = False)
        if not return_scores:
            return top_feature_importances_sorted.index[:n].tolist()
        else:
            return top_feature_importances_sorted[:n]
    def get_auc(self):
        return roc_auc_score(self.data[self.target_col],self.probs)
    def get_avg_train_auc(self):
        return sum(self.train_aucs) / len(self.train_aucs)
    def get_f1(self):
        return f1_score(self.data[self.target_col],self.preds)
    def display_confusion_matrix(self):
        ConfusionMatrixDisplay.from_predictions(self.data[self.target_col],self.preds)
    def display_roc_curve(self):
        RocCurveDisplay.from_predictions(self.data[self.target_col],self.probs)
    def set_features(self, new_features):
        self.features = new_features
    def set_params(self, new_params):
        self.params = new_params
    def copy(self):
        return self
    
class FeatureSelector():
    def __init__(self, cv_object, eval_metric = 'auc', threshold = .001):
        self.cv_object = cv_object
        assert eval_metric in ('auc', 'f1'), "Invalid evaluation metric. Choose either 'auc' or 'f1'"
        self.eval_metric = eval_metric
        self.threshold = threshold
        self.scores = []
        self.best_score = 0.0
        self.feature_lists = []
        self.best_features = []
        self.dropped_features = {}
        self.feature_states = []
    def sw_select_cust(self, win_corr = pd.Series(dtype = 'float64'), start_features = [], n_test_features = 3,
                       fw_select_type = 'corr', bw_select_type = 'tree', holdout_steps = 4, exclude_categorical = False):
        # Check for correct input values
        select_types = ('corr', 'tree', 'coef', 'all', 'none')
        select_error_strings = ["Feature selection is based on correlation with target ('corr')",
                                "initial model feature importance ('tree')",
                                "initial model feature coefficients ('coef')",
                                "or considers all available features ('all') to add or remove. ",
                                "Select 'none' on either type to perform just the other type."]
        select_error_string = ' '.join(select_error_strings)
        assert fw_select_type in select_types and bw_select_type in select_types, select_error_string
        # Initialize model and feature selection paradigm
        fw_selection_scores = pd.Series(dtype = 'float64')
        if fw_select_type in ('tree','coef'):
            self.cv_object.run_cv()
            if fw_select_type == 'tree':    
                fw_selection_scores = self.cv_object.get_feature_importances_df().mean()
            else:
                fw_selection_scores = self.cv_object.get_coefficients_df().mean().abs()
        all_features = set(self.cv_object.features)
        if fw_select_type == 'corr':
            cat_features = list(all_features.difference(set(win_corr.index)))
            if len(cat_features) > 0:
                if exclude_categorical == False:
                    avg_corr_mag = win_corr.abs().mean()
                    cat_series = pd.Series([avg_corr_mag]*len(cat_features), index = cat_features)
                    win_corr = pd.concat((win_corr, cat_series))
                else:
                    all_features = all_features.difference(cat_features)
            fw_selection_scores = win_corr.abs()
        self.dropped_features = all_features.difference(set(start_features))
        best_score_local = 0.0
        add_holdout = {}
        drop_holdout = {}
        feature_state = {'curr_features': start_features, # Current features being evaluated (list)
                         'free_features': all_features.difference(set(start_features)), # Features available (set)
                         # to add/drop in the next step
                         'to_add_step': True} # Next step 
        if len(start_features) > 0:
            if feature_state in self.feature_states:
                return f'Feature state with starting features {start_features} already considered'
            self.feature_states.append(feature_state)
            # Potential first CV run after object is initialized (if not starting from 0 features)
            if len(self.scores) == 0:
                self.init_run(features = start_features)
                best_score_local = self.best_score
            elif start_features in self.feature_lists:
                score_index = self.feature_lists.index(start_features)
                best_score_local = self.scores[score_index]
            else:
                best_score_local = self.run_test(start_features)  
            print(f'Running modified step-wise feature selector with {len(start_features)} starting features')
        else:
            self.cv_object.set_features(start_features)
        
        change_at = pd.Series({'add': True, 'drop': True})
        while change_at.any():
            if fw_select_type == 'none':
                change_at['add'] = False
            else:
                # Add feature step
                add_holdout = {key:value-1 for key,value in add_holdout.items() if value != 1}
                # If the method was just called or if a feature was added the feature_state is up-to-date
                if change_at['drop'] == True:
                    add_candidates = feature_state['free_features']
                # If not we need to update it
                else:
                    add_candidates = all_features.difference(set(feature_state['curr_features']) |
                                                             set(add_holdout.keys()))
                    feature_state['free_features'] = add_candidates
                    feature_state['to_add_step'] = True
                if fw_select_type in ('corr', 'tree', 'coef'):
                    add_candidates_imp = fw_selection_scores.loc[list(add_candidates)]
                    add_candidates_sorted = add_candidates_imp.sort_values(ascending = False).index.tolist()
                    test_feature_groups = [tuple(add_candidates_sorted[i*n_test_features:(i+1)*n_test_features])
                                          for i in range(len(add_candidates_sorted) // n_test_features)]
                    remainder = len(add_candidates_sorted) % n_test_features
                    if remainder > 0:
                        test_feature_groups.append(tuple(add_candidates_sorted[-1*remainder:]))    
                if fw_select_type == 'all':
                    test_feature_groups = [sorted(add_candidates)]
                for group in test_feature_groups:
                    test_scores = []
                    for add_candidate in group:
                        test_features = feature_state['curr_features'] + [add_candidate]
                        if test_features in self.feature_lists:
                            score_index = self.feature_lists.index(test_features)
                            print('Feature list already tested:', test_features)
                            test_score = self.scores[score_index]  
                            print(f'Test score = {test_score:0.4f}')
                        else:
                            test_score = self.run_test(test_features)
                        test_scores.append(test_score)
                    sorted_scores = sorted(zip(test_scores, group), reverse = True)
                    max_candidates = [tup for tup in sorted_scores if tup[0] > best_score_local + self.threshold]
                    while len(max_candidates) > 0:
                        # Create resulting feature state if the add candidate is selected
                        # If this feature state has been considered previously, do not select the add candidate
                        feature_to_add = max_candidates[0][1]
                        test_features = feature_state['curr_features'] + [feature_to_add]
                        test_feature_state = feature_state.copy()
                        test_feature_state['curr_features'] = test_features
                        holdout_features = {feature for feature in drop_holdout if drop_holdout[feature] > 1}
                        test_feature_state['free_features'] = set(feature_state['curr_features']).difference(holdout_features)
                        test_feature_state['to_add_step'] = False
                        if test_feature_state in self.feature_states:
                            max_candidates.remove(max_candidates[0])
                            print('Feature state already considered:',test_feature_state)
                            continue
                        print(f'Adding feature {feature_to_add}')
                        best_score_local = max_candidates[0][0]
                        feature_state = test_feature_state
                        self.feature_states.append(feature_state)
                        drop_holdout[feature_to_add] = holdout_steps
                        change_at['add'] = True
                        break
                    if len(max_candidates) > 0:
                        break
                    else:
                        change_at['add'] = False
                        print(f'Failed to add one of {len(group)} candidates')
                # Refit cv object to best features if any other feature sets were tested
                if len(test_feature_groups) > 0:
                    if change_at['add'] == False:
                        print('NO FEATURES ADDED')
                    self.fit_best(local_features = feature_state['curr_features'])
                else:
                    change_at['add'] = False
                    print('No features to add')
                # End process if no changes were made in the past drop/add cycle
                if not change_at.any():
                    break
            
            # Drop feature step
            if bw_select_type == 'none':
                change_at['drop'] = False
            else:
                drop_holdout = {key:value-1 for key,value in drop_holdout.items() if value != 1}
                # If a feature was added in the previous step the feature state is up-to-date
                if change_at['add'] == True:
                    drop_candidates = feature_state['free_features']
                # If not we must update it
                else:
                    drop_candidates = set(feature_state['curr_features']).difference(set(drop_holdout.keys()))
                    feature_state['free_features'] = drop_candidates
                    feature_state['to_add_step'] = False
                if bw_select_type in ('corr', 'tree', 'coef'):
                    if bw_select_type == 'corr':
                        drop_candidates_sorted = win_corr.loc[list(drop_candidates)].abs().sort_values().index.tolist()
                    else:
                        if bw_select_type == 'tree':
                            drop_candidates_imp = self.cv_object.get_feature_importances_df().mean().loc[list(drop_candidates)]
                        else:
                            drop_candidates_imp = self.cv_object.get_coefficients_df().mean().loc[list(drop_candidates)]
                        drop_candidates_sorted = drop_candidates_imp.sort_values().index.tolist()
                    test_feature_groups = [tuple(drop_candidates_sorted[i*n_test_features:(i+1)*n_test_features])
                                          for i in range(len(drop_candidates_sorted) // n_test_features)]
                    remainder = len(drop_candidates_sorted) % n_test_features
                    if remainder > 0:
                        test_feature_groups.append(tuple(drop_candidates_sorted[-1*remainder:]))    
                elif bw_select_type == 'all':
                    test_feature_groups = [sorted(drop_candidates)]
                for group in test_feature_groups:
                    test_scores = []
                    for drop_candidate in group:
                        test_features = feature_state['curr_features'].copy()
                        test_features.remove(drop_candidate)
                        if test_features in self.feature_lists:
                            score_index = self.feature_lists.index(test_features)
                            print('Feature set already tested:', test_features)
                            test_score = self.scores[score_index]  
                            print(f'Test score = {test_score:0.4f}')                        
                        else:
                            test_score = self.run_test(test_features)
                        test_scores.append(test_score)
                    sorted_scores = sorted(zip(test_scores, group), reverse = True)
                    max_candidates = [tup for tup in sorted_scores if tup[0] > best_score_local + self.threshold]
                    while len(max_candidates) > 0:
                        feature_to_drop = max_candidates[0][1]
                        test_features = feature_state['curr_features'].copy()
                        test_features.remove(feature_to_drop)
                        test_feature_state = feature_state.copy()
                        test_feature_state['curr_features'] = test_features
                        holdout_features = {feature for feature in add_holdout if add_holdout[feature] > 1}
                        test_feature_state['free_features'] = all_features.difference(set(feature_state['curr_features']) |
                                                                                      holdout_features)
                        test_feature_state['to_add_step'] = True
                        if test_feature_state in self.feature_states:
                            max_candidates.remove(max_candidates[0])
                            print('Feature state already considered:',test_feature_state)
                            continue
                        print(f'Dropping feature {feature_to_drop}')
                        best_score_local = max_candidates[0][0]
                        feature_state = test_feature_state
                        self.feature_states.append(feature_state)
                        add_holdout[feature_to_drop] = holdout_steps
                        change_at['drop'] = True
                        break    
                    if len(max_candidates) > 0:
                        break
                    else:
                        change_at['drop'] = False
                        print(f'Failed to drop one of {len(group)} candidates')
                if len(test_feature_groups) > 0:
                    if change_at['drop'] == False:
                        print('NO FEATURES DROPPED')
                    self.fit_best(local_features = feature_state['curr_features'])
                else:
                    change_at['drop'] = False
                    print('No features to drop')
        if best_score_local > self.best_score + self.threshold:
            self.best_score = best_score_local
            self.best_features = feature_state['curr_features']
        print(f'Best score {self.eval_metric}: {self.best_score:0.4f} with features', self.best_features)
 
    def bw_select_tree(self, recursive = True):
        if len(self.feature_lists) == 0:    
            self.init_run()
        avg_feat_importances_sorted = self.cv_object.get_feature_importances_df().mean().sort_values()
        i = 0
        while i < len(self.best_features) - 5:
            feature_to_drop = avg_feat_importances_sorted.index[i]
            feature_is_dropped = self.bw_select(feature_to_drop)
            if feature_is_dropped:
                if recursive == True:
                    self.bw_select_tree()
                break
            else:
                i+=1  
    def fw_select_corr(self, win_corr, start_features = [], recursive = True, exclude_categorical = False):
        if len(self.dropped_features) == 0:
            feature_set_curr = set(self.cv_object.features)
            cat_features = feature_set_curr.difference(set(win_corr.index))
            if len(cat_features) > 0:
                if exclude_categorical == False: 
                    start_features = list(set(start_features) | cat_features)
                else:
                    feature_set_curr = feature_set_curr.difference(cat_features)
            self.dropped_features = feature_set_curr.difference(set(start_features))
            self.cv_object.set_features(start_features)
            if len(start_features) > 0:
                self.init_run(features = start_features)
        win_corrs_sorted = win_corr.loc[list(self.dropped_features)].abs().sort_values(ascending = False)
        i = 0
        while i < len(win_corrs_sorted):
            feature_to_add = win_corrs_sorted.index[i]
            feature_is_added = self.fw_select(feature_to_add)
            if feature_is_added:
                if recursive == True:
                    self.fw_select_corr(win_corr, start_features = self.best_features, 
                                        exclude_categorical = exclude_categorical)
                break
            else:
                i+=1
    def bw_select(self, feature_to_drop):
        test_features = [feat for feat in self.best_features if feat != feature_to_drop
                         and feat not in self.dropped_features]
        if test_features not in self.feature_lists:
            new_score = self.run_test(test_features)
            if new_score > self.best_score + self.threshold:
                self.drop_feature(feature_to_drop, new_score, test_features)
                return True
            else:
                print(f'Failed to drop feature {feature_to_drop}')
        else:
            print(f'Failed to add feature {feature_to_drop}, test feature set already considered.')
        return False
    def drop_feature(self, feature_to_drop, new_score, test_features):
        self.best_score = new_score
        self.best_features = set(test_features)
        self.dropped_features.add(feature_to_drop)
        print(f'Dropping feature {feature_to_drop}')
        print(f'New max auc: {new_score:0.4f}')
    def fw_select(self, feature_to_add):
        test_features = self.best_features + [feature_to_add]
        if test_features not in self.feature_lists:
            new_score = self.run_test(test_features)
            if new_score > self.best_score + self.threshold:
                self.add_feature(feature_to_add, new_score, test_features)
                return True
            else:
                print(f'Failed to add feature {feature_to_add}')
        else:
            print(f'Failed to add feature {feature_to_add}, test feature set already considered.')
        return False
    def add_feature(self, feature_to_add, new_score, test_features):
        self.best_score = new_score
        self.best_features = set(test_features)
        self.dropped_features.remove(feature_to_add)
        print(f'Adding feature {feature_to_add}')
        print(f'New max auc: {new_score:0.4f}')       
    def fit_best(self, local_features = []):
        if len(local_features) > 0:
            features_to_fit = local_features
        else:
            features_to_fit = self.best_features
        print('Fitting to features',features_to_fit)
        self.cv_object.set_features(features_to_fit)
        self.cv_object.run_cv()
    def init_run(self, features = []):
        if len(features) > 0:
            self.cv_object.set_features(features)
        self.feature_lists.append(self.cv_object.features)
        self.best_features = self.cv_object.features
        self.cv_object.run_cv()
        if self.eval_metric == 'auc':
            new_score = self.cv_object.get_auc()
        elif self.eval_metric == 'f1':
            new_score = self.cv_object.get_f1()
        self.scores.append(new_score)
        self.best_score = new_score
        print('Initial model {} score: {:0.4f}'.format(self.eval_metric, new_score))
    def run_test(self, test_features):
        print('Testing features:', test_features)
        self.feature_lists.append(test_features)
        self.cv_object.set_features(test_features)
        self.cv_object.run_cv()
        if self.eval_metric == 'auc':
            new_score = self.cv_object.get_auc()
        elif self.eval_metric == 'f1':
            new_score = self.cv_object.get_f1()
        self.scores.append(new_score)
        print(f'Test score = {new_score:0.4f}')
        return new_score
    def set_dropped_features(self, dropped_features):
        self.dropped_features = set(dropped_features)
        
class FeatureSelectorNN(FeatureSelector):
    def __init__(self, cv_object, **cv_args):
        super().__init__(cv_object)
        self.cv_args = cv_args
    def sw_select_cust(self, 
                       primary_selection_heirarchy = pd.Series(dtype = 'float64'), 
                       secondary_selection_heirarchy = pd.Series(dtype = 'float64'),
                       start_features = [], n_test_features = 3,
                       fw_select_type = 'primary', bw_select_type = 'primary', 
                       holdout_steps = 4, exclude_unlisted_features = False):
        # Check for correct input values
        select_types = ('primary', 'secondary', 'all', 'none')
        select_error_strings = ["Feature selection is based on series provided in method call.",
                                "If you want to use different heirarchies for forward and backward selection",
                                "pass a second series and set that selection type to 'secondary'.",
                                "Select 'none' on either type to perform just the other type."]
        select_error_string = ' '.join(select_error_strings)
        assert fw_select_type in select_types and bw_select_type in select_types, select_error_string
        # Initialize model and feature selection paradigm
        if fw_select_type in ('primary', 'secondary') or bw_select_type in ('primary', 'secondary'):
            primary_features, primary_selection_heirarchy = self.update_features(self.cv_object.features,
                                                                                 primary_selection_heirarchy,
                                                                                 exclude_unlisted_features)
            secondary_features = set()
            if len(secondary_selection_heirarchy) > 0:
                secondary_features, secondary_selection_heirarchy = self.update_features(self.cv_object.features,
                                                                                         secondary_selection_heirarchy,
                                                                                         exclude_unlisted_features)  
            all_features = primary_features | secondary_features
        else:
            all_features = set(self.cv_object.features)
        selection_score_map = {'primary': primary_selection_heirarchy,
                              'secondary': secondary_selection_heirarchy, 
                              'all': pd.Series(dtype = 'float64'),
                              'none': pd.Series(dtype = 'float64')}

        fw_selection_scores = selection_score_map[fw_select_type]
        bw_selection_scores = selection_score_map[bw_select_type]
        best_score_local = 0.0
        add_holdout = {}
        drop_holdout = {}
        feature_state = {'curr_features': start_features, # Current features being evaluated (list)
                         'free_features': all_features.difference(set(start_features)), # Features available (set)
                         # to add/drop in the next step
                         'to_add_step': True} # Next step 
        if len(start_features) > 0:
            if feature_state in self.feature_states:
                return f'Feature state with starting features {start_features} already considered'
            self.feature_states.append(feature_state)
            # Potential first CV run after object is initialized (if not starting from 0 features)
            if len(self.scores) == 0:
                self.init_run(features = start_features)
                best_score_local = self.best_score
            elif start_features in self.feature_lists:
                score_index = self.feature_lists.index(start_features)
                best_score_local = self.scores[score_index]
            else:
                best_score_local = self.run_test(start_features)  
            print(f'Running modified step-wise feature selector with {len(start_features)} starting features')
        else:
            self.cv_object.set_features(start_features)
        
        change_at = pd.Series({'add': True, 'drop': True})
        while change_at.any():
            if fw_select_type == 'none':
                change_at['add'] = False
            else:
                # Add feature step
                add_holdout = {key:value-1 for key,value in add_holdout.items() if value != 1}
                # If the method was just called or if a feature was added the feature_state is up-to-date
                if change_at['drop'] == True:
                    add_candidates = feature_state['free_features']
                # If not we need to update it
                else:
                    add_candidates = all_features.difference(set(feature_state['curr_features']) |
                                                             set(add_holdout.keys()))
                    feature_state['free_features'] = add_candidates
                    feature_state['to_add_step'] = True
                test_feature_groups = self.get_test_feature_groups(add_candidates, 
                                                              fw_selection_scores,
                                                              fw_select_type, 
                                                              n_test_features)
                for group in test_feature_groups:
                    test_scores = []
                    for add_candidate in group:
                        test_features = feature_state['curr_features'] + [add_candidate]
                        if test_features in self.feature_lists:
                            score_index = self.feature_lists.index(test_features)
                            print('Feature set already tested:', test_features)
                            test_score = self.scores[score_index]  
                            print(f'Test score = {test_score:0.4f}')
                        else:
                            test_score = self.run_test(test_features)
                        test_scores.append(test_score)
                    sorted_scores = sorted(zip(test_scores, group), reverse = True)
                    max_candidates = [tup for tup in sorted_scores if tup[0] > best_score_local + self.threshold]
                    while len(max_candidates) > 0:
                        # Create resulting feature state if the add candidate is selected
                        # If this feature state has been considered previously, do not select the add candidate
                        feature_to_add = max_candidates[0][1]
                        test_features = feature_state['curr_features'] + [feature_to_add]
                        test_feature_state = feature_state.copy()
                        test_feature_state['curr_features'] = test_features
                        holdout_features = {feature for feature in drop_holdout if drop_holdout[feature] > 1}
                        test_feature_state['free_features'] = set(feature_state['curr_features']).difference(holdout_features)
                        test_feature_state['to_add_step'] = False
                        if test_feature_state in self.feature_states:
                            max_candidates.remove(max_candidates[0])
                            print('Feature state already considered:',test_feature_state)
                            continue
                        print(f'Adding feature {feature_to_add}')
                        best_score_local = max_candidates[0][0]
                        feature_state = test_feature_state
                        self.feature_states.append(feature_state)
                        drop_holdout[feature_to_add] = holdout_steps
                        change_at['add'] = True
                        break
                    if len(max_candidates) > 0:
                        break
                    else:
                        change_at['add'] = False
                        print(f'Failed to add one of {len(group)} candidates')
                # Refit cv object to best features if any other feature sets were tested
                if len(test_feature_groups) > 0:
                    if change_at['add'] == False:
                        print('NO FEATURES ADDED')
                    self.fit_best(local_features = feature_state['curr_features'])
                else:
                    change_at['add'] = False
                    print('No features to add')
                # End process if no changes were made in the past drop/add cycle
                if not change_at.any():
                    break
            
            # Drop feature step
            if bw_select_type == 'none':
                change_at['drop'] = False
            else:
                drop_holdout = {key:value-1 for key,value in drop_holdout.items() if value != 1}
                # If a feature was added in the previous step the feature state is up-to-date
                if change_at['add'] == True:
                    drop_candidates = feature_state['free_features']
                # If not we must update it
                else:
                    drop_candidates = set(feature_state['curr_features']).difference(set(drop_holdout.keys()))
                    feature_state['free_features'] = drop_candidates
                    feature_state['to_add_step'] = False
                test_feature_groups = self.get_test_feature_groups(drop_candidates, 
                                                              bw_selection_scores,
                                                              bw_select_type, 
                                                              n_test_features)
                for group in test_feature_groups:
                    test_scores = []
                    for drop_candidate in group:
                        test_features = feature_state['curr_features'].copy()
                        test_features.remove(drop_candidate)
                        if test_features in self.feature_lists:
                            score_index = self.feature_lists.index(test_features)
                            print('Feature set already tested:', test_features)
                            test_score = self.scores[score_index]  
                            print(f'Test score = {test_score:0.4f}')                        
                        else:
                            test_score = self.run_test(test_features)
                        test_scores.append(test_score)
                    sorted_scores = sorted(zip(test_scores, group), reverse = True)
                    max_candidates = [tup for tup in sorted_scores if tup[0] > best_score_local + self.threshold]
                    while len(max_candidates) > 0:
                        feature_to_drop = max_candidates[0][1]
                        test_features = feature_state['curr_features'].copy()
                        test_features.remove(feature_to_drop)
                        test_feature_state = feature_state.copy()
                        test_feature_state['curr_features'] = test_features
                        holdout_features = {feature for feature in add_holdout if add_holdout[feature] > 1}
                        test_feature_state['free_features'] = all_features.difference(feature_state['curr_features'] |
                                                                                      holdout_features)
                        test_feature_state['to_add_step'] = True
                        if test_feature_state in self.feature_states:
                            max_candidates.remove(max_candidates[0])
                            print('Feature state already considered:',test_feature_state)
                            continue
                        print(f'Dropping feature {feature_to_drop}')
                        best_score_local = max_candidates[0][0]
                        feature_state = test_feature_state
                        self.feature_states.append(feature_state)
                        add_holdout[feature_to_drop] = holdout_steps
                        change_at['drop'] = True
                        break    
                    if len(max_candidates) > 0:
                        break
                    else:
                        change_at['drop'] = False
                        print(f'Failed to drop one of {len(group)} candidates')
                if len(test_feature_groups) > 0:
                    if change_at['drop'] == False:
                        print('NO FEATURES DROPPED')
                    self.fit_best(local_features = feature_state['curr_features'])
                else:
                    change_at['drop'] = False
                    print('No features to drop')
        if best_score_local > self.best_score + self.threshold:
            self.best_score = best_score_local
            self.best_features = feature_state['curr_features']
        print(f'Best score {self.eval_metric}: {self.best_score:0.4f} with features', self.best_features)
    
    def fit_best(self, local_features = []):
        if len(local_features) > 0:
            features_to_fit = local_features
        else:
            features_to_fit = self.best_features
        print('Fitting to features',features_to_fit)
        self.cv_object.set_features(features_to_fit)
        model_params = {'n_features': len(features_to_fit)}
        self.cv_object.run_cv(model_params = model_params,
                              **self.cv_args)
    
    def init_run(self, features = []):
        if len(features) > 0:
            self.cv_object.set_features(features)
        self.feature_lists.append(self.cv_object.features)
        self.best_features = self.cv_object.features
        model_params = {'n_features': len(self.cv_object.features)}
        self.cv_object.run_cv(model_params = model_params,
                              **self.cv_args)
        if self.eval_metric == 'auc':
            new_score = self.cv_object.get_auc()
        elif self.eval_metric == 'f1':
            new_score = self.cv_object.get_f1()
        self.scores.append(new_score)
        self.best_score = new_score
        print('Initial model {} score: {:0.4f}'.format(self.eval_metric, new_score))
    
    def run_test(self, test_features):
        print('Testing features:', test_features)
        self.feature_lists.append(test_features)
        self.cv_object.set_features(test_features)
        model_params = {'n_features': len(test_features)}
        self.cv_object.run_cv(model_params = model_params,
                              **self.cv_args)
        if self.eval_metric == 'auc':
            new_score = self.cv_object.get_auc()
        elif self.eval_metric == 'f1':
            new_score = self.cv_object.get_f1()
        self.scores.append(new_score)
        print(f'Test score = {new_score:0.4f}')
        return new_score
    
    def update_features(self, all_features, feature_heirarchy, exclude_unlisted_features):
        unlisted_features = list(set(all_features).difference(set(feature_heirarchy.index)))
        if len(unlisted_features) > 0:
            if not exclude_unlisted_features:
                avg_magnitude = feature_heirarchy.abs().mean()
                unlisted_series = pd.Series([avg_magnitude]*len(unlisted_features), index = unlisted_features)
                feature_heirarchy = pd.concat((feature_heirarchy, unlisted_series))
            else:
                all_features = set(all_features).difference(unlisted_features)
        return set(all_features), feature_heirarchy
    
    def get_test_feature_groups(self, candidates, selection_scores, select_type, n_test_features):
        if select_type in ('primary', 'secondary'):
            candidates_sorted = selection_scores.loc[list(candidates)].abs().sort_values(ascending = False).index.tolist()
            test_feature_groups = [tuple(candidates_sorted[i*n_test_features:(i+1)*n_test_features])
                                  for i in range(len(candidates_sorted) // n_test_features)]
            remainder = len(candidates_sorted) % n_test_features
            if remainder > 0:
                test_feature_groups.append(tuple(candidates_sorted[-1*remainder:]))    
        elif select_type == 'all':
            test_feature_groups = [sorted(candidates)]
        return test_feature_groups

        
class GridSearchLOWOCV():
    def __init__(self, cv_object, param_grid, eval_metric = 'auc'):
        self.cv_object = cv_object
        self.param_grid = param_grid
        param_list = list(product(*self.param_grid.values()))
        self.param_sets = [dict(zip(param_grid.keys(),param_set)) for param_set in param_list]
        self.eval_metric = eval_metric
        self.auc_scores = []
        self.best_auc = 0.0
        self.f1_scores = []
        self.best_f1 = 0.0
        self.best_params = {}
    def run_gs(self, verbose = True):
        print('Fitting {} folds for {} candidates'.format(self.cv_object.num_weeks, len(self.param_sets)))
        for i, param_set in enumerate(self.param_sets):
            self.cv_object.set_params(param_set)
            self.cv_object.run_cv()
            test_auc = self.cv_object.get_auc()
            self.auc_scores.append(test_auc)
            if test_auc > self.best_auc:
                self.best_auc = test_auc
                if self.eval_metric == 'auc':
                    self.best_params = param_set
            test_f1 = self.cv_object.get_f1()
            self.f1_scores.append(test_f1)
            if test_f1 > self.best_f1:
                self.best_f1 = test_f1
                if self.eval_metric == 'f1':
                    self.best_params = param_set
            if verbose:
                param_string = '; '.join([f'{key}: {param_set[key]}' for key in param_set.keys()])
                print(f'Candidate {i+1} ' + param_string + f'; scores--- auc = {test_auc:0.4f} f1 = {test_f1:0.4f}')
    def param_set_scores_df(self, sort = True):
        df = pd.DataFrame(self.param_sets)
        df['auc'] = self.auc_scores
        df['f1'] = self.f1_scores
        if sort == True:
            df = df.sort_values(self.eval_metric, ascending = False)
        return df
    

    
    
