
import numpy as np
import pandas as pd
import os
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestClassifier
from typing import Optional, List
from  RF_with_hyperparameters1 import  get_eval_para

class Random_forest(object):
    '''This model compute random forest classification'''
    
    def __init__(self, data: Optional[str],
                 mode: Optional[str] = 'classification'):

        self.data = data
        self.mode = mode

        
        self.estimators = [100, 150, 200]
        self.min_samples_split = [2]
        self.min_samples_leaf = [1]
        self.max_features = [2, 'log2']
        self.max_depth = [10, 15, 25]
        
        self.best_model = None
        
    def fit(self, X_train, Y_train, nsplits):
        
        
        '''Train data based on hyperparameters
        :param X_train: Sampled x_train
        :param Y_train: Sampled Y_train'''
        
        
        cv_inner = KFold(n_splits=nsplits)
        
        parameter_grid = {
            'n_estimators': self.estimators,
            'min_samples_split': self.min_samples_split,
            'min_samples_leaf': self.min_samples_leaf,
            'max_features': self.max_features,
            'max_depth': self.max_depth
        }
        clf = GridSearchCV(RandomForestClassifier(), param_grid=parameter_grid, cv=cv_inner)
        clf.fit(X_train, Y_train)
        best_parameters = clf.best_params_
        self.best_model = RandomForestClassifier(**best_parameters, random_state=42)
        self.best_model.fit(X_train, Y_train)
        
        return self.best_model
    
    def predict(self, X_test, Y_test):
        
        if self.best_model is None:
            raise ValueError('Model is not computed')
        
        y_preds = self.best_model.predict(X_test)
        y_prob = self.best_model.predict_proba(X_test)[:,1]
        
        # Calculation evaluation metrics
        ACC_tst, TPR_tst, SPE_tst, F1_score_tst= get_eval_para(Y_test, y_preds)
        
        
        return  ACC_tst, TPR_tst, SPE_tst, F1_score_tst , y_prob
        
       
        
    
    
    