import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel, RFECV
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from typing_extensions import (Literal)
from typing import Optional, List, Union

#Add
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS

#Filter
from sklearn.feature_selection import VarianceThreshold
from sklearn.feature_selection import SelectKBest, mutual_info_regression



class AutoFeatSelection(BaseEstimator):
    
    def __init__(
        self,
        task: Literal["binary_classification", "multi_classification", "regression"] = "binary_classification",
        keep:  Union[list, str, None]= None,
        method: Literal["filter", "greedy", "intrinsic", "auto"] = "auto",
        verbose: Optional[int] = 1,
        ):
        
        """
        Cross-Validated Automated Feature Selection
        Inputs:
            - task: str, "binary_classification" or "multi_classification" or "regression" (default: "binary_classification")
            - keep: list of features that should be kept, this is useful to keep business context features (requirements)
            - method: str, "filter": This refers to statistic/data-based method e.g. Correlation/Statistical Test (Information Gain, Fisher Score)
                            "greedy": This refers to wrapper methods, forward selection/backward elimination/exhaustive feature selection.
                            "intrisic": This refers to algorithms such as Random Forest that perform automatic feature selection and Recursive methods. This includes Embedded Methods
                            "auto": Use a combination of above.
                            (default: "all")
            - verbose: verbosity level (int; default: 1)
        Attributes:
            - feats_selected_: list of good features (to select via pandas DataFrame columns)
            - num_features: remaining count of features after feature selection
            - original_columns_: original columns of X when calling fit
            - output_df_: dataframe of predictors and target variable
        """
        
        self.task = task
        self.keep = keep
        self.method = method
        self.verbose = verbose
        
    def _verbose_print(self, msg):
        if self.verbose > 0:
            print(msg)
    
    def _check_class_imbalance(self, target) -> bool:
        # TODO
        return True
        
    def generate_best_feats(self, predictors: pd.DataFrame, target) -> pd.DataFrame:
        if predictors.shape[0] <= 1:
            raise ValueError("Number of samples too small, n_samples = {}".format(predictors.shape[0]))
        if self.method == 'intrisic':
            feats_selected_, num_features = feature_select_instrisic(predictors, target)
        elif self.method == 'greedy':
            feats_selected_, num_features = self._feature_select_greedy(predictors, target)
        elif self.method == 'filter':
             feats_selected_, num_features = self._feature_select_filter(predictors, target)
             
        return feats_selected_, num_features

    
    #########################################################################################################
    # Feature Selection Methods Implementation
    #########################################################################################################
    def _feature_select_instrisic(X: pd.DataFrame, y):
        """
        Intrisic methods for feature selection
        Inputs:
            - X: pandas dataframe or numpy array with original features
            - y: pandas dataframe or numpy array with targets for all n_datapoints
        """
        _verbose_print('Beginning Instrisic Feature Selection Method.')
        feats_selected_, num_features = [], 0
        
        
        def _dt_selected_features(X, y):
            clf = DecisionTreeClassifier()
            trans = SelectFromModel(clf, threshold='median')
            X_trans = trans.fit_transform(X, y)
            feats_selected_ = X.iloc[:,1:].columns[trans.get_support()].values
            return feats_selected_, X_trans.shape[1]
        
        def _rfecv_selected_features(X, y, cross_val = 5):
            rfc = RandomForestClassifier(max_depth=8, random_state=0)
            clf = RFECV(rfc, step = 1, cv = cross_val)
            clf.fit(X, y)
            feats_selected_ = X.columns[clf.support_]
            num_features = clf.n_features_
            return feats_selected_, num_features
            
    
        feats_selected_, num_features = _dt_selected_features(X, y)
        return feats_selected_, num_features
            
    def _feature_select_greedy(self, X: pd.DataFrame, y):
        """
        Greedy methods for feature selection
        Inputs:
            - X: pandas dataframe or numpy array with original features
            - y: pandas dataframe or numpy array with targets for all n_datapoints
        """
        self._verbose_print('Beginning Greedy Feature Selection Method.')
        
        scoring_method = 'accuracy'

        #Possible Scoring : accuracy, f1, precision, recall, roc_auc
        def _forward_selected_feature(X, y):
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            sfs1 = SFS(clf,
                       k_features= (1, len(X.columns)),
                       forward=True,
                       floating=False,
                       verbose=2,
                       scoring=scoring_method,
                       cv=5)
            sfs1 = sfs1.fit(X, y)
            feats_selected = list(sfs1.k_feature_names_)
            self._verbose_print(sfs1.k_score_)
            return feats_selected, len(feats_selected)
        
        def _backward_selected_feature(X, y):
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            sfs1 = SFS(clf,
                       k_features= (1, len(X.columns)),
                       forward=False,
                       floating=False,
                       verbose=2,
                       scoring=scoring_method,
                       cv=5)
            sfs1 = sfs1.fit(X, y)
            feats_selected = list(sfs1.k_feature_names_)
            self._verbose_print(sfs1.k_score_)
            return feats_selected, len(feats_selected)
        
        def _exhausive_selected_feature(X, y):
            clf = RandomForestClassifier(n_estimators=100, n_jobs=-1)
            efs1 = EFS(clf, 
                       min_features=1,
                       max_features=len(X.columns),
                       scoring=scoring_method,
                       cv=3)
            
            efs1 = efs1.fit(X, y)
            feats_selected = list(efs1.best_feature_names_)
            self._verbose_print(efs1.best_score_)
            return feats_selected, len(feats_selected)
        
        
        return _forward_selected_feature(X, y)
        #return _backward_selected_feature(X, y)
        #return  _exhausive_selected_feature(X, y)
    
    
    def _feature_select_filter(self, X: pd.DataFrame, y):
        """
        Filter methods for feature selection
        Inputs:
            - X: pandas dataframe or numpy array with original features
            - y: pandas dataframe or numpy array with targets for all n_datapoints
        """
        self._verbose_print('Beginning Filter Feature Selection Method.')
        
        def _common_filter(X, y):
            #Remove Quasi Feature and Constant features 
            selector = VarianceThreshold(threshold=0.01)
            selector.fit(X)
            X_filtered = X[X.columns[selector.get_support(indices=True)]]
            #Remove Duplicate Columns
            X_filtered_T = X_filtered.T
            X_filtered = X_filtered_T.drop_duplicates(keep='first').T
            #Remove Correlated Features
            correlated_features = set()
            correlation_matrix = X_filtered.corr()
            for i in range(len(correlation_matrix.columns)):
                for j in range(i):
                    if abs(correlation_matrix.iloc[i, j]) > 0.8:
                        colname = correlation_matrix.columns[i]
                        correlated_features.add(colname)
            X_filtered.drop(correlated_features, axis = 1, inplace = True)
            feats_selected = X_filtered.columns.tolist()
            return feats_selected, len(feats_selected)
        
        return _common_filter(X, y)

        
        
    @property
    def n_features(self):
        return len(self.feats_selected_)