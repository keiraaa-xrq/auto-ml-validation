import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectFromModel, RFECV, VarianceThreshold, SelectKBest, mutual_info_regression
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LassoCV, LogisticRegression
from sklearn.model_selection import StratifiedKFold
from typing import Optional, List, Union, Literal

#Add
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from mlxtend.feature_selection import ExhaustiveFeatureSelector as EFS


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
                            "intrinsic": This refers to algorithms such as Random Forest that perform automatic feature selection and Recursive methods. This includes Embedded Methods
                            "auto": Use a combination of above.
                            (default: "auto")
            - verbose: verbosity level (int; default: 1)
        Attributes:
            - feats_selected_: list of good features (to select via pandas DataFrame columns)
            - num_features: remaining count of features after feature selection
            - predictors_original : original columns of X when calling fit
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
        
    def generate_best_feats(self, predictors: pd.DataFrame, target, model, scoring_method) -> pd.DataFrame:
        """
        Selects best predictive features given the data and targets.
        Inputs:
            - predictors: pandas DataFrame with n data points and p features
            - target: n dimensional array with targets corresponding to the data points in df
        Returns:
            - feats_selected_: list of column names selected
            - num_features: count of column names selected
        """
        feats_selected_, num_features = [], 0
        if self.keep is None:
            self.keep = []
        else:
            self._verbose_print('Retaining Features: ' + self.keep)
        predictors_original = predictors.copy(deep = True)
        predictors = predictors.drop(self.keep, axis = 1)
        
        # Data Check
        if predictors.shape[0] <= 1:
            raise ValueError("Number of samples too small, n_samples = {}".format(predictors.shape[0]))
        if not (len(predictors) == len(target)):
            raise ValueError("Predictors dataframe and target dimension mismatch.")
        if not all(np.issubdtype(predictors[col].dtype, np.number) for col in predictors.columns):
            raise TypeError("Object data found. Data needs to be processed to float or integer.")
        if any(predictors[col].nunique() == len(predictors) for col in predictors.columns):
            raise ValueError("Unique Identifier found in dataset.")
        
        if self.method == 'intrinsic':
            feats_selected_, num_features = self._feature_select_intrinsic(predictors, target)
        elif self.method == 'greedy':
            feats_selected_, num_features = self._feature_select_greedy(predictors, target, model, scoring_method)
        elif self.method == 'filter':
             feats_selected_, num_features = self._feature_select_filter(predictors, target)
        else: # auto
            feats_selected_, num_features = self._feature_select_greedy(predictors, target, model, scoring_method)

                
        self._verbose_print('Features Selection Completed.')
                
        if not self.keep:
            return feats_selected_, num_features
        else:
            return feats_selected_ + self.keep, len(feats_selected_ + self.keep)


    
    #########################################################################################################
    # Feature Selection Methods Implementation
    #########################################################################################################
    def _feature_select_intrinsic(self, X: pd.DataFrame, y):
        """
        Intrinsic methods for feature selection
        Inputs:
            - X: pandas dataframe or numpy array with original features
            - y: pandas dataframe or numpy array with targets for all n_datapoints
        """
        self._verbose_print('Beginning Intrinsic Feature Selection Method.')
        feats_selected_, num_features = [], 0
        
        def _lasso_selected_feature(X, y):
            skf = StratifiedKFold(n_splits=10)
            lasso = LassoCV(cv=skf, random_state=42).fit(X, y)
            feats_selected_ = list(X.columns[np.where(lasso.coef_!=0)[0]])
            num_features = len(feats_selected_)
            return feats_selected_, num_features
        
        def _log_reg_coefficients(X, y):
            sel_ = SelectFromModel(LogisticRegression(C=1000, penalty='l2', max_iter=500, random_state=10))
            sel_.fit(X, y)
            feats_selected_ = X.columns[(sel_.get_support())]
            num_features = len(feats_selected_)
            return feats_selected_, num_features

        def _rf_selected_features(X, y):
            hyper = {'min_samples_leaf':80, 'max_features':0.5, 'max_depth':15}
            sel_ = SelectFromModel(RandomForestClassifier(n_estimators=50, min_samples_leaf=hyper['min_samples_leaf'],
                                 max_features=hyper['max_features'],
                                 max_depth=hyper['max_depth'],
                                 oob_score=True,
                                 n_jobs=-1))
            sel_.fit(X, y)
            feats_selected_ = X.columns[(sel_.get_support())]
            return feats_selected_, len(feats_selected_)
        
        feats_selected_, num_features = _rf_selected_features(X, y)
        
        return feats_selected_, num_features
            
    def _feature_select_greedy(self, X: pd.DataFrame, y, model, scoring_method):
        """
        Greedy methods for feature selection
        Inputs:
            - X: pandas dataframe or numpy array with original features
            - y: pandas dataframe or numpy array with targets for all n_datapoints
        """
        self._verbose_print('Beginning Greedy Feature Selection Method.')
        
        #Possible Scoring : accuracy, f1, precision, recall, roc_auc
        def _forward_selected_feature(X, y, model, scoring_method):
            sfs1 = SFS(model,
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
        
        def _backward_selected_feature(X, y, model, scoring_method):
            sfs1 = SFS(model,
                       k_features= (1, len(X.columns)),
                       forward=False,
                       floating=False,
                       verbose=2,
                       scoring=scoring_method,
                       cv=3)
            sfs1 = sfs1.fit(X, y)
            feats_selected = list(sfs1.k_feature_names_)
            self._verbose_print(sfs1.k_score_)
            return feats_selected, len(feats_selected)
        
        def _exhausive_selected_feature(X, y, model, scoring_method):
            efs1 = EFS(model, 
                       min_features=1,
                       max_features=len(X.columns),
                       scoring=scoring_method,
                       cv=3)
            
            efs1 = efs1.fit(X, y)
            feats_selected = list(efs1.best_feature_names_)
            self._verbose_print(efs1.best_score_)
            return feats_selected, len(feats_selected)
        
        def _rfecv_selected_features(X, y, cross_val = 5):
            rfc = RandomForestClassifier(max_depth=8, random_state=0)
            clf = RFECV(rfc, step = 1, cv = cross_val)
            clf.fit(X, y)
            feats_selected_ = X.columns[clf.support_]
            num_features = clf.n_features_
            return feats_selected_, num_features
        
        #return _forward_selected_feature(X, y, model, scoring_method)
        return _backward_selected_feature(X, y, model, scoring_method)
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
    def get_n_features(self):
        return len(self.feats_selected_)
    
    @property
    def get_features_selected(self):
        return self.feats_selected_
    
    
if __name__ == "__main__":
    data = pd.read_csv('data/stage_2/UCI_Credit_Card.csv')
    X_dat = data.drop('default.payment.next.month', axis = 1)
    y_dat = data['default.payment.next.month']
    
    ft = AutoFeatSelection(task = "binary_classification", keep = None, method = 'auto')
    feats_selected_, num_features = ft.generate_best_feats(X_dat, y_dat)
    print(feats_selected_, num_features)
        
