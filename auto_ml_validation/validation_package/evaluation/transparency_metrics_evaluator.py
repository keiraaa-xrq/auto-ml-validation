from typing import *
import pandas as pd
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime.lime_tabular import LimeTabularExplainer
from lime import submodular_pick
from sklearn.base import BaseEstimator



class TransparencyMetricsEvaluator:
    def __init__(self, model, X, y, class_name_list=None):
        self.model = model
        self.X = X
        self.y = y
        self.class_name_list = class_name_list

        self.lime_explainer = LimeTabularExplainer(X.values, feature_names=X.columns, discretize_continuous=True,
                                                   class_names=[class_name_list] if class_name_list is not None else None)
        self.shap_explainer = shap.Explainer(model.predict, X)
    
    def lime_interpretability(self):
        i = np.random.randint(0, self.X.shape[0])
        predict_fn = lambda x: self.model.predict_proba(x).astype(float)
        local_lime_fig = self.lime_explainer.explain_instance(self.X.iloc[i], predict_fn, num_features=len(self.X.columns))
        # Generate global LIME plot
        sp_obj = submodular_pick.SubmodularPick(self.lime_explainer,
                                                self.X.values,
                                                predict_fn,
                                                sample_size=1,
                                                num_features=len(self.X.columns),
                                                num_exps_desired=1)
        global_lime_fig = sp_obj.sp_explanations[0]
        # Print local feature importances
        print('{:<30} {:<10}'.format('Feature', 'Weight'))
        print('-' * 40)
        for feature, weight in local_lime_fig.as_list():
            print('{:<30} {:<10.2f}'.format(feature, weight))
        # Print global feature importances
        print('{:<30} {:<10}'.format('Feature', 'Weight'))
        print('-' * 40)
        for feature, weight in global_lime_fig.as_map()[0]:
            print('{}: {:.2f}'.format(self.X.columns[feature], weight))
        # Show plots
        local_lime_fig.show_in_notebook(show_table=True)
        global_lime_fig.show_in_notebook(show_table=True)
        return local_lime_fig, global_lime_fig, local_lime_fig.as_list(), global_lime_fig.as_map()[0]

    def shap_interpretability(self):
        shap_values = self.shap_explainer(self.X)
        i = np.random.randint(0, self.X.shape[0])
        local_shap_fig = shap.plots.waterfall(shap_values[i])
        global_shap_fig = shap.plots.bar(shap_values, show=True, max_display=12)
        # Compute and print local feature importances
        feature_names = self.X.columns.tolist()
        feature_importance = shap_values.values[i]
        local_impt_map = {imp: name for imp, name in zip(feature_importance, feature_names)}
        local_impt_map = sorted(local_impt_map.items(), reverse=True)
        print("#" * 50)
        print("Local feature importances:")
        for imp, name in local_impt_map:
            print(f"Feature {name}: importance = {imp:.3f}")
        #Compute and print feature importances
        feature_importance = np.abs(shap_values.values).mean(axis=0)
        global_impt_map = {imp: name for imp, name in zip(feature_importance, feature_names)}
        global_impt_map = sorted(global_impt_map.items(),key=lambda x: abs(x[0]), reverse=True)
        print("#" * 50)
        print("Global feature importances:")
        for imp, name in global_impt_map:
            print(f"Feature {name}: importance = {imp:.3f}")
        return local_shap_fig, global_shap_fig, local_impt_map, global_impt_map


    # def get_partial_dependence(self, feature: Optional[str] = None):
    #     if feature is None:
    #         feature = self.X.columns
    #     n_cols = 5
    #     return PartialDependenceDisplay.from_estimator(self.model, self.X, features=feature, n_cols=n_cols)
        
    def cal_gini(self):
        """Calculate GINI index for class and attributes"""  
        def _gini_impurity (value_counts):
            n = value_counts.sum()
            p_sum = 0
            for key in value_counts.keys():
                p_sum = p_sum  +  (value_counts[key] / n ) * (value_counts[key] / n ) 
            gini = 1 - p_sum
            return gini
        
        def _gini_attribute(attribute_name):
            attribute_values = self.X[attribute_name].value_counts()
            gini_A = 0 
            for key in attribute_values.keys():
                df_k = pd.DataFrame(self.y)[self.X[attribute_name] == key].value_counts()
                n_k = attribute_values[key]
                n = self.X.shape[0]
                gini_A = gini_A + (( n_k / n) * _gini_impurity(df_k))
            return gini_A
        
        result = {}
        for key in (self.X).columns[0:20]: # for testing
            result[key] = _gini_attribute(key)
        
        return result
