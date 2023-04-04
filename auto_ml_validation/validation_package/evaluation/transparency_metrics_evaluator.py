import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
from lime import submodular_pick
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from sklearn.linear_model import LogisticRegression

import numpy as np
import matplotlib.pyplot as plt
import shap
import lime
from lime import submodular_pick
from lime.lime_tabular import LimeTabularExplainer
import pandas as pd
from sklearn.linear_model import LogisticRegression

class TransparencyMetricsEvaluator:
    def __init__(self, model, X, class_name_list=None):
        """
        Initializes the TransparencyMetricsEvaluator object.

        Args:
            model (object): The machine learning model to evaluate interpretability for.
            X (pandas.DataFrame): The input data for the machine learning model.
            class_name_list (list, optional): A list of class names. Defaults to None.
        """
        sample_X = X.sample(n=int(X.shape[0]/100))

        self.model = model
        self.X = sample_X
        self.class_name_list = class_name_list


        
        # Initialize the LIME explainer
        self.lime_explainer = LimeTabularExplainer(
            sample_X.values,
            feature_names=sample_X.columns,
            discretize_continuous=False,
            class_names=[class_name_list] if class_name_list is not None else None
        )
        
        # Initialize the SHAP explainer
        self.shap_explainer = shap.Explainer(model.predict, sample_X)
    
    def lime_interpretability(self):
        """
        Calculates LIME interpretability metrics.

        Returns:
            tuple: A tuple containing local and global LIME plots, and local and global LIME feature importances.
        """
        i = np.random.randint(0, self.X.shape[0])
        predict_fn = lambda x: self.model.predict_proba(x).astype(float)
        local_lime_fig = self.lime_explainer.explain_instance(self.X.iloc[i], predict_fn, num_features=10).as_pyplot_figure()
        local_lime_fig.gca().set_title('')

        sp_obj = submodular_pick.SubmodularPick(self.lime_explainer, self.X.values, predict_fn, sample_size=5, num_features=10, num_exps_desired=1).sp_explanations[0]

        
        if 0 in sp_obj.as_map():
            global_lime_fig = sp_obj.as_pyplot_figure(label = 0)
        else:
            global_lime_fig = sp_obj.as_pyplot_figure(label = 1)
 

        global_lime_fig.gca().set_title('')

        local_text = self.lime_explainer.explain_instance(self.X.iloc[i], predict_fn, num_features=len(self.X.columns))

        global_text = submodular_pick.SubmodularPick(self.lime_explainer, self.X.values, predict_fn, sample_size=10, num_features=len(self.X.columns), num_exps_desired=1).sp_explanations[0]


        if 0 in sp_obj.as_map():
            global_res = [(self.X.columns[feature], weight) for feature, weight in global_text.as_map()[0]]
        else:
            global_res = [(self.X.columns[feature], weight) for feature, weight in global_text.as_map()[1]]        
        
        local_res = [(feature, weight) for feature, weight in local_text.as_list()]

        return local_lime_fig, global_lime_fig, local_res,  global_res

    def shap_interpretability(self):
        """
        Calculates SHAP interpretability metrics.

        Returns:
            tuple: A tuple containing local and global SHAP plots, and local and global SHAP feature importances.
        """
        # Get SHAP values
        shap_values = self.shap_explainer(self.X)

        # Get random index
        idx = np.random.randint(0, self.X.shape[0])

        # Get feature names and importance for the selected index
        feature_names = self.X.columns.tolist()
        local_feature_importance = shap_values.values[idx]
        local_impt_map = sorted(zip(local_feature_importance, feature_names), key=lambda x: abs(x[0]), reverse=True)

        # Get feature importance across all data points
        global_feature_importance = shap_values.values.mean(axis=0)
        global_impt_map = sorted(zip(global_feature_importance, feature_names), key=lambda x: abs(x[0]), reverse=True)

        # Round feature importance values and swap the order of the tuples
        local_impt_map = [(label, num) for num, label in local_impt_map]
        global_impt_map = [(label, num) for num, label in global_impt_map]

        # Generate local and global plots
        plt.figure()
        shap.plots.bar(shap_values[idx], show=False)
        local_plot = plt.gcf()
        plt.figure()
        shap.plots.bar(shap_values, show=False)
        global_plot = plt.gcf()
        
        return local_plot, global_plot, local_impt_map, global_impt_map


#Test
'''
X = pd.read_csv('processed_X_for_trans.csv').iloc[0:100]
y = pd.read_csv('y_train.csv').iloc[0:100]
lr = LogisticRegression().fit(X, y)
evaluator = TransparencyMetricsEvaluator(lr, X)
local_lime_fig, global_lime_fig, local_text_lime, global_text_lime = evaluator.lime_interpretability()
local_shap_fig, global_shap_fig, local_text_shap, global_text_shap = evaluator.shap_interpretability()

local_shap_fig.savefig('local_shap.png',  bbox_inches='tight')
global_shap_fig.savefig('global_shap.png', bbox_inches='tight')
global_lime_fig.savefig('global_lime.png', bbox_inches='tight')
local_lime_fig.savefig('local_lime.png', bbox_inches='tight')
'''


