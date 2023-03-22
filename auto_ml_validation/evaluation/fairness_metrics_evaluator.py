import pandas as pd
import numpy as np
import aequitas
from aequitas.preprocessing import preprocess_input_df
from aequitas.plotting import Plot
from aequitas.group import Group
import warnings
warnings.filterwarnings("ignore")
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from veritastool.util.utility import test_function_cs
from veritastool.model import ModelContainer
from veritastool.fairness import CreditScoring

class FairnessMetricsEvaluator:
    def __init__(self,
                train = None,
                test = None,
                cat_var_list = None,
                model = None):
        self.train = train
        self.test = test
        self.cat_var_list = cat_var_list
        self.model = model

    def data_preprocess(self, 
                       score_col_name: str,
                       label_col_name: str)-> tuple[pd.DataFrame,
                                                    pd.DataFrame]:
        # double-check that categorical columns are of type 'string'
        for col in self.train.columns:
            if type(col) == object:
                self.train['col'] = self.train['col'].astype(str)
        # double-check that categorical columns are of type 'string'
        for col in self.test.columns:
            if type(col) == object:
                self.test['col'] = self.test['col'].astype(str)

        # preprocessing using aequitas
        train_df, _ = preprocess_input_df(self.train)
        test_df, _ = preprocess_input_df(self.test)

        return train_df, test_df

    def plot_group_metric(self, 
                          data: pd.DataFrame,
                          group_metric:str, 
                          ax_lim=None,
                          min_group_size = None):
        g = Group()
        xtab, _ = g.get_crosstabs(data)
        aqp = Plot()
        plot = aqp.plot_group_metric(xtab, 
                                     group_metric,
                                     ax_lim = ax_lim,
                                     min_group_size = min_group_size)
        return plot
    
    def fairness_visualization(self, 
                                X_train: pd.DataFrame,
                                X_test: pd.DataFrame, 
                                y_test: pd.DataFrame, 
                                y_pred: pd.DataFrame, 
                                y_train: pd.DataFrame, 
                                y_prob: pd.DataFrame, 
                                prot_cat_var_value: dict):
        """
        X_train : All the feature values in training set.
        X_test : All the feature values in testing set.
        y_true : Ground truth target values.
        y_pred : Predicted targets as returned by classifier.
        y_train : Ground truth for training data.
        y_prob : Predicted probabilities as returned by classifier. 
        cat_var_list: list of names for all the categorical variables 
        prot_cat_var_value: a python dict where the key is string and value is a list of string 
        model : an sklearn pipeline object 
        """

        test_function_cs()

        # pass in the data and model info
        y_true = np.array(y_test)
        y_pred = np.array(y_pred)
        y_train = np.array(y_train)
        p_var = self.cat_var_list
        p_grp = prot_cat_var_value
        x_train = X_train
        x_test = X_test
        model_object = self.model
        # set default to credit 
        model_type = 'credit'
        y_prob = y_prob

        container = ModelContainer(y_true = y_true, y_train = y_train, p_var = p_var, p_grp = p_grp, 
        x_train = x_train,  x_test = x_test, model_object = model_object, model_type  = model_type,
        y_pred= y_pred, y_prob= y_prob)

        cre_sco_obj= CreditScoring(model_params = [container], fair_threshold = 0.43, 
        fair_concern = "eligible", fair_priority = "benefit", fair_impact = "significant", 
        perf_metric_name = "balanced_acc", fair_metric_name = "equal_opportunity") 
        
        cre_sco_obj.evaluate()
        
        cre_sco_obj.evaluate(visualize = True)