import pandas as pd
import numpy as np
import aequitas
from aequitas.preprocessing import preprocess_input_df
from aequitas.plotting import Plot
from aequitas.group import Group


class FairnessMetricsEvaluator:
    def __init__(self,
                 train=None,
                 test=None):
        self.train = train
        self.test = test
        self.cat_var_list = cat_var_list
        self.model = model

    def data_preprocess(self,
                        score_col_name: str,
                        label_col_name: str) -> tuple[pd.DataFrame,
                                                      pd.DataFrame]:
        # double-check that categorical columns are of type 'string'
        for col in self.train.columns:
            if type(col) == object:
                self.train['col'] = self.train['col'].astype(str)
        # double-check that categorical columns are of type 'string'
        for col in self.test.columns:
            if type(col) == object:
                self.test['col'] = self.test['col'].astype(str)

        # change the name of score column and label column
        self.train.rename(columns={score_col_name: "score",
                                   label_col_name: "label_value"})
        self.test.rename(columns={score_col_name: "score",
                                  label_col_name: "label_value"})

        # preprocessing using aequitas
        train_df, _ = preprocess_input_df(self.train)
        test_df, _ = preprocess_input_df(self.test)

        return train_df, test_df

    def plot_group_metric(self,
                          data: pd.DataFrame,
                          group_metric: str,
                          ax_lim=None,
                          min_group_size=None):
        g = Group()
        xtab, _ = g.get_crosstabs(data)
        aqp = Plot()
        plot = aqp.plot_group_metric(xtab,
                                     group_metric,
                                     ax_lim=ax_lim,
                                     min_group_size=min_group_size)
