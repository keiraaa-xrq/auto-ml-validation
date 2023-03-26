from typing import *
import math
import pandas as pd
import numpy as np
<<<<<<< HEAD
from scipy.stats import ks_2samp
=======
import scipy
from ..utils.utils import check_columns

>>>>>>> keira_main

class StatisticalMetricsEvaluator:
    def __init__(
        self,
        train_data: Dict[str, Union[pd.DataFrame, np.ndarray]],
        test_data: Dict[str, Union[pd.DataFrame, np.ndarray]],
    ):
        self.train_raw_X = train_data['raw_X']
        self.train_processed_X = train_data['processed_X']
        self.train_y = train_data['y']
        self.train_proba = train_data['pred_proba'][:, 1]
        self.test_raw_X = test_data['raw_X']
        self.test_processed_X = test_data['processed_X']
        self.test_y = test_data['y']
        self.test_proba = test_data['pred_proba'][:, 1]

<<<<<<< HEAD
    def csi_for_each_feature(self,
                         ft_name: str,
                         num_bins=10):
        lower_range = min(min(self.train[ft_name]), min(self.test[ft_name]))
        upper_range = max(max(self.train[ft_name]), max(self.test[ft_name]))
        
        bin_list = np.arange(lower_range, 
                            upper_range+(upper_range-lower_range)/num_bins, 
                            (upper_range-lower_range)/num_bins)
        
        grouped_train = self.train.groupby(pd.cut(self.train[ft_name], bin_list)).count()
        grouped_train['perc_'+ft_name] = grouped_train[ft_name]/sum(grouped_train[ft_name])
        grouped_test = self.test.groupby(pd.cut(self.test[ft_name], bin_list)).count()
        grouped_test['perc_'+ft_name] = grouped_test[ft_name]/sum(grouped_test[ft_name])
        
        # build a CSI dataframe to show the output
        output_df = pd.DataFrame()
        output_df['train_count'] = grouped_train[ft_name]
        output_df['train_perc'] = grouped_train['perc_'+ft_name]
        output_df['test_count'] = grouped_test[ft_name]
        output_df['test_perc'] = grouped_test['perc_'+ft_name]
        output_df['index'] =  [0]*num_bins
        
        csi = 0
        for i, row in output_df.iterrows():
=======
    def _generate_output(
        self,
        bin_list: List[float],
        train_values: pd.Series,
        test_values: pd.Series
    ) -> Tuple[float, pd.DataFrame]:
        """
        Calculate the value and generate output df for psi and csi.
        """
        train_count = train_values.groupby(
            pd.cut(train_values, bin_list)).count()
        train_perc = train_count.apply(
            lambda x: x / sum(train_count))
        test_count = test_values.groupby(
            pd.cut(test_values, bin_list)).count()
        test_perc = test_count.apply(
            lambda x: x / sum(test_count))

        df_index = [
            f'({bin_list[i]:.2f}, {bin_list[i + 1]:.2f}]' for i in range(len(bin_list) - 1)]
        output_df = pd.DataFrame({
            'train_count': train_count,
            'train_perc': train_perc,
            'test_count': test_count,
            'test_perc': test_perc,
        })
        output_df.index = df_index

        x = 0
        values = []
        for _, row in output_df.iterrows():
>>>>>>> keira_main
            actual = row['train_perc']
            expected = row['test_perc']
            if actual == 0 or expected == 0:
                value = 0
            else:
                value = (actual - expected) * math.log(actual / expected)
            values.append(value)
            x = x + value
        output_df['index_value'] = values
        return x, output_df

    def calculate_psi(self, num_bins=10) -> float:
        train_proba = pd.Series(self.train_proba)
        test_proba = pd.Series(self.test_proba)
        bin_list = np.arange(0, 1+1/num_bins, 1/num_bins).tolist()
        psi, output_df = self._generate_output(
            bin_list,
            train_proba,
            test_proba
        )
        return psi, output_df

    def csi_for_single_feature(self, ft_name: str, num_bins=10):
        train_l = self.train_raw_X[ft_name]
        test_l = self.test_raw_X[ft_name]
        lower = min(min(train_l), min(test_l))
        upper = max(max(train_l), max(test_l))

        bin_list = np.arange(
            lower,
            upper + (upper - lower) / num_bins,
            (upper - lower) / num_bins
        ).tolist()

        csi, output_df = self._generate_output(
            bin_list,
            train_l,
            test_l
        )
        return csi, output_df

<<<<<<< HEAD
    def csi_for_all_features(self,
                         ft_name_list: list[str],
                         num_bins=10):
=======
    def csi_for_all_features(self, ft_names: List[str], num_bins=10):
        # check_columns(self.train_raw_X, [ft_names])
>>>>>>> keira_main
        df_list = []
        csi_dict = dict()
        for feature in ft_names:
            csi, df = self.csi_for_single_feature(feature, num_bins)
            csi_dict[feature] = csi
            df_list.append(df)
        return df_list, csi_dict

<<<<<<< HEAD
    
    def kstest(self,
               score_col_name: str)-> float:
        return ks_2samp(self.processed_train[score_col_name], self.processed_test[score_col_name])
=======
    def kstest(
        self,
        score_col_name: str
    ) -> float:
        return scipy.stats.ks_2samp(self.train_processed_X[score_col_name], self.test_processed_X[score_col_name])
>>>>>>> keira_main
