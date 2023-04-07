from typing import *
import math
import pandas as pd
import numpy as np
import scipy
from sklearn.metrics import roc_auc_score


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
            'train_perc': round(train_perc,3),
            'test_count': round(test_count,3),
            'test_perc': round(test_perc,3),
        })
        output_df.index = df_index

        x = 0
        values = []
        for _, row in output_df.iterrows():
            actual = row['train_perc']
            expected = row['test_perc']
            if actual == 0 or expected == 0:
                value = 0
            else:
                value = (actual - expected) * math.log(actual / expected)
            values.append(value)
            x = x + value
        output_df['index_value'] = [round(elem, 3) for elem in values]
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
        return round(csi,3), output_df

    def csi_for_all_features(self, ft_names: List[str], num_bins=10):
        # check_columns(self.train_raw_X, [ft_names])
        df_list = []
        csi_dict = dict()
        for feature in ft_names:
            csi, df = self.csi_for_single_feature(feature, num_bins)
            csi_dict[feature] = csi
            df_list.append(df)
        return df_list, csi_dict

    def kstest(self) -> Dict[str, float]:
        train_pos = []
        train_neg = []
        for i in range(0, len(self.train_y)):
            if self.train_y[i] == 0:
                train_neg.append(self.train_proba[i])
            else: 
                train_pos.append(self.train_proba[i])

        test_pos = []
        test_neg = []
        for i in range(0, len(self.test_y)):
            if self.test_y[i] == 0:
                test_neg.append(self.train_proba[i])
            else: 
                test_pos.append(self.train_proba[i])
        
        output_dict = {'Train' : round(scipy.stats.ks_2samp(train_pos, test_pos).statistic,3),
                       'Test' :  round(scipy.stats.ks_2samp(test_pos, test_neg).statistic,3),
                       'Train vs Test' : round(scipy.stats.ks_2samp(self.train_proba, 
                                                       self.test_proba).statistic,3)}
        return output_dict
    

    def cal_normalized_gini(self):
        """Simple normalized Gini based on Scikit-Learn's roc_auc_score""" 
        gini = lambda a, p: 2 * roc_auc_score(a, p) - 1
        return round(gini(self.test_y, self.test_proba) / gini(self.test_y, self.test_y),3)

    
    def cal_feature_gini(self):
        """Calculate GINI index for attributes"""  

        pop_X = self.train_raw_X.iloc[0:100, :]
        pop_y = self.train_y[0:100]

        # Get the indices of the samples belonging to each class
        class_0_indices = np.where(pop_y == 0)[0]
        class_1_indices = np.where(pop_y == 1)[0]

        # Set the number of samples to be randomly sampled as 1/100 of overall population
        num_samples_per_class = int(pop_X.shape[0]/100)

        # Randomly sample the indices from each class
        class_0_sampled_indices = np.random.choice(class_0_indices, size=num_samples_per_class, replace=False)
        class_1_sampled_indices = np.random.choice(class_1_indices, size=num_samples_per_class, replace=False)

        # Concatenate the sampled indices and sort them to preserve the order of the original data
        sampled_indices = np.sort(np.concatenate([class_0_sampled_indices, class_1_sampled_indices]))

        # Select the corresponding samples from X and y
        X = pop_X.iloc[sampled_indices,:]
        y = pop_y[sampled_indices]

        def _gini_impurity (value_counts):
            n = value_counts.sum()
            p_sum = 0
            for key in value_counts.keys():
                p_sum = p_sum  +  (value_counts[key] / n ) * (value_counts[key] / n ) 
            gini = 1 - p_sum
            return gini
        
        def _gini_attribute(attribute_name):
            attribute_values = X[attribute_name].value_counts()
            gini_A = 0 
            for key in attribute_values.keys():
                df_k = pd.DataFrame(y)[X[attribute_name] == key].value_counts()
                n_k = attribute_values[key]
                n = X.shape[0]
                gini_A = gini_A + (( n_k / n) * _gini_impurity(df_k))
            return round(gini_A,3)
        
        result = {}
        for key in (X).columns:
            result[key] = _gini_attribute(key)
        
        return result