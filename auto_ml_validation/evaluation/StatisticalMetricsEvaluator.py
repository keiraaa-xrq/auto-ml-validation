import math
import pandas as pd
import numpy as np

class StatisticalMetricsEvaluator:
    def __init__(self,
                 train = None, 
                 test = None, 
                 processed_train = None,
                 processed_test = None):
        self.train = train
        self.test = test
        self.processed_train = processed_train
        self.processed_test = processed_test
    
    def calculate_psi(self,
                 score_col_name: str,
                 num_bins=10
                 ) -> int:
        bin_list = np.arange(0, 1+1/num_bins, 1/num_bins)
        grouped_train = self.processed_train.groupby(pd.cut(self.processed_train[score_col_name], bin_list)).count()
        grouped_train['perc_'+score_col_name] = grouped_train[score_col_name]/sum(grouped_train[score_col_name])
        grouped_test = self.processed_test.groupby(pd.cut(self.processed_test[score_col_name], bin_list)).count()
        grouped_test['perc_'+score_col_name] = grouped_test[score_col_name]/sum(grouped_test[score_col_name])
        
        # build a PSI dataframe to show the output
        output_df = pd.DataFrame()
        output_df['train_prob_count'] = grouped_train[score_col_name]
        output_df['train_prob_perc'] = grouped_train['perc_'+score_col_name]
        output_df['test_prob_count'] = grouped_test[score_col_name]
        output_df['test_prob_perc'] = grouped_test['perc_'+score_col_name]
        output_df['index'] =  [0]*num_bins
        
        psi = 0
        for i, row in output_df.iterrows():
            actual = row['train_prob_perc']
            expected = row['test_prob_perc']
            if actual == 0 or expected == 0:
                continue
            index = (actual-expected)*math.log(actual/expected)
            output_df.loc[i, 'index'] = index
            psi = psi + index
        return psi, output_df

    def csi_for_each_feature(self,
                         ft_name: str,
                         num_bins=10):
        lower_range = min(min(train_set[ft_name]), min(test_set[ft_name]))
        upper_range = max(max(train_set[ft_name]), max(test_set[ft_name]))
        
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
            actual = row['train_perc']
            expected = row['test_perc']
            if actual == 0 or expected == 0:
                continue
            index = (actual-expected)*math.log(actual/expected)
            output_df.loc[i, 'index'] = index
            csi = csi + index
            
        return csi, output_df

    def csi_for_all_features(self,
                         ft_name_list: [str],
                         num_bins=10):
        df_list = []
        csi_dict = dict()
        for feature in ft_name_list:
            csi, df = self.csi_for_each_feature(feature, num_bins)
            csi_dict[feature] = csi
            df_list.append(df)
        return df_list, csi_dict

    
    def kstest(self,
               score_col_name: str)-> float:
        return ks_2samp(self.train['score_col_name'], self.test['score_col_name'])