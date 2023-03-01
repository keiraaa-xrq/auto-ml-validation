class statistical_metrics_evaluator:
    def __init__(self, train = None, test = None):
        self.train = train
        self.test = test
    
    def calculate_psi(self, train_set: pd.DataFrame,
                 test_set: pd.DataFrame,
                 score_col_name: str,
                 num_bins=10
                 ) -> int:
    
        grouped_train = train_set.groupby(pd.cut(train_set[score_col_name], num_bins)).count()
        grouped_train['perc_'+score_col_name] = grouped_train[score_col_name]/sum(grouped_train[score_col_name])
        grouped_test = test_set.groupby(pd.cut(test_set[score_col_name], num_bins)).count()
        grouped_test['perc_'+score_col_name] = grouped_test[score_col_name]/sum(grouped_test[score_col_name])
        
        psi = 0
        for i in range(0, grouped_train.shape[0]):
            actual = grouped_train.iloc[i]['perc_'+score_col_name]
            expected = grouped_test.iloc[i]['perc_'+score_col_name]
            if actual == 0 or expected == 0:
                continue
            index = (actual-expected)*math.log(actual/expected)
            psi = psi + index
        return psi
    
    def calculate_csi(self, scores_for_each_range,
                 train_set: pd.DataFrame,
                 test_set: pd.DataFrame,
                 ft_name: str,
                 num_bins=10,
                 ) -> int:
    
        grouped_train = train_set.groupby(pd.cut(train_set[ft_name], num_bins)).count()
        grouped_train['perc_'+ft_name] = grouped_train[ft_name]/sum(grouped_train[ft_name])
        grouped_test = test_set.groupby(pd.cut(test_set[ft_name], num_bins)).count()
        grouped_test['perc_'+ft_name] = grouped_test[ft_name]/sum(grouped_test[ft_name])
        # print(grouped_test)
        csi = 0
        
        for i in range(0, grouped_train.shape[0]):
            actual = grouped_train['perc_'+ft_name].iloc[i]
            expected = grouped_test['perc_'+ft_name].iloc[i]
            index = (actual-expected)*scores_for_each_range[i]
            csi = csi + index
            
        return csi

    
    def kstest(self, train_set: pd.DataFrame,
                 test_set: pd.DataFrame,
                 score_col_name: str)-> float:
        return ks_2samp(train_set['score_col_name'], test_set['score_col_name'])

    