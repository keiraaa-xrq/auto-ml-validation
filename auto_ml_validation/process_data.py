from typing import *
import pandas as pd
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from .utils import check_columns


def split_x_y(data: pd.DataFrame, target: str):
    '''
    Split a dataframe into features and target.
    '''
    check_columns(data, [target])
    data.columns = [c.lower() for c in data.columns]
    X = data.drop(target, axis=1)
    y = data[target]
    return X, y


def process_data(
    train_X: pd.DataFrame,
    test_X: pd.DataFrame,
    cat_cols: List[str]
):
    """
    Perform one-hot encoding on categorical features 
    and standard scaling on continuous features.
    """

    check_columns(train_X, cat_cols)
    # OHE
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = enc.fit_transform(train_X[cat_cols])
    X_train_cat = pd.DataFrame(
        X_train_cat, columns=enc.get_feature_names_out())
    X_test_cat = enc.transform(test_X[cat_cols])
    X_test_cat = pd.DataFrame(X_test_cat, columns=enc.get_feature_names_out())

    # Standard Scaling
    num_variables = train_X.drop(cat_cols, axis=1).columns
    sc = StandardScaler()
    X_train_num = sc.fit_transform(train_X[num_variables])
    X_train_num = pd.DataFrame(X_train_num, columns=num_variables)
    X_train_num.reset_index(drop=True, inplace=True)

    X_test_num = sc.transform(test_X[num_variables])
    X_test_num = pd.DataFrame(X_test_num, columns=num_variables)
    X_test_num.reset_index(drop=True, inplace=True)

    X_train_processed = pd.concat([X_train_cat, X_train_num], axis=1)
    X_test_processed = pd.concat([X_test_cat, X_test_num], axis=1)

    return X_train_processed, X_test_processed
