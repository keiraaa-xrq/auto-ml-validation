from typing import *
import pandas as pd
import re
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from .utils.utils import check_columns


def split_x_y(
    data: pd.DataFrame,
    target: str
) -> Tuple[pd.DataFrame, pd.Series]:
    '''
    Split a dataframe into features and target.
    '''
    check_columns(data, [target])
    data.columns = [c.lower() for c in data.columns]
    X = data.drop(target, axis=1)
    y = data[target]
    return X, y

def split_train_val(
    X_train: pd.DataFrame, 
    y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    '''
    Split to train and validation sets
    '''
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)
    return X_train, X_val, y_train, y_val
    

def process_features(
    X_train: pd.DataFrame,
    X_others: List[pd.DataFrame],
    cat_cols: List[str]
) -> Tuple[pd.DataFrame, List[pd.DataFrame]]:
    """
    Perform one-hot encoding on categorical features 
    and standard scaling on continuous features.
    """

    check_columns(X_train, cat_cols)
    # OHE
    enc = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
    X_train_cat = enc.fit_transform(X_train[cat_cols])
    X_train_cat = pd.DataFrame(
        X_train_cat, columns=enc.get_feature_names_out())
    
    # Dictionary Mapping
    col_mapping = {}
    for col_name in X_train[cat_cols]: # Mapping Categorical Variables
        # Extract the column prefix using regex
        col_prefix = re.findall(r'^[a-z]+', col_name)[0]
        # Add the mapping to the dictionary
        for ohe_col in X_train_cat.columns:
            if col_prefix in ohe_col:
                col_mapping[ohe_col] = col_name
    for col in X_train.drop(cat_cols, axis=1).columns: # Mapping Numerical Variables
        col_mapping[col] = col
    
    # Standard Scaling
    num_variables = X_train.drop(cat_cols, axis=1).columns
    sc = StandardScaler()
    X_train_num = sc.fit_transform(X_train[num_variables])
    X_train_num = pd.DataFrame(X_train_num, columns=num_variables)
    X_train_processed = pd.concat([X_train_cat, X_train_num], axis=1)
    # Process other dfs
    X_others_processed = []
    for X in X_others:
        X_cat = enc.transform(X[cat_cols])
        X_cat = pd.DataFrame(X_cat, columns=enc.get_feature_names_out())
        X_num = sc.transform(X[num_variables])
        X_num = pd.DataFrame(X_num, columns=num_variables)
        X_processed = pd.concat([X_cat, X_num], axis=1)
        X_others_processed.append(X_processed)
    return X_train_processed, X_others_processed, col_mapping


def process_data(
    train_df: pd.DataFrame,
    other_dfs: List[pd.DataFrame],
    target: str,
    cat_cols: List[str]
) -> Tuple[pd.DataFrame, pd.Series, List[Tuple[pd.DataFrame, pd.Series]]]:
    """
    Perform data processing.
    """
    X_train, y_train = split_x_y(train_df, target)
    X_others, y_others = [], []
    for df in other_dfs:
        X, y = split_x_y(df, target)
        X_others.append(X)
        y_others.append(y)
    X_train_processed, X_others_processed, col_mapping = process_features(
        X_train, X_others, cat_cols)
    others_processed = list(zip(X_others_processed, y_others))
    return X_train_processed, y_train, others_processed, col_mapping
