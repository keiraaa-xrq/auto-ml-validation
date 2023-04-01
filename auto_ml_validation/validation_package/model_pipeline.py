from typing import *
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import logging

from auto_ml_validation.validation_package.algorithms.xgboost import XGBoostClassifier
from auto_ml_validation.validation_package.algorithms.random_forest import RFClassifier
from auto_ml_validation.validation_package.process_data import split_x_y, process_data, split_train_val
from auto_ml_validation.validation_package.benchmark_pipeline import auto_benchmark, save_benchmark_output
from auto_ml_validation.validation_package.train_pipeline import train as replicate
from .utils.logger import setup_logger, log_info, log_error

logger = setup_logger(logging.getLogger(__name__))

def autoML(project_name: str, algorithm: str, hyperparams: dict,
           rep_train: pd.DataFrame, rep_test: pd.DataFrame, rep_other: List, target: str, cat_cols: List,
           auto_train: pd.DataFrame, auto_test: pd.DataFrame, auto_other: List, metric: str, feat_sel_bool: bool) -> Dict[str, Dict[str, pd.DataFrame]]:
    """Main Function to generate the output

    Args:
        project_name (str): Project Name
        algorithm (str): Algorithm of the Model to Replicate
        hyperparams (dict): Hyperparameters of the model
        rep_train (pd.DataFrame): Processed Data for Model Replication Training
        rep_test (pd.DataFrame): Processed Data for Model Replication Testing
        rep_other (List): Processed Data for Model Replication Others
        target (str): Target Variable
        cat_cols (List): Columns that are categorical in nature
        auto_train (pd.DataFrame): Raw Data for Auto-Benchmarking Training
        auto_test (pd.DataFrame): Raw Data for Auto-Benchmarking Testing
        auto_other (List): Raw Data for Auto-Benchmarking Others
        metric (str): Metric to tune e.g. F1, Precision, Recall, ROC/AUC
        feat_sel_bool (bool): True/False boolean, whether to use feature selection tool

    Returns:
        Dict[str, Dict[str, pd.DataFrame]]: Full dictionary containing raw and processed dataframes and predictions e.g. {data name: {raw/processed :x_df, y: y_df, predict_proba : proba} }
        str: Output file name produced from model
    """
    # Variables and Logging
    DATE = datetime.today().strftime('%Y-%m-%d')
    save_path = f'models/{project_name}_{algorithm}_{DATE}.pkl'
    output_dict = bm_train_data = bm_other_data =  re_train_data = re_other_data = {}
    try:
        log_info(logger, 'Replicating the model...')
        re_train_data, re_other_data = run_model_replication(auto_train, auto_test, auto_other, rep_train, rep_test, rep_other, target, algorithm, hyperparams, metric, save_path)
    except Exception as e:
        error_message = "Returning to homepage... An error occurred while replicating the model: %s." % str(e)
        log_error(logger, error_message)
        raise e

    try:
        log_info(logger, 'Creating the benchmark model...')
        bm_train_data, bm_other_data = run_auto_bmk(auto_train, auto_test, auto_other, target, cat_cols, metric, feat_sel_bool, n_jobs =-1, mode = 'parallel')
    except Exception as e:
        error_message = "Returning to homepage... An error occurred while creating the benchmark model: %s. " % str(e)
        log_error(logger, error_message)
        raise e
    
    output_dict = {'bm_train_data': bm_train_data, 'bm_other_data': bm_other_data, 're_train_data': re_train_data, 're_other_data': re_other_data}
    log_info(logger, 'Almost done...')
    
    # Save Output
    with open(f'data/validator_input/{project_name}_{algorithm}_{DATE}_data.pkl', 'wb') as f:
        pickle.dump(output_dict, f)

    return output_dict, f'{project_name}_{algorithm}_{DATE}_data.pkl'
    

def run_model_replication(auto_train, auto_test, auto_other, rep_train, rep_test, rep_other, target, algorithm, hyperparams, metric, save_path):
    """Full Cycle of Model Replication
    
    Returns:
        Tuple of two dictionaries, train and test
    """
    auto_others = [auto_test] + auto_other # For Raw set to return
    rep_others = [rep_test] + rep_other  # Add rep_test to the beginning of the list
    rep_X_train, rep_y_train = split_x_y(rep_train, target)

    rep_others_X_y = [(split_x_y(each, target)) for each in rep_others]
    rep_model, valid_threshold = replicate(rep_X_train, rep_y_train, *rep_others_X_y[0], algorithm, hyperparams, metric, save=True, save_path=save_path)
    re_train_proba = rep_model.predict_proba(rep_X_train)
    
    re_train_data = {'raw_X': auto_train.drop(target, axis=1), 'processed_X': rep_X_train, 'y': rep_y_train, 'pred_proba': re_train_proba}
    re_other_data = {}
    for i, (X,y) in enumerate(rep_others_X_y):
        key = "Test" if i == 0 else f"Other{i}"
        re_other_data[key] = {}
        re_other_data[key]['raw_X'] = auto_others[i].drop(columns=[target])
        re_other_data[key]['processed_X'] = rep_others_X_y[i][0]
        re_other_data[key]['y'] = [tup[1] for tup in rep_others_X_y][i]
        #pred_proba = 
        re_other_data[key]['pred_proba'] = rep_model.predict_proba(rep_others_X_y[i][0])# pd.DataFrame(pred_proba, columns=["proba_0", "proba_1"])

    return re_train_data, re_other_data

def run_auto_bmk(auto_train, auto_test, auto_other, target, cat_cols, metric, feat_sel_bool, n_jobs, mode):
    """Full Cycle of Auto-Benchmarking
    
    Returns:
        Tuple of two dictionaries, train and test
    """
    # Auto-Benchmarking
    auto_others_raw = [auto_test] + auto_other
    full_auto_X_train, full_auto_y_train, auto_others, col_mapping = process_data(auto_train, auto_others_raw, target, cat_cols)

    # Split to Train and Validation sets for building auto benchmark, here we split the train set and treat test and others set as out of sample
    auto_X_train, auto_X_val, auto_y_train, auto_y_val = split_train_val(full_auto_X_train, full_auto_y_train)
    
    benchmark_model, benchmark_output = auto_benchmark(auto_X_train, auto_y_train, 
                                                    auto_X_val, auto_y_val, metric, feat_sel_bool, 
                                                    n_jobs=n_jobs, mode=mode, verbose=True)
    
    # To inverse OHE, col_mapping dictionary (k, v) where k is OHE processed column and v is original column
    feats_selected = benchmark_output[benchmark_model.name]['features_selected']
    feats_selected_mapped = [col_mapping.get(col, col) for col in feats_selected]

    bm_train_proba = benchmark_model.predict_proba(full_auto_X_train[feats_selected])
    # To predict for test and all other datasets
    best_auto_X_train = full_auto_X_train[feats_selected] 
    bm_train_data = {'raw_X': auto_train[feats_selected_mapped].T.drop_duplicates().T, 'processed_X': best_auto_X_train, 'y': full_auto_y_train, 'pred_proba': bm_train_proba}

    bm_other_data = {}
    for i, (X,y) in enumerate(auto_others):
        key = "Test" if i == 0 else f"Other{i}"
        bm_other_data[key] = {}
        bm_other_data[key]['raw_X'] = auto_others_raw[i][feats_selected_mapped].T.drop_duplicates().T
        bm_other_data[key]['processed_X'] = auto_others[i][0][feats_selected]
        bm_other_data[key]['y'] = auto_others[i][1]
        #pred_proba = 
        bm_other_data[key]['pred_proba'] = benchmark_model.predict_proba(auto_others[i][0][feats_selected]) #pd.DataFrame(pred_proba, columns=["proba_0", "proba_1"])
             
    
    
    return bm_train_data, bm_other_data
