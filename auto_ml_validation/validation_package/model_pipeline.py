from typing import *
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import time

from auto_ml_validation.validation_package.algorithms.xgboost import XGBoostClassifier
from auto_ml_validation.validation_package.algorithms.random_forest import RFClassifier
from auto_ml_validation.validation_package.process_data import split_x_y, process_data, split_train_val
from auto_ml_validation.validation_package.benchmark_pipeline import auto_benchmark, save_benchmark_output
from auto_ml_validation.validation_package.train_pipeline import train as replicate


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
    """
    # Variables and Logging
    DATE = datetime.today().strftime('%Y-%m-%d')
    save_path = f'models/{project_name}_{algorithm}_{DATE}.pkl'
    output_dict = {}
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)
    # Create a file handler and set the logging level
    file_handler = logging.FileHandler('logs.log')
    file_handler.setLevel(logging.INFO)

    # Create a log formatter and add it to the file handler
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)

    # Add the file handler to the logger
    logger.addHandler(file_handler)

    # Log a message to indicate the start of the function
    logger.info('Preparing...')
    
    try:
        logger.info('Replicating the Model...')
        vm_train_data, vm_other_data = run_model_replication(auto_train, auto_other, rep_train, rep_test, rep_other, target, algorithm, hyperparams, metric, save_path)
        if feat_sel_bool:
            logger.info('Performing feature selection...')
        logger.info('Building the benchmark model...')
        bm_train_data, bm_test_data = run_auto_bmk(auto_train, auto_test, auto_other, target, cat_cols, metric, feat_sel_bool, n_jobs =-1, mode = 'parallel')
        output_dict = {'bm_train_data': bm_train_data, 'bm_test_data': bm_test_data, 'vm_train_data': vm_train_data, 'vm_other_data': vm_other_data}
        logger.info('Almost done...')
        
        return output_dict
    
    except Exception as e:
        logger.info(e)

def run_model_replication(auto_train, auto_other, rep_train, rep_test, rep_other, target, algorithm, hyperparams, metric, save_path):
    """Full Cycle of Model Replication
    
    Returns:
        Tuple of two dictionaries, train and test
    """
    rep_others = [rep_test] + rep_other  # Add rep_test to the beginning of the list
    rep_X_train, rep_y_train = split_x_y(rep_train, target)

    rep_others_X_y = [(split_x_y(each, target)) for each in rep_others]

    valid_model, valid_threshold = replicate(rep_X_train, rep_y_train, *rep_others_X_y[0], algorithm, hyperparams, metric, save=True, save_path=save_path)
    vm_train_proba = valid_model.predict_proba(rep_X_train)
    
    vm_others_proba = []
    for X, y in rep_others_X_y:
        proba = valid_model.predict_proba(X)
        vm_others_proba.append(proba)

    vm_train_data = {'raw_X': auto_train, 'processed_X': rep_X_train, 'y': rep_y_train, 'pred_proba': vm_train_proba}
    vm_other_data = {'raw_X':auto_other, 'processed_X': [tup[0] for tup in rep_others_X_y], 'y': [tup[1] for tup in rep_others_X_y], 'pred_proba': vm_others_proba}

    return vm_train_data, vm_other_data

def run_auto_bmk(auto_train, auto_test, auto_other, target, cat_cols, metric, feat_sel_bool, n_jobs, mode):
    """Full Cycle of Auto-Benchmarking
    
    Returns:
        Tuple of two dictionaries, train and test
    """
    # Auto-Benchmarking
    auto_others = [auto_test] + auto_other
    full_auto_X_train, full_auto_y_train, auto_others = process_data(auto_train, auto_others, target, cat_cols)
    
    # Split to Train and Validation sets for building auto benchmark, here we split the train set and treat test and others set as out of sample
    auto_X_train, auto_X_val, auto_y_train, auto_y_val = split_train_val(full_auto_X_train, full_auto_y_train)
    
    auto_X_test, auto_y_test = auto_others[0] # Get Test Set
    
    benchmark_model, benchmark_output = auto_benchmark(auto_X_train, auto_y_train, 
                                                    auto_X_val, auto_y_val, metric, feat_sel_bool, 
                                                    n_jobs=n_jobs, mode=mode, verbose=True)

    bm_train_proba = benchmark_model.predict_proba(full_auto_X_train) # full_auto_X_train[[benchmark_output[benchmark_model.name]['features_selected']]])
    bm_test_proba = benchmark_model.predict_proba(auto_X_test) #auto_X_test[[benchmark_output[benchmark_model.name]['features_selected']]])
    
    
    best_auto_X_train = full_auto_X_train #full_auto_X_train[[benchmark_output[benchmark_model.name]['features_selected']]] Need to re-look as need to inverse transform
    best_auto_X_test = auto_X_test #auto_X_test[[benchmark_output[benchmark_model.name]['features_selected']]]
    
    bm_train_data = {'raw_X': auto_train, 'processed_X': best_auto_X_train, 'y': full_auto_y_train, 'pred_proba': bm_train_proba}
    bm_test_data = {'raw_X': auto_test, 'processed_X': best_auto_X_test, 'y': auto_y_test, 'pred_proba': bm_test_proba} # Only return the test data
    return bm_train_data, bm_test_data
