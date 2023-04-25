from typing import *
import os
import pandas as pd
import numpy as np
from datetime import datetime
import pickle
import logging

from auto_ml_validation.validation_package.process_data import split_x_y, process_data
from auto_ml_validation.validation_package.benchmark_pipeline import auto_benchmark, save_benchmark_output
from auto_ml_validation.validation_package.train_pipeline import train as replicate
from .utils.logger import setup_main_logger, log_info, log_error

logger = logging.getLogger("main."+__name__)


def auto_ml(project_name: str, algorithm: str, date: str, hyperparams: dict,
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
    outputs_dir = f'./outputs/{project_name}/{date}'
    os.makedirs(outputs_dir, exist_ok=True)
    rep_save_path = f'{outputs_dir}/{algorithm}_replicated.pkl'
    auto_save_path = f'{outputs_dir}/benchmark_model.pkl'

    logger = setup_main_logger(project_name)

    try:
        log_info(logger, 'Replicating the model...')
        re_train_data, re_other_data = run_model_replication(
            auto_train, auto_test, auto_other, rep_train, rep_test, rep_other, target, algorithm, hyperparams, metric, rep_save_path)
    except Exception as e:
        error_message = "Returning to homepage... An error occurred while replicating the model: %s." % str(
            e)
        log_error(logger, error_message)
        raise e

    try:
        if feat_sel_bool:
            log_info(logger, 'Beginning Feature Selection...')
        log_info(logger, 'Creating the benchmark model...')
        bm_models_dir = f'{outputs_dir}/bm_models'
        os.makedirs(bm_models_dir, exist_ok=True)
        bm_results_path = f'{outputs_dir}/bm_results.json'
        bm_train_data, bm_other_data, bm_name = run_auto_bmk(
            auto_train, auto_test, auto_other, target, cat_cols, metric, feat_sel_bool, n_jobs=-1, mode='parallel',
            save_path=auto_save_path, models_dir=bm_models_dir, results_path=bm_results_path
        )
    except Exception as e:
        error_message = "Returning to homepage... An error occurred while creating the benchmark model: %s. " % str(
            e)
        log_error(logger, error_message)
        raise e

    output_dict = {'bm_train_data': bm_train_data, 'bm_other_data': bm_other_data,
                   're_train_data': re_train_data, 're_other_data': re_other_data}
    log_info(logger, 'Almost done...')

    # Save Output
    output_path = f'{outputs_dir}/validator_input_data.pkl'
    with open(output_path, 'wb') as f:
        pickle.dump(output_dict, f)

    return output_path, rep_save_path, auto_save_path, bm_name


def run_model_replication(raw_train, raw_test, raw_others, rep_train, rep_test, rep_others, target, algorithm, hyperparams, metric, save_path):
    """Full Cycle of Model Replication

    Returns:
        Tuple of two dictionaries, train and test
    """
    raw_others = [raw_test] + raw_others  # For Raw set to return
    # Add rep_test to the beginning of the list
    rep_others = [rep_test] + rep_others
    rep_X_train, rep_y_train = split_x_y(rep_train, target)

    rep_others_X_y = [(split_x_y(df, target)) for df in rep_others]
    rep_model, threshold = replicate(
        rep_X_train, rep_y_train, *rep_others_X_y[0], algorithm, hyperparams, metric, save=True, save_path=save_path)

    re_train_proba = rep_model.predict_proba(rep_X_train)
    re_train_data = {
        'raw_X': raw_train.drop(columns=[target]),
        'processed_X': rep_X_train,
        'y': rep_y_train,
        'pred_proba': re_train_proba
    }

    re_other_data = {}
    for i, (X, y) in enumerate(rep_others_X_y):
        key = "Test" if i == 0 else f"Other{i}"
        re_other_data[key] = {}
        re_other_data[key]['raw_X'] = raw_others[i].drop(columns=[target])
        re_other_data[key]['processed_X'] = X
        re_other_data[key]['y'] = y
        re_other_data[key]['pred_proba'] = rep_model.predict_proba(X)
    return re_train_data, re_other_data


def run_auto_bmk(raw_train, raw_test, raw_others, target, cat_cols, metric, feat_sel_bool, n_jobs, mode, save_path, models_dir, results_path):
    """Full Cycle of Auto-Benchmarking

    Returns:
        Tuple of two dictionaries, train and test
    """
    # Auto-Benchmarking
    raw_others = [raw_test] + raw_others
    X_train, y_train, others, col_mapping = process_data(
        raw_train, raw_others, target, cat_cols)
    # print(col_mapping)
    X_test, y_test = others[0]

    benchmark_model, benchmark_output = auto_benchmark(X_train, y_train,
                                                       X_test, y_test, metric, feat_sel_bool,
                                                       n_jobs=n_jobs, mode=mode, save=True, save_path=save_path, verbose=True)

    save_benchmark_output(benchmark_output, models_dir, results_path)
    # To inverse OHE, col_mapping dictionary (k, v) where k is OHE processed column and v is original column
    feats_selected = benchmark_output[benchmark_model.name]['features_selected']
    feats_selected_mapped = set([col_mapping.get(
        col, col) for col in feats_selected])

    X_train_selected = X_train[feats_selected]
    bm_train_proba = benchmark_model.predict_proba(X_train_selected)
    bm_train_data = {
        'raw_X': raw_train[list(feats_selected_mapped)],
        'processed_X': X_train_selected,
        'y': y_train,
        'pred_proba': bm_train_proba
    }
    # To predict for test and all other datasets
    bm_other_data = {}
    for i, (X, y) in enumerate(others):
        key = "Test" if i == 0 else f"Other{i}"
        bm_other_data[key] = {}
        bm_other_data[key]['raw_X'] = raw_others[i][list(
            feats_selected_mapped)],
        bm_other_data[key]['processed_X'] = X[feats_selected]
        bm_other_data[key]['y'] = y
        bm_other_data[key]['pred_proba'] = benchmark_model.predict_proba(
            X[feats_selected])

    return bm_train_data, bm_other_data, benchmark_model.name
