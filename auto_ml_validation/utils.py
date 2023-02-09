from typing import *
import pandas as pd


def check_columns(data: pd.DataFrame, columns: List[str]):
    """
    Check the presence of columns in the data.
    """
    cols_all = set(data.columns.tolist())
    all_present = True
    not_present = []
    for c in columns:
        if c not in cols_all:
            all_present = False
            not_present.append(c)
    if not all_present:
        raise KeyError(f"Columns {not_present} are not present in the data.")
