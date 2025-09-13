import numpy as np
import pandas as pd
from typing import Optional


def insert_random_missing_entries(
    df: pd.DataFrame,
    p_delete: float = 0.2,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Insert missing values (NaNs) at random positions in the DataFrame with probability p_delete.
    The first row is never deleted.
    """
    data = df.copy()
    if seed is not None:
        np.random.seed(seed)
    mask = np.random.rand(len(data)) < p_delete
    mask[0] = False  # never delete the first row
    data.loc[mask] = np.nan
    return data


def insert_random_missing_periods(
    df: pd.DataFrame,
    period_length: int = 5,
    p_delete_periods: float = 0.05,
    seed: Optional[int] = None,
) -> pd.DataFrame:
    """
    Insert missing values (NaNs) in random periods of given length with probability p_delete_periods.
    """
    data = df.copy()
    if seed is not None:
        np.random.seed(seed)
    for i in range(len(data) - period_length + 1):
        if np.random.random() < p_delete_periods:
            end_idx = min(i + period_length, len(data))
            data.iloc[i:end_idx] = np.nan
    return data


def insert_missing_from_offset(
    df: pd.DataFrame,
    offset: int,
) -> pd.DataFrame:
    """
    Set all values from a given offset (row index) to NaN, except for the final value.
    """
    data = df.copy()
    if offset < 0 or offset >= len(data) - 1:
        return data
    data.iloc[offset:-1] = np.nan
    return data


def remove_weekends(
    df: pd.DataFrame,
    weekend_length: int = 2,
    week_length: int = 7,
) -> pd.DataFrame:
    """
    Remove weekends (2 consecutive days every 7 days) from the DataFrame by setting them to NaN.
    By default, removes 2 days every 7 days (e.g. Saturday and Sunday).
    """
    data = df.copy()
    n = len(data)
    for i in range(weekend_length - 1, n, week_length):
        end_idx = min(i + 1, n)
        data.iloc[i - (weekend_length - 1) : end_idx] = np.nan
    return data
