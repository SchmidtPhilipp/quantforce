import numpy as np
import pandas as pd


def reindex(
    data: pd.DataFrame, n_trading_days: int, start=None, end=None
) -> pd.DataFrame:
    """
    Reindex the data to full calendar days.

    This function reindexes a DataFrame to include all calendar days between the start and end dates,
    filling missing values with NaN. This is useful when working with 365-day trading calendars
    to ensure all calendar days are represented in the data.

    Parameters
    ----------
    data : pd.DataFrame
        The DataFrame to reindex.
    n_trading_days : int
        Number of trading days per year. Must be either 252 or 365.
        If 365, the data will be reindexed to full calendar days.
    start : datetime, optional
        Start date for the reindexing. If None, uses the first date in the data.
    end : datetime, optional
        End date for the reindexing. If None, uses the last date in the data.

    Returns
    -------
    pd.DataFrame
        Reindexed DataFrame with all calendar days included (when n_trading_days=365).

    Raises
    ------
    ValueError
        If n_trading_days is not 252 or 365.
    """

    data = data.copy()
    # If we are using 365 trading days, reindex to full calendar days
    if n_trading_days == 365:
        # Get full date range
        if start is None:
            start = data.index[0]
        if end is None:
            end = data.index[-1]
        full_range = pd.date_range(start=start, end=end, freq="D")
        # Reindex to full calendar days, forward filling missing values with nan
        data = data.reindex(full_range, fill_value=np.nan)

    elif n_trading_days != 252:
        raise ValueError("n_trading_days must be either 252 or 365.")

    return data
