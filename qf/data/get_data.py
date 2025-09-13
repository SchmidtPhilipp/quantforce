"""Get historical financial data and add technical indicators.

This module provides a function to download historical financial data and add technical indicators.
It supports different data imputation methods and allows for reindexing the data to a full date range.
"""

import pandas as pd


from qf.settings import VERBOSITY
from qf.data.config.data_config import DataConfig
from qf.data.utils import (
    DataManager,
    add_technical_indicators,
    drop_columns,
    impute,
    reindex,
)
from qf.utils.logging_config import get_logger

logger = get_logger(__name__)


def get_data(
    data_config: DataConfig,
    verbosity: int = VERBOSITY,
) -> pd.DataFrame:
    """
    Downloads historical financial data and adds technical indicators.

    Args:
        data_config: DataConfig object
        verbosity: Verbosity level

    Returns:
        pd.DataFrame: Multi-index DataFrame with tickers and OHLCV data
    """
    # Initialize data manager and load data
    data_manager = DataManager(
        cache_dir=data_config.cache_dir,
        interval=data_config.interval,
        downloader=data_config.downloader,
        verbosity=verbosity,
    )
    data = data_manager.get_data(
        data_config.tickers, data_config.start, data_config.end
    )

    # If data is empty, return it without further processing
    if data.empty:
        logger.info("No data available, returning empty DataFrame")
        return data

    # Reindex the data to full calendar days if we are using 365 trading days
    data = reindex(data, data_config.n_trading_days, data_config.start, data_config.end)

    # Handle missing values between trading days
    data = impute(
        data, data_config.imputation_method, data_config.start, data_config.end
    )

    # Drop the 'Adj Close' column if it exists
    if "Adj Close" in data.columns.get_level_values(1):
        adj_close_tickers = set(t for t, field in data.columns if field == "Adj Close")
        data = data.drop(
            columns=[(ticker, "Adj Close") for ticker in adj_close_tickers]
        )

    # Handle missing values at the front of the data so before IPO
    if data_config.backfill_method == "bfill":
        data = data.bfill()
    elif data_config.backfill_method == "shrinkage":
        data = data.dropna()
    elif data_config.backfill_method == "remove_short_stocks":
        data = data.dropna(axis=1, how="any")
    elif data_config.backfill_method == "keep_nan":
        pass
    elif data_config.backfill_method == "insert_zeros":
        data = data.fillna(0)
    else:
        raise ValueError(
            f"Available backfill methods: bfill, shrinkage, remove_short_stocks, keep_nan"
        )

    # Add the technical indicators to the data
    data = add_technical_indicators(
        data, indicators=data_config.indicators, verbosity=verbosity
    )

    # Remove the columns that are not 'Close' or the specified indicators
    data = drop_columns(data, data_config.indicators)

    # Ensure that close is in the dataframe
    if "Close" not in data.columns.get_level_values(1):
        raise ValueError("The 'Close' column is missing from the data.")

    return data
