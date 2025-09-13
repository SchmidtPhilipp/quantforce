"""Clean and prepare financial data.

This module provides functions to clean and prepare financial data for analysis.
"""

from typing import List, Union

import pandas as pd


def reindex_data(data, start, end):
    """
    Cleans and reindexes the data to ensure consistency.

    Parameters:
        data (pd.DataFrame): The input data.
        start (str): Start date in 'YYYY-MM-DD' format.
        end (str): End date in 'YYYY-MM-DD' format.

    Returns:
        pd.DataFrame: Cleaned data.
    """
    # Generate a complete date range
    date_range = pd.date_range(start=start, end=end, freq="D")
    data = data.reindex(date_range)  # Reindex to ensure all dates are present
    data = (
        data.ffill()
    )  # Fill any remaining missing values so the weekend data is the same as the last trading day
    return data


def drop_columns(data: pd.DataFrame, indicators: Union[List[str], str]) -> pd.DataFrame:
    """
    Drop columns that are not 'Close' or the specified indicators.

    Args:
        data: DataFrame with multi-index columns (ticker, field)
        indicators: List of indicators to keep, or 'all' to keep all columns

    Returns:
        pd.DataFrame: DataFrame with only the specified columns
    """
    if indicators == "all":
        return data

    if isinstance(indicators, str):
        indicators = [indicators]

    # Get all columns that are either 'Close' or in the indicators list
    keep_columns = []
    for ticker, field in data.columns:
        if field == "Close" or field in indicators:
            keep_columns.append((ticker, field))

    return data[keep_columns]
