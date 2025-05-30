import pandas as pd


def clean_data(data, start, end):
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
    data = data.ffill().bfill()  # Fill any remaining missing values
    return data


def drop_columns(data, indicators):
    """
    Drops all columns that are not 'Close' or specified indicators.

    Parameters:
        data (pd.DataFrame): The input data with multi-indexed columns.
        indicators (list[str]): List of indicator names to retain.

    Returns:
        pd.DataFrame: Filtered data containing only 'Close' and specified indicator columns.
    """
    # Keep only 'Close' and columns matching the specified indicators
    filtered_columns = [
        col for col in data.columns if col[1] == "Close" or col[1] in indicators
    ]
    return data.loc[:, filtered_columns]