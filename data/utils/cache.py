import os
import pandas as pd



def load_cache(cache_file, verbosity):
    """
    Loads cached data from a file if it exists.

    Parameters:
        cache_file (str): Path to the cache file.
        verbosity (int): Verbosity level for logging.

    Returns:
        pd.DataFrame: Cached data or an empty DataFrame if the file does not exist.
    """
    if os.path.exists(cache_file):
        if verbosity > 0:
            print(f"Loading data from cache: {cache_file}")
        return pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
    return pd.DataFrame()


def save_cache(data, cache_file, verbosity):
    """
    Saves data to a cache file.

    Parameters:
        data (pd.DataFrame): The data to save.
        cache_file (str): Path to the cache file.
        verbosity (int): Verbosity level for logging.
    """
    data.to_csv(cache_file)
    if verbosity > 0:
        print(f"Data cached to: {cache_file}")


def update_cache(existing_data, new_data):
    """
    Updates the cache with new data.

    Parameters:
        existing_data (pd.DataFrame): The existing cached data.
        new_data (pd.DataFrame): The new data to add.

    Returns:
        pd.DataFrame: Updated cached data.
    """
    # Combine existing and new data, ensuring no duplicates
    combined_data = pd.concat([existing_data, new_data]).sort_index()
    combined_data = combined_data[~combined_data.index.duplicated(keep="last")]
    return combined_data