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
        try:
            df = pd.read_csv(cache_file, header=[0, 1], index_col=0, parse_dates=True)
            # Prüfe, ob DataFrame leer ist oder keine Spalten hat
            if df.empty or df.shape[1] == 0:
                if verbosity > 0:
                    print(f"Cache file {cache_file} is empty or has no columns. Deleting file.")
                os.remove(cache_file)
                return pd.DataFrame()
            return df
        except Exception as e:
            if verbosity > 0:
                print(f"Error loading cache file {cache_file}: {e}. Deleting file.")
            os.remove(cache_file)
            return pd.DataFrame()
    return pd.DataFrame()


def save_cache(data, cache_file, verbosity):
    """
    Saves data to a cache file.

    Parameters:
        data (pd.DataFrame): The data to save.
        cache_file (str): Path to the cache file.
        verbosity (int): Verbosity level for logging.
    """
    # We ensure that there is data in the data before saving
    if data.empty:
        raise Warning("Attempted to save an empty DataFrame to cache. No data saved.")

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