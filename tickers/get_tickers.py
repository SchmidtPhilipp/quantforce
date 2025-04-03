import yfinance as yf
import string
from itertools import product
import json
import os
import time
import random

def load_tickers_from_file(file_path):
    """
    Loads tickers from a JSON file if it exists.

    Parameters:
        file_path (str): Path to the JSON file.

    Returns:
        list: A list of tickers loaded from the file.
    """
    if os.path.exists(file_path):
        with open(file_path, "r") as file:
            return json.load(file)
    return []

def save_tickers_to_file(file_path, tickers):
    """
    Saves tickers to a JSON file.

    Parameters:
        file_path (str): Path to the JSON file.
        tickers (list): List of tickers to save.
    """
    with open(file_path, "w") as file:
        json.dump(tickers, file, indent=4)

def get_starting_point(valid_tickers, max_length):
    """
    Determines the starting point for the brute-force algorithm based on the last valid ticker.

    Parameters:
        valid_tickers (list): List of already fetched valid tickers.
        max_length (int): Maximum length of the ticker.

    Returns:
        str: The starting ticker.
    """
    if valid_tickers:
        return valid_tickers[-1]  # Start from the last valid ticker
    return "A" * max_length  # Start from the first possible ticker

def brute_force_tickers(max_length=4, max_success=100, file_path="tickers/valid_tickers.json"):
    """
    Brute-forces all possible ticker combinations up to a given length and fetches valid tickers.

    Parameters:
        max_length (int): Maximum length of the ticker (default is 4).
        max_success (int): Number of additional valid tickers to fetch.
        file_path (str): Path to the JSON file to save/load tickers.

    Returns:
        list: A list of valid tickers.
    """
    # Load existing tickers from the file
    valid_tickers = load_tickers_from_file(file_path)
    print(f"Loaded {len(valid_tickers)} tickers from file.")

    alphabet = string.ascii_uppercase  # A-Z
    starting_point = get_starting_point(valid_tickers, max_length)
    print(f"Starting from ticker: {starting_point}")

    # Calculate how many more tickers are needed
    additional_needed = max_success
    print(f"Searching for {additional_needed} additional tickers...")

    # Generate all combinations of 1 to max_length characters
    found_starting_point = False
    for length in range(1, max_length + 1):
        for ticker_tuple in product(alphabet, repeat=length):
            ticker = ''.join(ticker_tuple)
            if not found_starting_point:
                if ticker == starting_point:
                    found_starting_point = True
                continue  # Skip until we reach the starting point

            if ticker in valid_tickers:
                continue  # Skip already fetched tickers
            try:
                # Check if the ticker exists using yfinance
                stock = yf.Ticker(ticker)
                info = stock.info
                if "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                    valid_tickers.append(ticker)
                    print(f"Valid ticker found: {ticker}")
                    # Save tickers to file after each successful fetch
                    save_tickers_to_file(file_path, valid_tickers)
                    # Stop if we reach the required number of additional tickers
                    additional_needed -= 1
                    if additional_needed <= 0:
                        return valid_tickers
            except Exception:
                # Ignore invalid tickers
                pass

    return valid_tickers

def brute_force_all_tickers(max_length=4, file_path="tickers/valid_tickers.json"):
    """
    Brute-forces all possible ticker combinations up to a given length until all are tried.

    Parameters:
        max_length (int): Maximum length of the ticker (default is 4).
        file_path (str): Path to the JSON file to save/load tickers.

    Returns:
        list: A list of all valid tickers.
    """
    # Load existing tickers from the file
    valid_tickers = load_tickers_from_file(file_path)
    print(f"Loaded {len(valid_tickers)} tickers from file.")

    alphabet = string.ascii_uppercase  # A-Z
    starting_point = get_starting_point(valid_tickers, max_length)
    print(f"Starting from ticker: {starting_point}")

    # Generate all combinations of 1 to max_length characters
    found_starting_point = False
    for length in range(1, max_length + 1):
        for ticker_tuple in product(alphabet, repeat=length):
            ticker = ''.join(ticker_tuple)
            if not found_starting_point:
                if ticker == starting_point:
                    found_starting_point = True
                continue  # Skip until we reach the starting point

            if ticker in valid_tickers:
                continue  # Skip already fetched tickers
            try:
                # Random pause to avoid overwhelming the API
                pause_time = random.uniform(0, 1.0)
                print(f"Making request for ticker: {ticker} (pausing for {pause_time:.2f} seconds)")
                time.sleep(pause_time)

                # Check if the ticker exists using yfinance
                stock = yf.Ticker(ticker)
                info = stock.info
                if "regularMarketPrice" in info and info["regularMarketPrice"] is not None:
                    valid_tickers.append(ticker)
                    print(f"Valid ticker found: {ticker}")
                    # Save tickers to file after each successful fetch
                    save_tickers_to_file(file_path, valid_tickers)
            except Exception:
                # Ignore invalid tickers
                pass

    print("All possible tickers have been tried.")
    return valid_tickers

if __name__ == "__main__":
    # Run the brute-force algorithm to fetch all tickers
    valid_tickers = brute_force_all_tickers(max_length=4, file_path="tickers/valid_tickers.json")
    print(f"Total valid tickers found: {len(valid_tickers)}")
    print(valid_tickers)