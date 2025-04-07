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
    Saves tickers and their names to a JSON file.

    Parameters:
        file_path (str): Path to the JSON file.
        tickers (list): List of dictionaries containing tickers and their names.
    """
    with open(file_path, "w") as file:
        json.dump(tickers, file, indent=4)

def check_ticker(ticker):
    """
    Checks if a ticker is valid and fetches its details.

    Parameters:
        ticker (str): The ticker symbol to check.

    Returns:
        dict: A dictionary containing the ticker details if valid, otherwise None.
    """
    try:
        # Check if the ticker exists using yfinance
        stock = yf.Ticker(ticker)
        info = stock.info
        if info is None or "regularMarketPrice" not in info or info["regularMarketPrice"] is None:
            print(f"Ticker {ticker} returned no data or is invalid.")
            return None

        # Extract additional information
        name = info.get("longName") or info.get("shortName") or "Unknown Name"
        sector = info.get("sector", "Unknown Sector")
        industry = info.get("industry", "Unknown Industry")
        country = info.get("country", "Unknown Country")
        market_cap = info.get("marketCap", "N/A")
        currency = info.get("currency", "Unknown Currency")
        exchange = info.get("exchange", "Unknown Exchange")
        ipo_year = info.get("ipoYear", "N/A")
        dividend_yield = info.get("dividendYield", "N/A")
        beta = info.get("beta", "N/A")
        trailing_pe = info.get("trailingPE", "N/A")
        forward_pe = info.get("forwardPE", "N/A")
        fifty_two_week_high = info.get("fiftyTwoWeekHigh", "N/A")
        fifty_two_week_low = info.get("fiftyTwoWeekLow", "N/A")

        # Return the ticker details
        return {
            "ticker": ticker,
            "name": name,
            "sector": sector,
            "industry": industry,
            "country": country,
            "market_cap": market_cap,
            "currency": currency,
            "exchange": exchange,
            "ipo_year": ipo_year,
            "dividend_yield": dividend_yield,
            "beta": beta,
            "trailing_pe": trailing_pe,
            "forward_pe": forward_pe,
            "fifty_two_week_high": fifty_two_week_high,
            "fifty_two_week_low": fifty_two_week_low
        }
    except Exception as e:
        print(f"Error processing ticker {ticker}: {e}")
        return None

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
        # Get the last valid ticker
        last_ticker = valid_tickers[-1]["ticker"]
        print(f"Resuming from last valid ticker: {last_ticker}")
        return last_ticker
    return "A"  # Start from the first possible ticker if no valid tickers exist

def brute_force_all_tickers(max_length=4, file_path="tickers/valid_tickers.json"):
    """
    Brute-forces all possible ticker combinations up to a given length until all are tried.

    Parameters:
        max_length (int): Maximum length of the ticker (default is 4).
        file_path (str): Path to the JSON file to save/load tickers.

    Returns:
        list: A list of all valid tickers with their details.
    """
    # Load existing tickers from the file
    valid_tickers = load_tickers_from_file(file_path)
    print(f"Loaded {len(valid_tickers)} tickers from file.")

    # Convert to a set of tickers for faster lookup
    existing_tickers = {entry["ticker"] for entry in valid_tickers}

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

            if ticker in existing_tickers:
                continue  # Skip already fetched tickers

            # Random pause to avoid overwhelming the API
            pause_time = random.uniform(0, 1.0)
            print(f"Making request for ticker: {ticker} (pausing for {pause_time:.2f} seconds)")
            time.sleep(pause_time)

            # Check the ticker
            ticker_details = check_ticker(ticker)
            if ticker_details:
                valid_tickers.append(ticker_details)
                print(f"Valid ticker found: {ticker} ({ticker_details['name']})")
                # Save tickers to file after each successful fetch
                save_tickers_to_file(file_path, valid_tickers)

    print("All possible tickers have been tried.")
    return valid_tickers

if __name__ == "__main__":
    # Example: Manually test a single ticker
    test_ticker = "AAPL"
    print(f"Testing ticker: {test_ticker}")
    result = check_ticker(test_ticker)
    if result:
        print(f"Ticker details: {result}")
    else:
        print(f"Ticker {test_ticker} is invalid or returned no data.")

    # Run the brute-force algorithm to fetch all tickers
    valid_tickers = brute_force_all_tickers(max_length=4, file_path="tickers/valid_tickers.json")
    print(f"Total valid tickers found: {len(valid_tickers)}")