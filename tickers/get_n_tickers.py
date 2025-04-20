
import json
import random
import os
import shutil
import yfinance as yf
#####
# Lets gennerate a list of n tickers from valid_tickers.json that are not etfs or funds

def get_n_tickers(n, file_path="valid_tickers.json"):
    """
    Generates a list of n tickers from valid_tickers.json that are not ETFs or funds.

    Parameters:
        n (int): Number of tickers to generate.
        file_path (str): Path to the JSON file containing valid tickers.

    Returns:
        list: A list of n ticker symbols.
    """
    # Load tickers from the JSON file
    with open(file_path, "r") as file:
        tickers = json.load(file)

    non_etf_tickers = []

    # Filter out ETFs and funds
    for ticker in tickers:
        # Test if any of the entries of the dict contains "ETF" or "Fund"
        for key, value in ticker.items():
            if isinstance(value, str) and ("ETF" in value or "Fund" in value):
                break
            else:
                non_etf_tickers.append(ticker)


    # Test if these are available on Yahoo Finance in the date range from 2000-01-01 to 2025-01-01
    for ticker in non_etf_tickers:
        try:
            stock = yf.Ticker(ticker["ticker"])
            data = stock.history(start="2000-01-01", end="2025-01-01")
            if data.empty:
                print(f"Ticker {ticker['ticker']} has no data in the specified date range.")
                non_etf_tickers.remove(ticker)
        except Exception as e:
            print(f"Error fetching data for ticker {ticker['ticker']}: {e}")
            non_etf_tickers.remove(ticker)
    # Check if the list is empty

    # Randomly select n tickers from the filtered list
    selected_tickers = random.sample(non_etf_tickers, min(n, len(non_etf_tickers)))

    selected_ticker_strings = []

    for i, ticker in enumerate(selected_tickers):
        selected_ticker_strings.append(ticker["ticker"])

    return selected_ticker_strings


if __name__ == "__main__":
    n = 100  # Number of tickers to generate
    file_path = "tickers/valid_tickers.json"  # Path to the JSON file containing valid tickers

    # Get n tickers
    tickers = get_n_tickers(n, file_path)

    # Print the selected tickers as a list of strings use really " " to separate the tickers
    print("Selected tickers:")
    for ticker in tickers:
        print(f'"{ticker}"', end=", ")
    print("\n")