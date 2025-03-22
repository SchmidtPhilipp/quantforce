import yfinance as yf
import pandas as pd

def download_data(tickers, start, end):
    data = yf.download(tickers, start=start, end=end, group_by='ticker')
    close_prices = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in tickers})

    
    return close_prices.dropna()
