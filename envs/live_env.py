import numpy as np
import pandas as pd
import yfinance as yf
import datetime

class LiveTradingEnv:
    def __init__(self, tickers, window_size=10, initial_balance=1_000):
        self.tickers = tickers
        self.window_size = window_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.history = []

    def fetch_latest_data(self):
        end = datetime.datetime.now()
        start = end - datetime.timedelta(days=30)
        data = yf.download(self.tickers, start=start, end=end, interval='1d')
        close_prices = pd.DataFrame({ticker: data[ticker]['Close'] for ticker in self.tickers})
        return close_prices.dropna()

    def reset(self):
        self.balance = self.initial_balance
        self.history = self.fetch_latest_data()
        return self.history[-self.window_size:].values[-1]

    def step(self, action):
        new_data = self.fetch_latest_data()
        if len(new_data) <= len(self.history):
            # No new data yet
            return self.history[-1].values, 0.0, False, {}

        self.history = new_data
        prev_prices = self.history.iloc[-2].values
        new_prices = self.history.iloc[-1].values
        returns = new_prices / prev_prices
        portfolio_return = np.dot(action, returns)
        self.balance *= portfolio_return
        reward = np.log(portfolio_return)
        obs = new_prices

        print(f"[LIVE] Action: {np.round(action, 2)} | Reward: {reward:.4f} | Balance: {self.balance:.2f}")
        return obs, reward, False, {}
