import sys
import os
import numpy as np
import matplotlib.pyplot as plt

# Include ../../ to access the get_data and tickers modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data.get_data import get_data
from data.tickers import getNasdaq100


def calculate_portfolio_metrics(returns, weights):
    """
    Calculate portfolio return and risk (standard deviation).

    Parameters:
        returns (pd.DataFrame): Historical returns of assets.
        weights (np.ndarray): Portfolio weights.

    Returns:
        tuple: Portfolio return and risk.
    """
    portfolio_return = np.dot(weights, returns.mean())
    portfolio_risk = np.sqrt(np.dot(weights.T, np.dot(returns.cov(), weights)))
    return portfolio_return, portfolio_risk


def main():
    # Load NASDAQ-100 tickers
    tickers = getNasdaq100()

    # Fetch historical data
    start_date = "2020-01-01"
    end_date = "2023-01-01"
    data = get_data(tickers, start_date, end_date, indicators=("Close",))

    # Calculate daily returns
    returns = data.xs("Close", level=1, axis=1).pct_change().dropna()

    # Generate 500 random portfolios
    n_portfolios = 500
    n_assets = len(tickers)
    portfolio_returns = []
    portfolio_risks = []

    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)  # Normalize weights
        ret, risk = calculate_portfolio_metrics(returns, weights)
        portfolio_returns.append(ret)
        portfolio_risks.append(risk)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_risks, portfolio_returns, c=portfolio_returns, cmap="viridis", marker="o")
    plt.colorbar(label="Portfolio Return")
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Return")
    plt.title("Random Portfolio Risk vs. Return")
    plt.grid(True)
    plt.show()


if __name__ == "__main__":
    main()