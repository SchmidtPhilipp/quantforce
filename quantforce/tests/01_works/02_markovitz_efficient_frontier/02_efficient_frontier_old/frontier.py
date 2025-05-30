import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# Include ../../ to access the get_data and tickers modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data.get_data import get_data
from data.tickers import getNasdaq100
import tqdm 
from utils.plot import plot_lines_grayscale


# set seeds for reproducibility
#np.random.seed(42)


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
    portfolio_risk = (np.dot(weights.T, np.dot(returns.cov(), weights)))
    return portfolio_return, portfolio_risk


def calculate_efficient_frontier(returns):
    """
    Calculate the efficient frontier by optimizing portfolio weights.

    Parameters:
        returns (pd.DataFrame): Historical returns of assets.

    Returns:
        tuple: Lists of risks and returns for the efficient frontier.
    """
    n_assets = returns.shape[1]
    mean_returns = returns.mean()
    cov_matrix = returns.cov()

    def portfolio_volatility(weights):
        return np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))

    # Constraints: weights sum to 1
    constraints = ({'type': 'eq', 'fun': lambda x: np.sum(x) - 1})
    bounds = tuple((0, 1) for _ in range(n_assets))

    efficient_risks = []
    efficient_returns = []

    target_returns = np.linspace(mean_returns.min(), mean_returns.max(), 100)
    for target in target_returns:
        constraints = (
            {'type': 'eq', 'fun': lambda x: np.sum(x) - 1},
            {'type': 'eq', 'fun': lambda x: np.dot(x, mean_returns) - target}
        )
        result = minimize(portfolio_volatility, n_assets * [1. / n_assets], 
                          method='SLSQP', bounds=bounds, constraints=constraints)
        if result.success:
            efficient_risks.append(result.fun)
            efficient_returns.append(target)

    return efficient_risks, efficient_returns


def generate_deterministic_weights(n_assets, n_portfolios):
    """
    Generate n_portfolios deterministic weight vectors for n_assets.

    Parameters:
        n_assets (int): Number of assets in the portfolio.
        n_portfolios (int): Number of portfolios to generate.

    Returns:
        list: List of weight vectors.
    """
    weights_list = []
    step = 1 / (n_portfolios - 1)  # Step size for weights
    for i in range(n_portfolios):
        weights = np.zeros(n_assets)
        weights[i % n_assets] = 1 - (i // n_assets) * step
        weights[(i + 1) % n_assets] = (i // n_assets) * step
        weights_list.append(weights / np.sum(weights))  # Normalize weights
    return weights_list


def generate_random_weights(n_assets, n_portfolios):
    """
    Generate n_portfolios random weight vectors for n_assets.

    Parameters:
        n_assets (int): Number of assets in the portfolio.
        n_portfolios (int): Number of portfolios to generate.

    Returns:
        list: List of weight vectors.
    """
    weights_list = []
    for _ in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)  # Normalize weights
        weights_list.append(weights)
    return weights_list


def main():
    from data.tickers import DOWJONES, NASDAQ100, SNP_500

    tickers = [
    "JNJ",  # Johnson & Johnson (Health, USA)
    "NVS",  # Novartis (Health, Switzerland)
    "LMT",  # Lockheed Martin (Defense, USA)
    "BA",   # Boeing (Defense, USA)
    "AAPL", # Apple (Technology, USA)
    "SAP",  # SAP (Technology, Germany)
    "XOM",  # Exxon Mobil (Energy, USA)
    "BP",   # BP (Energy, UK)
    "HSBC", # HSBC (Finance, UK)
    "DB",   # Deutsche Bank (Finance, Germany)
    ]

    tickers = ["AAPL", "JNJ", "XOM"]

    # Fetch historical data
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    data = get_data(tickers, start_date, end_date, indicators=("Close",))

    # Remove the second entry of the Multiindex
    data.columns = data.columns.droplevel(1)

    n_assets = len(tickers)
    prices = np.array(data)
    T = len(prices)  # Anzahl der Zeitpunkte

    # Schritt 1: Log-Renditen berechnen
    returns = np.log(prices[1:] / prices[:-1])
    T = len(returns)  # Anzahl der Zeitpunkte

    # Schritt 2: Erwartete Renditen und Kovarianzmatrix
    mean_returns = np.mean(returns, axis=0)
    cov_matrix = np.cov(returns, rowvar=False)

    # Schritt 3: Monte Carlo Simulation
    n_portfolios = 10_000

    portfolio_returns = []
    portfolio_risks = []
    sharpe_ratios = []

    for i in range(n_portfolios):
        weights = np.random.random(n_assets)
        weights /= np.sum(weights)

        portfolio_return = np.dot(weights, mean_returns)*T
        portfolio_vol = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))*np.sqrt(T)
        sharpe_ratio = portfolio_return / portfolio_vol

        portfolio_returns.append(portfolio_return)
        portfolio_risks.append(portfolio_vol)
        sharpe_ratios.append(sharpe_ratio)
        

    # Visualize the results
    plt.figure(figsize=(10, 4))
    plt.scatter(portfolio_risks, portfolio_returns, c=sharpe_ratios, cmap="Greys", marker="o", s=10, alpha=0.5)
    plt.colorbar(label="Sharpe Ratio")
    max_sharpe_idx = np.argmax(sharpe_ratios)
    max_sharpe_return = portfolio_returns[max_sharpe_idx]
    plt.scatter(portfolio_risks[max_sharpe_idx], portfolio_returns[max_sharpe_idx], color="red", marker="*", s=200, label="Max Sharpe Ratio")
    
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Return")
    plt.legend()
    plt.grid(True)
    #plt.show()
    #plt.xlim(min(portfolio_risks), max(portfolio_risks))
    #plt.ylim(min(portfolio_returns), max(portfolio_returns))

    plt.ylim(-0.1, 0.3)

    plt.savefig("tests/02_efficient_frontier/efficient_frontier.png")


if __name__ == "__main__":
    main()