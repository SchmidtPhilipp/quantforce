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
    tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"]
    # Fetch historical data
    start_date = "2024-01-01"
    end_date = "2024-12-31"
    data = get_data(tickers, start_date, end_date, indicators=("Close",))

    # Remove the second entry of the Multiindex
    data.columns = data.columns.droplevel(1)

    # Berechnung der Renditen der Wertpapiere R(T) = (S(T) - S(0)) / S(0)
    returns = (data.iloc[:] - data.iloc[0]) / data.iloc[0]


    # Plot the historic returns of 5 Assets
    
    plot_lines_grayscale(returns[tickers], xlabel="Date", ylabel="Return", filename="historic_returns_5_assets", y_limits=(-0.5, 1), 
                         figsize=(8, 2.5), max_xticks=12, save_dir="tests/02_efficient_frontier", linewidth=1)
    
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
    n_portfolios = 5_000

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
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_risks, portfolio_returns, c=sharpe_ratios, cmap="Greys", marker="o", s=10, alpha=0.5)

    plt.colorbar(label="Sharpe Ratio")
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Return")
    #plt.legend()
    plt.grid(True)
    #plt.show()
    plt.xlim(min(portfolio_risks), max(portfolio_risks))
    plt.ylim(min(portfolio_returns), max(portfolio_returns))

    plt.savefig("efficient_frontier.png")


if __name__ == "__main__":
    main()