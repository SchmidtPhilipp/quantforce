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

# set seeds for reproducibility
np.random.seed(42)


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
    from data.tickers import DOWJONES, NASDAQ100
    tickers = NASDAQ100

    # Fetch historical data
    start_date = "2024-01-01"
    end_date = "2025-01-01"
    data = get_data(tickers, start_date, end_date, indicators=("Close",))

    # calculate the returns referenced to the first date
    returns = data["Close"].diff().cumsum().dropna()





    # Generate deterministic portfolios
    n_portfolios = 5000  # Number of portfolios
    n_assets = len(tickers)
    #weights_list = generate_deterministic_weights(n_assets, n_portfolios)
    weights_list = generate_random_weights(n_assets, n_portfolios)

    portfolio_returns = []
    portfolio_risks = []
    sharpe_ratios = []

    for weights in tqdm.tqdm(weights_list, desc="Calculating Portfolio Metrics", unit="portfolio"):
        ret, risk = calculate_portfolio_metrics(returns, weights)
        portfolio_returns.append(ret)
        portfolio_risks.append(risk)
        sharpe_ratios.append(ret / risk if risk > 0 else 0)

    # Calculate the efficient frontier
    #efficient_risks, efficient_returns = calculate_efficient_frontier(returns)

    # Visualize the results
    plt.figure(figsize=(10, 6))
    plt.scatter(portfolio_risks, portfolio_returns, c=sharpe_ratios, cmap="viridis", marker="o", alpha=0.5, label="Deterministic Portfolios")
    
    
    #plt.plot(efficient_risks, efficient_returns, color="red", linewidth=2, label="Efficient Frontier")
    plt.colorbar(label="Portfolio Return")
    plt.xlabel("Risk (Standard Deviation)")
    plt.ylabel("Return")
    plt.title("Efficient Frontier and Deterministic Portfolios")
    plt.legend()
    plt.grid(True)
    plt.show()
    plt.savefig("efficient_frontier.png")


if __name__ == "__main__":
    main()