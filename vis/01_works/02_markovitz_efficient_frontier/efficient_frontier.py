import pandas as pd
import numpy as np
import os
import sys
# Include ../../ to access the get_data and tickers modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
from data import get_data
from utils.plot import setup_pgf, reset_pgf

from data import DOWJONES, NASDAQ100, SNP500

# Get folder of this script
import os
import matplotlib as mpl
script_dir = os.path.dirname(os.path.abspath(__file__))


# seed the random number generator for reproducibility
np.random.seed(42)


def main():
    """
    Main function to calculate and visualize the efficient frontier for a given set of assets.
    Includes Monte Carlo simulation for random portfolios and highlights optimal portfolios.
    """

    # Step 1: Define weight bounds and tickers
    weight_bounds = (0, 1)  # Bounds for weights: between 0% and 100%
    tickers = ["NVDA", "BLDR", "UBER", "WBD"]  # Example tickers from different sectors

    # Step 2: Load historical price data
    df = get_data(
        tickers=tickers,  # Selected tickers
        start="2020-01-01",  # Start date for historical data
        end="2025-01-01",  # End date for historical data
        indicators=("Close",)  # Use closing prices
    )

    # Remove the second level of the MultiIndex (the 'Close' indicator)
    df.columns = df.columns.droplevel(1)

    # Step 3: Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(df)  # Expected returns
    S = risk_models.sample_cov(df)  # Covariance matrix


    # My calculation of the covariance matrix:
    def sample_cov(df, periods=252):
        """
        Calculate the sample covariance matrix of asset returns.
        This is a custom implementation to demonstrate covariance calculation.
        """

        returns = df.pct_change().dropna()
        R = returns.values
        n = len(R)

        mu = R.mean(axis=0)                    

        C = (R.T @ R / (n - 1) - np.outer(mu, mu))
        C = pd.DataFrame(C, index=returns.columns, columns=returns.columns)*periods

        # Ensure the covariance matrix is positive semi-definite
        C = (C + C.T) / 2
    
        return C, mu


    # Step 4: Create an efficient frontier object
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

    # Step 5: Calculate key portfolios
    # Max Sharpe Ratio Portfolio
    ef_max_sharpe = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    max_sharpe_weights = ef_max_sharpe.max_sharpe()
    ret_ms, std_ms, sharpe_ms = ef_max_sharpe.portfolio_performance()

    # Min Volatility Portfolio
    ef_min_vol = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    min_vol_weights = ef_min_vol.min_volatility()
    ret_mv, std_mv, _ = ef_min_vol.portfolio_performance()

    # Step 6: Plot Efficient Frontier
    fig, ax = plt.subplots(figsize=(6, 6))

    # Plot individual asset points in grayscale
    for i, ticker in enumerate(df.columns):
        asset_return = mu[ticker]
        asset_vol = np.sqrt(S.loc[ticker, ticker])
        ax.scatter(asset_vol, asset_return, marker=".", s=70, color="0.2", label=ticker)
        ax.annotate(ticker, (asset_vol, asset_return),
                    textcoords="offset points", xytext=(5, -2.5), ha="left", fontsize=9, color="0.2")

    # Plot the efficient frontier
    plot_efficient_frontier(ef, ax=ax, show_assets=False, color="1", linewidth=2)

    # Step 7: Monte Carlo Simulation - Generate random portfolios
    n_portfolios = 2000  # Number of random portfolios
    mc_returns = []  # List to store portfolio returns
    mc_vols = []  # List to store portfolio volatilities
    mc_sharpes = []  # List to store Sharpe ratios

    risk_free_rate = 0.0  # Risk-free rate (adjust as needed)

    for _ in range(n_portfolios):
        # Generate random weights
        weights = np.random.dirichlet(np.ones(len(tickers)), size=1)[0]
        port_return = np.dot(weights, mu)  # Portfolio return
        port_vol = np.sqrt(np.dot(weights.T, np.dot(S, weights)))  # Portfolio volatility
        sharpe = (port_return - risk_free_rate) / port_vol if port_vol > 0 else np.nan  # Sharpe ratio
        mc_returns.append(port_return)
        mc_vols.append(port_vol)
        mc_sharpes.append(sharpe)

    # Plot Monte Carlo portfolios, colored by Sharpe Ratio
    cmap = "hsv"  # Colormap for Sharpe Ratio
    sharpe_normalized = (mc_sharpes - np.nanmin(mc_sharpes)) / (np.nanmax(mc_sharpes) - np.nanmin(mc_sharpes))
    sc = ax.scatter(mc_vols, mc_returns, c=mc_sharpes, cmap=cmap, s=5, alpha=0.6, label="Monte Carlo Portfolios")
    cbar = plt.colorbar(sc, ax=ax, label="Sharpe Ratio", orientation='horizontal', location="bottom", shrink=1, aspect=40, anchor=(0.5, 0.5))

    # Highlight optimal portfolios
    ax.scatter(std_ms, ret_ms, marker="*", s=100, color="red", label="Max Sharpe Ratio")
    ax.scatter(std_mv, ret_mv, marker=5, s=100, color="black", label="Min Volatility")

    # Step 8: Customize plot
    ax.set_xlabel("$\sigma_p = \sqrt{\mathbb{V}ar[r_p]}$ (Portfolio risk)")  # X-axis label
    ax.set_ylabel("$\mathbb{E}[r_p]$ (Expected rate of return)")  # Y-axis label

    # Legend above the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.3), ncol=4, frameon=True)
    #ax.legend(loc="right", bbox_to_anchor=(0.5, 0.5, 0.1, 0.5), frameon=True)
    # Grid and layout adjustments
    plt.grid(True, color="0.85")
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.xlim((0.3, 0.475))  # Adjust x-axis limits
    plt.ylim((-0.2, 0.6))  # Adjust y-axis limits

    # Step 9: Save plot
    plt.savefig(os.path.join(script_dir, "efficient_frontier.png"), dpi=300, bbox_inches='tight')
    plt.savefig(os.path.join(script_dir, "efficient_frontier.pgf"), bbox_inches='tight')

    # Show plot
    plt.show()

if __name__ == "__main__":
    # Configure Matplotlib to use LaTeX for text rendering
    setup_pgf()

    # Run the main function
    main()

    # Reset Matplotlib configuration
    reset_pgf()