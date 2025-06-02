import pandas as pd
import numpy as np
import os
import sys

import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier

from qf import get_data
from qf import setup_pgf, reset_pgf

# Get folder of this script
import os
import matplotlib as mpl
script_dir = os.path.dirname(os.path.abspath(__file__))
from matplotlib.widgets import Slider

# seed the random number generator for reproducibility
np.random.seed(42)

import yfinance


def main():
    """
    Main function to calculate and visualize the efficient frontier for a given set of assets.
    Includes Monte Carlo simulation for random portfolios and highlights optimal portfolios.
    """

    # Step 1: Define weight bounds and tickers
    weight_bounds = (0, 1)  # Bounds for weights: between 0% and 100%
    acc_tickers = [

    ## GOLD
    #iShares Physical Gold ETC

    # 
    "CSSPX.MI",         # iShares Core S&P 500 UCITS ETF USD            (Acc) – ISIN: IE00B5BMR087
    "IWDA.L",           # iShares Core MSCI World UCITS ETF USD         (Acc) – ISIN: IE00B4L5Y983 - 5J,R = 90%

    # Defence (America, Europe)
    "ASWC.DE",           # Future of Defence UCITS ETF                   (Acc) – ISIN: IE000OJ5TQP4 - 1J,R = 60%  - Maybe

    # Mixed Austria
    "XB4A.DE",          # Xtrackers ATX UCITS ETF 1C                    (Acc) – ISIN: LU0659579063 - 5J,R = 120%

    ### Emerging Markets 
    # India
    #"IE00BHZRQZ17.SG",   # Franklin FTSE India UCITS ETF                 (Acc) - ISIN: IE00BHZRQZ17 - 5J,R = 120%
    # Emerging Markets China, Taiwan, India
    "EMIM.L",           # iShares Core MSCI EM IMI UCITS ETF USD        (Acc) – ISIN: IE00BKM4GZ66 - 5J,R = 44%

    ]

    dis_tickers = [

    ## Big Data
    "XAIX.DE",          # Xtrackers Artificial Intelligence and Big D (Dis) – ISIN: IE00BGV5VN51 - 5J,R = 150%    -  Yes
    
    ### Bank ETFs
    # Europe Banks
    "EXX1.DE",          # iShares EURO STOXX Banks 30-15 UCITS ETF (DE)(Dis) – ISIN: DE0006289309 - 5J,R = 260%   -  Yes

    ### Mixed Asset ETFs
    # Europe Mixed 
    "IQQE.DE",          # iShares Core MSCI Europe UCITS ETF EUR        (Dis) – ISIN: IE00B1YZSC51  - 5J,R = 60%

    ### Emerging Markets 
    # Emerging Markets - China, India, Taiwan  
    "IEMM.AS",          # iShares MSCI EM UCITS ETF USD                 (Dis) – ISIN: IE00B0M63177  - 5J,R = 23%

    ### Technology    
    # Mostly American Tech
    "IWRD.L",           # iShares MSCI World UCITS ETF                  (Dis) – ISIN: IE00B0M62Q58  - 5J,R = 75%
    "VWRD.L",           # Vanguard FTSE All-World UCITS ETF (USD)       (Dis) – ISIN: IE00B3RBWM25  - 5J,R = 75%
    
    # America Tech
    "SPY5.DE",          # SPDR S&P 500 UCITS ETF                        (Dis) – ISIN: IE00B6YX5C33   - 5J,R = 80%
    "VUSA.L",           # Vanguard S&P 500 UCITS ETF (USD) Distributing (Dis) – ISIN: IE00B3XXRP09  - 5J,R = 90%

    ]

    # Combine both lists of tickers
    tickers = acc_tickers + dis_tickers

    # Step 2: Load historical price data
    df = get_data(tickers, start="2020-06-01", end="2025-06-01", indicators=["Close"])

    # Remove the second level of the MultiIndex (the 'Close' indicator)
    df.columns = df.columns.droplevel(1)

    # Gestapelter Plot erstellen
    plot = False
    if plot:
        df.plot(subplots=True, layout=(len(df.columns), 1), figsize=(20, len(df.columns) * 1), sharex=True)
        plt.tight_layout()
        plt.show()

        linear_rate_of_return = df.pct_change()
        linear_rate_of_return.plot(subplots=True, layout=(len(linear_rate_of_return.columns), 1), figsize=(20, len(linear_rate_of_return.columns) * 1), sharex=True)
        plt.tight_layout()
        plt.show()


    log_returns = True


    mu = expected_returns.mean_historical_return(df, frequency=365, log_returns=log_returns)
    S = risk_models.risk_matrix(df, frequency=365, method="ledoit_wolf", log_returns=log_returns)


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
   #plt.xlim((0.3, 0.475))  # Adjust x-axis limits
    #plt.ylim((-0.2, 0.6))  # Adjust y-axis limits

    # Step 9: Save plot
    plt.savefig(os.path.join(script_dir, "efficient_frontier.png"), dpi=300, bbox_inches='tight')
    #plt.savefig(os.path.join(script_dir, "efficient_frontier.pgf"), bbox_inches='tight')

    # Show plot
    plt.show()

if __name__ == "__main__":
    # Configure Matplotlib to use LaTeX for text rendering
    #setup_pgf()

    # Run the main function
    main()

    # Reset Matplotlib configuration
    #reset_pgf()