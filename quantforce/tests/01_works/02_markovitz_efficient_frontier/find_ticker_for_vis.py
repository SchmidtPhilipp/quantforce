import pandas as pd
import numpy as np
import os
import sys
# Include ../../ to access the get_data and tickers modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier
from data import get_data
from data import DOWJONES, NASDAQ100, SNP500

weight_bounds = (0, 1)  # Bounds for weights: between 0% and 100%


tickers = ["NVDA", "BLDR", "UBER", "WBD"]
tickers = SNP500

# Step 1: Load historical price data
df = get_data(
    tickers=tickers,
    #tickers=DOWJONES,  # Use DOWJONES for Dow Jones Industrial Average
    start="2024-01-01",
    end="2025-01-01",
    indicators=("Close",)
)

# remove the second level of the MultiIndex (the 'Close' indicator)
df.columns = df.columns.droplevel(1)

# Step 2: Calculate expected returns and covariance matrix
mu = expected_returns.mean_historical_return(df)
S = risk_models.sample_cov(df)



# Step 5: Plot Efficient Frontier
fig, ax = plt.subplots(figsize=(10, 6))

# Plot individual asset points in grayscale
for i, ticker in enumerate(df.columns):
    asset_return = mu[ticker]
    asset_vol = np.sqrt(S.loc[ticker, ticker])
    ax.scatter(asset_vol, asset_return, marker="x", s=70, color="0.2", label=ticker)
    ax.annotate(ticker, (asset_vol, asset_return),
                textcoords="offset points", xytext=(5, 5), ha="left", fontsize=9, color="0.2")



# Axis labels and legend with LaTeX
#ax.set_title(r"\textbf{Efficient Frontier with Risk-free Asset}", color="0.2")
#ax.set_xlabel(r"$\sigma$~(\mathrm{Volatility})", color="0.2")
#ax.set_ylabel(r"$\mathrm{E}[R]$~(\mathrm{Expected~Return})", color="0.2")
#ax.legend()
plt.grid(True, color="0.85")
plt.tight_layout()
plt.show()
plt.savefig("efficient_frontier.png", dpi=300)
