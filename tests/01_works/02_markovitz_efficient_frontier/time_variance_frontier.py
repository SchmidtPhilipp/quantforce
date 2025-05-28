import pandas as pd
import numpy as np
import os
import sys
# Include ../../ to access the get_data and tickers modules
sys.path.append(os.path.abspath(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))))
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models

from data import get_data


# Get folder of this script
script_dir = os.path.dirname(os.path.abspath(__file__))


def main():

    # Step 1: Define weight bounds and tickers
    tickers = ["NVDA", "BLDR", "UBER", "WBD"]  # Example tickers from different sectors

    # Step 2: Load historical price data
    df = get_data(
        tickers=tickers,  # Selected tickers
        start="2023-01-01",  # Start date for historical data
        end="2025-01-01",  # End date for historical data
        indicators=("Close",)  # Use closing prices
    )

    # Remove the second level of the MultiIndex (the 'Close' indicator)
    df.columns = df.columns.droplevel(1)

    data = df

    # Step 2: Define a rolling window of 1 year (252 trading days)
    rolling_window = 365  # Approx. 1 year of trading days

    # Initialize tensors to store results
    rolling_mu_tensor = []
    rolling_S_tensor = []

    # Step 3: Calculate rolling expected returns and variances
    for i in range(rolling_window, len(data)):
        window_data = data.iloc[i - rolling_window:i].values  # 1-year window as NumPy array
        returns = np.diff(np.log(window_data), axis=0)  # Log returns
        mu = np.mean(returns, axis=0) * 365  # Annualized expected returns
        S = np.cov(returns, rowvar=False) * 365  # Annualized covariance matrix
        rolling_mu_tensor.append(mu)
        rolling_S_tensor.append(S)

    # Convert tensors to NumPy arrays
    rolling_mu_tensor = np.array(rolling_mu_tensor)  # Shape: (n_windows, n_assets)
    rolling_S_tensor = np.array(rolling_S_tensor)  # Shape: (n_windows, n_assets, n_assets)

    # Step 4: Smooth the results using a rolling average
    smoothing_window = 30  # Smoothing window size (e.g., 30 days)
    smoothed_mu_tensor = pd.DataFrame(rolling_mu_tensor, columns=tickers).rolling(smoothing_window, min_periods=1).mean().values
    smoothed_S_diag = pd.DataFrame(
        rolling_S_tensor[:, np.arange(len(tickers)), np.arange(len(tickers))],
        columns=tickers
    ).rolling(smoothing_window, min_periods=1).mean().values

    # Step 5: Visualize the results in mean-variance space
    fig, ax = plt.subplots(figsize=(10, 6))

    for i, ticker in enumerate(tickers):
        ax.plot(
            smoothed_S_diag[:, i],  # Smoothed variance (diagonal of covariance matrix)
            smoothed_mu_tensor[:, i],  # Smoothed expected return
            label=ticker,
            alpha=0.8
        )

    ax.set_title("Mean-Variance Space (Smoothed Rolling 1-Year Window)")
    ax.set_xlabel("Variance")
    ax.set_ylabel("Expected Return")
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    plt.show()

    # Step 6: Save the smoothed results to a CSV file
    smoothed_df = pd.DataFrame(
        data=smoothed_mu_tensor,
        index=data.index[rolling_window:],
        columns=tickers
    )
    smoothed_variances = pd.DataFrame(
        data=smoothed_S_diag,
        index=data.index[rolling_window:],
        columns=tickers
    )
    smoothed_df = pd.concat([smoothed_df, smoothed_variances], axis=1, keys=['Expected Return', 'Variance'])
    smoothed_df.to_csv(os.path.join(script_dir, "smoothed_mean_variance.csv"))


if __name__ == "__main__":
    main()