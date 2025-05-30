import pandas as pd
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.discrete_allocation import DiscreteAllocation, get_latest_prices
from data import get_data

# Step 1: Load historical price data
# Replace 'stock_prices.csv' with your CSV file containing historical price data
# The CSV should have dates as the index and asset tickers as columns
df = get_data(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],  # Example tickers
    start="2020-01-01",
    end="2023-01-01",
    indicators=("Close",)  # Use only the closing prices for MVO
)

# Step 2: Calculate expected returns and sample covariance matrix
mu = expected_returns.mean_historical_return(df)  # Expected annual returns
S = risk_models.sample_cov(df)  # Sample covariance matrix of returns

# Step 3: Optimize for maximum Sharpe ratio
ef = EfficientFrontier(mu, S)
weights = ef.max_sharpe()  # Maximize Sharpe ratio
cleaned_weights = ef.clean_weights()  # Clean weights to remove negligible allocations
print("Optimized Weights:")
print(cleaned_weights)

# Step 4: Display expected performance
expected_annual_return, annual_volatility, sharpe_ratio = ef.portfolio_performance(verbose=True)

# Step 5: Compute discrete allocation of each share per asset
latest_prices = get_latest_prices(df)
total_portfolio_value = 100000  # Example: $100,000 investment
da = DiscreteAllocation(weights, latest_prices, total_portfolio_value=total_portfolio_value)
allocation, leftover = da.greedy_portfolio()

print("\nDiscrete Allocation:")
for asset, shares in allocation.items():
    print(f"{asset}: {shares} shares")
print(f"Funds remaining: ${leftover:.2f}")
