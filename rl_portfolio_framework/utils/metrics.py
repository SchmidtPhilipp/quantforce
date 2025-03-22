import numpy as np

def calculate_returns(balances):
    balances = np.array(balances)
    returns = balances[1:] / balances[:-1] - 1
    return returns

def sharpe_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)

def sortino_ratio(returns, risk_free_rate=0.0):
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns) + 1e-8
    return np.mean(excess_returns) / downside_std

def max_drawdown(balances):
    balances = np.array(balances)
    peak = np.maximum.accumulate(balances)
    drawdown = (balances - peak) / peak
    return np.min(drawdown)

def volatility(returns):
    return np.std(returns)

def cumulative_return(balances):
    return balances[-1] / balances[0] - 1

def annualized_return(balances, periods_per_year=252):
    returns = calculate_returns(balances)
    total_return = balances[-1] / balances[0]
    n_periods = len(returns)
    return total_return**(periods_per_year / n_periods) - 1

def calmar_ratio(annual_ret, max_dd):
    return annual_ret / abs(max_dd + 1e-8)
