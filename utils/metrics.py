import numpy as np

# Individual metric functions (for general usage)
def calculate_returns(balances):
    balances = np.array(balances)
    if len(balances) < 2:
        return np.array([])  # Return empty array if not enough data
    returns = balances[1:] / (balances[:-1] + 1e-8) - 1   
    return returns

def sharpe_ratio(returns, risk_free_rate=0.0):
    if len(returns) == 0:
        return 0.0  # Return 0 if no returns are available
    excess_returns = returns - risk_free_rate
    return np.mean(excess_returns) / (np.std(excess_returns) + 1e-8)   

def sortino_ratio(returns, risk_free_rate=0.0):
    if len(returns) == 0:
        return 0.0  # Return 0 if no returns are available
    excess_returns = returns - risk_free_rate
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = np.std(downside_returns) + 1e-8   
    return np.mean(excess_returns) / downside_std

def max_drawdown(balances):
    balances = np.array(balances)
    if len(balances) == 0:
        return 0.0  # Return 0 if no balances are available
    peak = np.maximum.accumulate(balances)
    drawdown = (balances - peak) / (peak + 1e-8)   
    return np.min(drawdown)

def volatility(returns):
    if len(returns) == 0:
        return 0.0  # Return 0 if no returns are available
    return np.std(returns)

def cumulative_return(balances):
    if len(balances) < 2:
        return 0.0  # Return 0 if not enough data
    return balances[-1] / (balances[0] + 1e-8) - 1   

def annualized_return(balances, periods_per_year=252):
    if len(balances) < 2:
        return 0.0  # Return 0 if not enough data
    returns = calculate_returns(balances)
    total_return = balances[-1] / (balances[0] + 1e-8)   
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0  # Return 0 if no periods are available
    return total_return**(periods_per_year / n_periods) - 1

def calmar_ratio(annual_ret, max_dd):
    if max_dd == 0:
        return 0.0  # Return 0 if no drawdown is available
    return annual_ret / abs(max_dd + 1e-8)   


# Metrics class
class Metrics:
    def __init__(self):
        self.metrics = {
            "sharpe": [],
            "sortino": [],
            "drawdown": [],
            "volatility": [],
            "cumulative": [],
            "annualized": [],
            "calmar": [],
        }

    def calculate(self, balances):
        """
        Calculate metrics for a single run and store them.

        Parameters:
            balances (list): Portfolio balances over time.
        """
        returns = calculate_returns(balances)
        drawdown = max_drawdown(balances)
        annual_ret = annualized_return(balances)

        self.metrics["sharpe"].append(sharpe_ratio(returns))
        self.metrics["sortino"].append(sortino_ratio(returns))
        self.metrics["drawdown"].append(drawdown)
        self.metrics["volatility"].append(volatility(returns))
        self.metrics["cumulative"].append(cumulative_return(balances))
        self.metrics["annualized"].append(annual_ret)
        self.metrics["calmar"].append(calmar_ratio(annual_ret, drawdown))

    def mean_and_std(self):
        """
        Calculate the mean and standard deviation for each metric.

        Returns:
            dict: A dictionary with mean and std for each metric.
        """
        return {
            metric: {
                "mean": np.mean(values) if len(values) > 1 else values[0],
                "std": np.std(values) if len(values) > 1 else 0
            }
            for metric, values in self.metrics.items()
        }

    def print_report(self):
        """
        Print a summary report of the metrics.
        """
        summary = self.mean_and_std()
        print("\n📊 Metrics Summary:")
        for name, stats in summary.items():
            if name in ["sharpe", "sortino", "calmar"]:  # Ratios
                print(f"{name.capitalize():<20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")
            elif name in ["drawdown", "cumulative", "annualized"]:  # Percentages
                print(f"{name.capitalize():<20s}: {stats['mean'] * 100:.2f}% ± {stats['std'] * 100:.2f}%")
            elif name == "volatility":  # Volatility as percentage
                print(f"{name.capitalize():<20s}: {stats['mean'] * 100:.2f}% ± {stats['std'] * 100:.2f}%")
            else:
                print(f"{name.capitalize():<20s}: {stats['mean']:.4f} ± {stats['std']:.4f}")

    def log_metrics(self, logger):
        """
        Log metrics using a logger.

        Parameters:
            logger (Logger): Logger instance for logging metrics.
        """
        summary = self.mean_and_std()
        for name, stats in summary.items():
            logger.log_scalar(f"metrics/{name}_mean", stats["mean"])
            logger.log_scalar(f"metrics/{name}_std", stats["std"])
