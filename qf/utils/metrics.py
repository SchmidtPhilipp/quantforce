import numpy as np

from qf.utils.logging_config import get_logger

logger = get_logger(__name__)


# Individual metric functions (for general usage)
def calculate_returns(balances):
    balances = np.array(balances)
    if len(balances) < 2:
        return np.array([])  # Return empty array if not enough data
    returns = balances[1:] / (balances[:-1] + 1e-8) - 1
    return returns


def sharpe_ratio(returns, risk_free_rate=0.0, periods_per_year=365):
    if len(returns) == 0:
        return 0.0
    excess_returns = returns - risk_free_rate
    mean_excess = np.mean(excess_returns)
    std_excess = np.std(excess_returns) + 1e-8
    return (mean_excess / std_excess) * np.sqrt(periods_per_year)


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


def volatility(returns, periods_per_year=365):
    if len(returns) == 0:
        return 0.0
    return np.std(returns) * np.sqrt(periods_per_year)


def cumulative_return(balances):
    if len(balances) < 2:
        return 0.0  # Return 0 if not enough data
    return balances[-1] / (balances[0] + 1e-8) - 1


def annualized_return(balances, periods_per_year=365):
    if len(balances) < 2:
        return 0.0  # Return 0 if not enough data
    returns = calculate_returns(balances)
    total_return = balances[-1] / (balances[0] + 1e-8)
    n_periods = len(returns)
    if n_periods == 0:
        return 0.0  # Return 0 if no periods are available
    return total_return ** (periods_per_year / n_periods) - 1


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

    def append(self, balances):
        """
        Calculate metrics for a single run and store them.

        Parameters:
            balances (list): Portfolio balances over time.
        """
        # balances = balances.squeeze()
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
                "std": np.std(values) if len(values) > 1 else 0,
            }
            for metric, values in self.metrics.items()
        }

    def formated(self, std_dev=True):
        """
        Format the mean and optionally the standard deviation for each metric.

        Parameters:
            std_dev (bool): Ob die Standardabweichung in der Ausgabe enthalten sein soll.

        Returns:
            dict: A dictionary with formatted mean (and optionally std) for each metric.
        """
        summary = self.mean_and_std()
        formatted_summary = {}
        for name, stats in summary.items():
            if name in ["drawdown", "cumulative", "annualized", "volatility"]:

                # Remap names
                if name == "drawdown":
                    name = "Max Drawdown"
                elif name == "cumulative":
                    name = "Cumulative Return"
                elif name == "annualized":
                    name = "Annualized Return"
                elif name == "volatility":
                    name = "Annualized Volatility"

                if std_dev:
                    formatted_summary[name] = (
                        f"{stats['mean'] * 100:.2f} \\% ± {stats['std'] * 100:.2f} \\%"
                    )
                else:
                    formatted_summary[name] = f"{stats['mean'] * 100:.2f} \\%"
            else:

                # Remap names
                if name == "sharpe":
                    name = "Sharpe Ratio"
                elif name == "sortino":
                    name = "Sortino Ratio"
                elif name == "calmar":
                    name = "Calmar Ratio"

                if std_dev:
                    formatted_summary[name] = (
                        f"{stats['mean']:.3f} ± {stats['std']:.3f}"
                    )
                else:
                    formatted_summary[name] = f"{stats['mean']:.3f}"
        return formatted_summary

    def print_report(self):
        """
        Print a summary report of the metrics.
        """
        summary = self.mean_and_std()
        logger.info("📊 Metrics Summary:")
        for name, stats in summary.items():
            if name in [
                "drawdown",
                "cumulative",
                "annualized",
                "volatility",
            ]:  # Percentages
                logger.info(
                    f"{name.capitalize():<12s}: {stats['mean'] * 100:>7.2f}% ± {stats['std'] * 100:>4.2f}%"
                )
            else:
                logger.info(
                    f"{name.capitalize():<12s}: {stats['mean']:>8.3f} ± {stats['std']:>5.3f}"
                )

    def log(self, logger, run_type="train"):
        """
        Log metrics using a logger.

        Parameters:
            logger (Logger): Logger instance for logging metrics.
        """
        summary = self.mean_and_std()
        for name, stats in summary.items():
            logger.log_scalar(f"{run_type}_metrics/{name}_mean", stats["mean"])
            logger.log_scalar(f"{run_type}_metrics/{name}_std", stats["std"])

    def reset(self):
        """
        Reset the metrics.
        """
        self.metrics = {
            "sharpe": [],
            "sortino": [],
            "drawdown": [],
            "volatility": [],
            "cumulative": [],
            "annualized": [],
            "calmar": [],
        }

    def save(self, path):
        """
        Save the metrics to a file.

        Parameters:
            path (str): Path to save the metrics.
        """
        np.savez(path, **self.metrics)

    @staticmethod
    def load(path):
        """
        Load metrics from a file.

        Parameters:
            path (str): Path to load the metrics from.

        Returns:
            Metrics: An instance of Metrics with loaded data.
        """
        data = np.load(path, allow_pickle=True)
        metrics = Metrics()
        metrics.metrics = {key: data[key].tolist() for key in data.files}
        return metrics

    @staticmethod
    def get_metrics_files(folders):
        """
        Sammelt alle Metrics-Dateien aus den angegebenen Ordnern.

        :param folders: Liste von Ordnern, in denen nach Metrics-Dateien gesucht wird.
        :return: Liste von Pfaden zu den gefundenen Metrics-Dateien.
        """
        import os

        metrics_files = []
        for folder in folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith(".npz") and "metrics" in file.lower():
                        metrics_files.append(os.path.join(root, file))
        return metrics_files
