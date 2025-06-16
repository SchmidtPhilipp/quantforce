from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

import qf
from qf.agents.agent import Agent

value = risk_models.sample_cov


def terminal_statistics(price_df: pd.DataFrame, log_returns: bool = False):
    """
    Berechnet terminale Renditen (linear oder log), Kovarianzmatrix und Mittelwert
    aus einem DataFrame mit Preiszeitreihen für mehrere Assets.

    :param price_df: DataFrame mit shape [T, n_assets], Spalten = Assets
    :param use_log_returns: Falls True, werden logarithmische statt linearer Renditen berechnet
    :return:
        - terminal_returns: Tensor mit shape [T-1, n_assets]
        - mean_returns: Tensor mit shape [n_assets]
        - cov_matrix: Tensor mit shape [n_assets, n_assets]
    """
    assert isinstance(price_df, pd.DataFrame), "Input must be a DataFrame"
    assert price_df.shape[0] > 1, "DataFrame must contain at least two time steps"

    prices = torch.tensor(price_df.values, dtype=torch.float32)  # shape: [T, N]
    p0 = prices[0] + 1e-8  # Initial prices, shape: [N]
    pt = prices[1:]  # Later prices, shape: [T-1, N]

    if log_returns:
        terminal_returns = torch.log(pt) - torch.log(p0)
    else:
        terminal_returns = (pt / p0) - 1  # shape: [T-1, N]

    mean_returns = terminal_returns.mean(dim=0)  # shape: [N]
    cov_matrix = torch.cov(terminal_returns.T)  # shape: [N, N]

    # Convert mean_returns and cov_matrix to pandas Series/DataFrame for compatibility
    mean_returns = pd.Series(mean_returns.numpy(), index=price_df.columns)
    cov_matrix = pd.DataFrame(
        cov_matrix.numpy(), index=price_df.columns, columns=price_df.columns
    )

    return mean_returns, cov_matrix


class ClassicOnePeriodMarkovitzAgent(Agent):
    def __init__(self, env, config=None):
        """
        Initializes the ClassicOnePeriodMarkovitzAgent with the given environment.

        Parameters:
            env: The environment in which the agent operates.
            config (dict, optional): Configuration dictionary for the agent.
        """
        super().__init__(env=env)

        default_config = {
            "log_returns": True,  # Whether to use log returns for calculations
            "target": "Tangency",  # Optimization target: Tangency, MaxExpReturn, MinVariance
            "risk_model": "sample_cov",  # Risk model: sample_cov, semicovariance, exp_cov, etc.
            "risk_free_rate": 0.00,  # Risk-free rate for Tangency optimization
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        self.historical_data = None
        self.weights = None  # torch.Tensor of shape (n_assets + 1,)

    def train(
        self,
        episodes=0,
        total_timesteps=0,
        use_tqdm=True,
        save_best=True,
        eval_env=None,
        n_eval_steps=None,
        n_eval_episodes=1,
        print_eval_metrics=False,
    ):
        """
        Trains the agent by calculating portfolio weights based on the selected target and risk model.
        """

        # We reload the data with "shrinkage" imputation method to ensure that the data is complete and std dev is not biased
        self.historical_data = qf.get_data(
            tickers=self.env.tickers,
            start=self.env.start,
            end=self.env.end,
            indicators="Close",
            imputation_method="shrinkage",
        )

        if self.config["risk_model"] == "ML_brownian_motion_logreturn":
            # Use the MultiAssetBrownianMotionLogReturn model to estimate drift and covariance
            from qf.agents.classic_agents.utils.ml_brownian_motion_logreturn import (
                MultiAssetBrownianMotionLogReturn,
            )

            delta_t = qf.DEFAULT_INTERVAL

            # Transform string of format "1d", "1h", etc. to float
            if delta_t.endswith("d"):
                delta_t = float(delta_t[:-1])
            elif delta_t.endswith("h"):
                delta_t = float(delta_t[:-1]) / 24.0
            elif delta_t.endswith("m"):
                delta_t = float(delta_t[:-1]) / (24.0 * 60.0)
            else:
                raise ValueError(
                    f"Unsupported delta_t format: {delta_t}. Expected format like '1d', '1h', or '1m'."
                )

            drift, covariance = (
                MultiAssetBrownianMotionLogReturn.estimate_drift_and_covariance(
                    self.historical_data, delta_t
                )
            )
            expected_return, cov_matrix = (
                MultiAssetBrownianMotionLogReturn.estimate_linear_return_expectation_and_covariance(
                    drift, covariance, T=1.0
                )
            )
            expected_return = pd.Series(
                expected_return, index=self.historical_data.columns
            )

        elif self.config["risk_model"] == "terminal_statistics":
            expected_return, cov_matrix = terminal_statistics(
                self.historical_data, log_returns=self.config["log_returns"]
            )

        elif self.config["risk_model"] == "stepwise_statistics":
            expected_return = expected_returns.mean_historical_return(
                self.historical_data,
                frequency=365,
                log_returns=self.config["log_returns"],
            )  # Expected returns
            cov_matrix = risk_models.risk_matrix(
                prices=self.historical_data,
                method="sample_cov",  # Use sample covariance for stepwise statistics
                log_returns=self.config["log_returns"],
                frequency=365,  # Default frequency for historical returns
            )
        else:
            # Compute covariance matrix using the selected risk model
            try:
                # Calculate expected returns using the selected method
                expected_return = expected_returns.mean_historical_return(
                    self.historical_data,
                    frequency=365,
                    log_returns=self.config["log_returns"],
                )  # Expected returns
                cov_matrix = risk_models.risk_matrix(
                    prices=self.historical_data,
                    method=self.config["risk_model"],
                    log_returns=self.config["log_returns"],
                    frequency=365,  # Default frequency for historical returns
                )
            except ValueError:
                raise ValueError(f"Unsupported risk model: {self.config['risk_model']}")

        # Fix the matrix if it is not positive definite
        if not np.all(np.linalg.eigvals(cov_matrix) > 0):
            cov_matrix = risk_models.fix_non_positive_definite(cov_matrix)

        # Select optimization target
        n_assets = cov_matrix.shape[0]
        if self.config["target"] == "Tangency":
            ef = EfficientFrontier(expected_return, cov_matrix)
            weights = ef.max_sharpe(risk_free_rate=self.config["risk_free_rate"])
        elif self.config["target"] == "MaxExpReturn":
            weights = pd.Series(
                0, index=expected_return.index
            )  # Initialize weights as a pandas Series with asset labels
            weights[expected_return.idxmax()] = (
                1.0  # Allocate all weight to the asset with the highest mean return
            )
            weights = OrderedDict(weights.items())
        elif self.config["target"] == "MinVariance":
            ef = EfficientFrontier(expected_return, cov_matrix)
            weights = ef.min_volatility()
        else:
            raise ValueError(f"Unsupported target: {self.config['target']}")

        # Convert weights to torch.Tensor and add cash weight
        weights = torch.tensor(list(weights.values()), dtype=torch.float32)
        weights_with_cash = torch.cat(
            [weights, torch.tensor([0.0])]
        )  # Add cash weight (set to 0)

        self.weights = weights_with_cash

    def act(self, state):
        """
        Returns the full action vector (including cash weight).

        Parameters:
            state: The current state of the environment.

        Returns:
            torch.Tensor: Action vector.
        """
        if self.weights is None:
            raise ValueError("Agent has not been trained yet. Call `train()` first.")
        return self.weights.unsqueeze(0)

    def _save_impl(self, path):
        """
        Saves the ClassicOnePeriodMarkovitzAgent's model to a file.

        Parameters:
            path (str): Path to save the model.
        """
        name = self.__class__.__name__
        if not path.endswith(".pt"):
            path = f"{path}/{name}.pt"
        else:
            path = path.replace(".pt", f"_{name}.pt")

        print(f"Saving ClassicOnePeriodMarkovitzAgent to {path}")

        # Save the weights and config
        torch.save({"weights": self.weights, "config": self.config}, path)

    def load(self, path):
        """
        Loads the ClassicOnePeriodMarkovitzAgent's model from a file.

        Parameters:
            path (str): Path to load the model from.
        """
        # print(f"Loading ClassicOnePeriodMarkovitzAgent from {path}")
        name = self.__class__.__name__
        if not path.endswith(".pt"):
            path = f"{path}/{name}.pt"
        else:
            path = path.replace(".pt", f"_{name}.pt")
        checkpoint = torch.load(path)
        self.weights = checkpoint["weights"]
        self.config = checkpoint["config"]

    @staticmethod
    def get_hyperparameter_space():
        """
        Returns the hyperparameters of the ClassicOnePeriodMarkovitzAgent.

        Returns:
            dict: Hyperparameter space for the agent.
        """
        return qf.DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZ_HYPERPARAMETER_SPACE

    @staticmethod
    def get_default_config():
        """
        Returns the default configuration for the ClassicOnePeriodMarkovitzAgent.

        Returns:
            dict: Default configuration for the agent.
        """
        return qf.DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_CONFIG
