import os
from collections import OrderedDict
from typing import Optional

import numpy as np
import pandas as pd
import torch
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier

import qf
from qf.agents.agent import Agent
from qf.agents.classic_agents.estimators.estimators import (
    estimate_expected_rate_of_returns_and_covariance,
)
from qf.agents.classic_agents.estimators.ml_brownian_motion_logreturn import (
    MultiAssetBrownianMotionLogReturn,
)
from qf.agents.config.classic_agents.classic_one_period_markowitz_agent_config import (
    ClassicOnePeriodMarkowitzAgentConfig,
)


class ClassicOnePeriodMarkowitzAgent(Agent):
    def __init__(
        self, env, config: Optional[ClassicOnePeriodMarkowitzAgentConfig] = None
    ):
        """
        Initializes the ClassicOnePeriodMarkowitzAgent with the given environment.

        Parameters:
            env: The environment in which the agent operates.
            config (dict, optional): Configuration dictionary for the agent.
        """
        self.config = (
            config or ClassicOnePeriodMarkowitzAgentConfig.get_default_config()
        )
        super().__init__(env=env, config=self.config)

        self.historical_data = None
        self.weights = None  # torch.Tensor of shape (n_assets + 1,)

    def train(
        self,
        episodes=0,
        total_timesteps=0,
        use_tqdm=True,
        save_best=True,
        eval_env=None,
        eval_every_n_steps=None,
        n_eval_episodes=1,
        print_eval_metrics=False,
    ):
        """
        Trains the agent by calculating portfolio weights based on the selected target and risk model.
        """

        # We reload the data with "shrinkage" imputation method to ensure that the data is complete and std dev is not biased
        data_config = self.env.env_config.data_config.copy()
        data_config.indicators = ["Close"]
        data_config.backfill_method = "shrinkage"

        self.historical_data = qf.get_data(data_config=data_config)

        # Use the simple estimation function
        expected_return, cov_matrix = estimate_expected_rate_of_returns_and_covariance(
            prices=self.historical_data,
            method=self.config.risk_model,
            frequency=data_config.n_trading_days,
            delta_t=data_config.interval,
            log_returns=self.config.log_returns,
        )

        # TODO: I think we have to convert them to linear returns here
        # if self.config.log_returns or self.config.risk_model == "ML_brownian_motion_logreturn":
        #    # Then we have to convert them to linear returns
        #    expected_return, cov_matrix = MultiAssetBrownianMotionLogReturn.estimate_linear_return_expectation_and_covariance(expected_return, cov_matrix, T=1)

        column_names = expected_return.index

        if isinstance(expected_return, pd.Series):
            # convert to tensor
            expected_return = torch.tensor(expected_return.values, dtype=torch.float32)
            cov_matrix = torch.tensor(cov_matrix.values, dtype=torch.float32)

        expected_return, cov_matrix = (
            MultiAssetBrownianMotionLogReturn.estimate_linear_return_expectation_and_covariance(
                expected_return, cov_matrix, T=1
            )
        )

        if isinstance(expected_return, torch.Tensor):
            # convert to pandas series
            expected_return = pd.Series(expected_return.numpy(), index=column_names)
            cov_matrix = pd.DataFrame(
                cov_matrix.numpy(), index=column_names, columns=column_names
            )

        # Fix the matrix if it is not positive definite
        if not np.all(np.linalg.eigvals(cov_matrix) > 0):
            cov_matrix = risk_models.fix_nonpositive_semidefinite(
                cov_matrix, "spectral"
            )

        # Select optimization target
        n_assets = cov_matrix.shape[0]
        if self.config.target == "Tangency":
            ef = EfficientFrontier(expected_return, cov_matrix)
            weights = ef.max_sharpe(risk_free_rate=self.config.risk_free_rate)
        elif self.config.target == "MaxExpReturn":
            weights = pd.Series(
                0, index=expected_return.index
            )  # Initialize weights as a pandas Series with asset labels
            weights[expected_return.idxmax()] = (
                1.0  # Allocate all weight to the asset with the highest mean return
            )
            weights = OrderedDict(weights.items())
        elif self.config.target == "MinVariance":
            ef = EfficientFrontier(expected_return, cov_matrix)
            weights = ef.min_volatility()
        else:
            raise ValueError(f"Unsupported target: {self.config.target}")

        # Convert weights to torch.Tensor and add cash weight
        weights = torch.tensor(list(weights.values()), dtype=torch.float32)
        weights_with_cash = torch.cat(
            [weights, torch.tensor([0.0])]
        )  # Add cash weight (set to 0)

        self.weights = weights_with_cash

        if save_best:
            best_model_path = os.path.join(self.env.save_dir, "best_model")
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
            self.save(best_model_path)

    def act(self, state):
        """
        Returns the full action vector (including cash weight).

        Parameters:
            state: The current state of the environment.

        Returns:
            numpy.ndarray: Action vector.
        """
        if self.weights is None:
            raise ValueError("Agent has not been trained yet. Call `train()` first.")

        # Convert tensor to numpy array and ensure it's on CPU
        if hasattr(self.weights, "cpu"):
            return self.weights.cpu().numpy()
        else:
            return self.weights.numpy()

    def _save_impl(self, path):
        """
        Saves the ClassicOnePeriodMarkowitzAgent's model to a file.

        Parameters:
            path (str): Path to save the model.
        """
        name = self.__class__.__name__
        if not path.endswith(".pt"):
            path = f"{path}/{name}.pt"
        else:
            path = path.replace(".pt", f"_{name}.pt")

        print(f"Saving ClassicOnePeriodMarkowitzAgent to {path}")

        # Save the weights and config
        torch.save({"weights": self.weights, "config": self.config}, path)

    def load(self, path):
        """
        Loads the ClassicOnePeriodMarkowitzAgent's model from a file.

        Parameters:
            path (str): Path to load the model from.
        """
        # print(f"Loading ClassicOnePeriodMarkowitzAgent from {path}")
        name = self.__class__.__name__
        if not path.endswith(".pt"):
            path = f"{path}/{name}.pt"
        else:
            path = path.replace(".pt", f"_{name}.pt")
        checkpoint = torch.load(path, weights_only=False)
        self.weights = checkpoint["weights"]
        self.config = checkpoint["config"]

    @staticmethod
    def get_hyperparameter_space():
        """
        Returns the hyperparameters of the ClassicOnePeriodMarkowitzAgent.

        Returns:
            dict: Hyperparameter space for the agent.
        """
        return ClassicOnePeriodMarkowitzAgentConfig.get_hyperparameter_space()

    @staticmethod
    def get_default_config():
        """
        Returns the default configuration for the ClassicOnePeriodMarkowitzAgent.

        Returns:
            dict: Default configuration for the agent.
        """
        return ClassicOnePeriodMarkowitzAgentConfig.get_default_config()
