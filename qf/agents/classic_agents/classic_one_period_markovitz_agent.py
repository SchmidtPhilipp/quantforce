import numpy as np
import torch
from qf.agents.agent import Agent
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt import risk_models
import pandas as pd
from collections import OrderedDict


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

    def train(self, episodes=0, total_timesteps=0, use_tqdm=True):
        """
        Trains the agent by calculating portfolio weights based on the selected target and risk model.
        """
        dataset = self.env.get_dataset()
        self.historical_data = dataset.get_data()
        self.historical_data = self.historical_data.xs('Close', level=1, axis=1)

        # Calculate returns as a pandas DataFrame
        if self.config["log_returns"]:
            returns = self.historical_data.pct_change().apply(lambda x: np.log(1 + x))  # Log returns
        else:
            returns = self.historical_data.pct_change()  # Simple returns

        # Drop NaN values caused by pct_change
        returns = returns.dropna()

        # Compute covariance matrix using the selected risk model
        try:
            cov_matrix = risk_models.risk_matrix(
                prices=returns,
                method=self.config["risk_model"], 
                returns_data=True, 
                log_returns=self.config["log_returns"]
            )
        except ValueError:
            raise ValueError(f"Unsupported risk model: {self.config['risk_model']}")
        
        # Fix the matrix if it is not positive definite
        if not np.all(np.linalg.eigvals(cov_matrix) > 0):
            cov_matrix = risk_models.fix_non_positive_definite(cov_matrix)

        mean_returns = returns.mean(axis=0)

        # Select optimization target
        n_assets = cov_matrix.shape[0]
        if self.config["target"] == "Tangency":
            ef = EfficientFrontier(mean_returns, cov_matrix)
            weights = ef.max_sharpe(risk_free_rate=self.config["risk_free_rate"])
        elif self.config["target"] == "MaxExpReturn":
            weights = pd.Series(0, index=mean_returns.index)  # Initialize weights as a pandas Series with asset labels
            weights[mean_returns.idxmax()] = 1.0  # Allocate all weight to the asset with the highest mean return
            weights = OrderedDict(weights.items())
        elif self.config["target"] == "MinVariance":
            ef = EfficientFrontier(mean_returns, cov_matrix)
            weights = ef.min_volatility()
        else:
            raise ValueError(f"Unsupported target: {self.config['target']}")

        # Convert weights to torch.Tensor and add cash weight
        weights = torch.tensor(list(weights.values()), dtype=torch.float32)
        weights_with_cash = torch.cat([weights, torch.tensor([0.0])])  # Add cash weight (set to 0)

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

    def save(self, path):
        """
        Saves the ClassicOnePeriodMarkovitzAgent's model to a file.

        Parameters:
            path (str): Path to save the model.
        """
        name = self.__class__.__name__
        if not path.endswith('.pt'):
            path = f"{path}/{name}.pt"
        else:
            path = path.replace('.pt', f'_{name}.pt')

        print(f"Saving ClassicOnePeriodMarkovitzAgent to {path}")

        # Save the weights and config
        torch.save({
            'weights': self.weights,
            'config': self.config
        }, path)

    def load(self, path):
        """
        Loads the ClassicOnePeriodMarkovitzAgent's model from a file.

        Parameters:
            path (str): Path to load the model from.
        """
        print(f"Loading ClassicOnePeriodMarkovitzAgent from {path}")
        checkpoint = torch.load(path)
        self.weights = checkpoint['weights']
        self.config = checkpoint['config']