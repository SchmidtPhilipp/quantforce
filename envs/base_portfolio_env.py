import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class BasePortfolioEnv(gym.Env):
    def __init__(self, data, initial_balance=1_000, verbosity=0, n_agents=1):
        """
        Initialize the BasePortfolioEnv.

        Parameters:
            data (pd.DataFrame): The historical data for the assets.
            initial_balance (float): The initial balance for the portfolio.
            verbosity (int): Verbosity level for logging.
            n_agents (int): Number of agents.
        """
        assert isinstance(data.columns, pd.MultiIndex), "DataFrame must have MultiIndex columns (ticker, feature)"

        self.data = data
        self.assets = sorted(data.columns.get_level_values(0).unique())
        self.n_assets = len(self.assets)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.verbosity = verbosity
        self.n_agents = n_agents

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        return self._get_observation()

    def _get_observation(self):
        raise NotImplementedError

    def step(self, actions):
        raise NotImplementedError

    def render(self, mode='human'):
        pass

    def close(self):
        pass