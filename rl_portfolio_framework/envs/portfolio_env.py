import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    def __init__(self, data, initial_balance=1_000, verbosity=0):
        assert isinstance(data.columns, pd.MultiIndex), "DataFrame must have MultiIndex columns (ticker, feature)"

        self.data = data
        self.assets = sorted(data.columns.get_level_values(0).unique())
        self.n_assets = len(self.assets)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        # Action space: n_assets + 1 (last one is for cash)
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        self.verbosity = verbosity

        # Observation space: all features for all assets (flattened)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.data.shape[1],),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        return self._get_observation()

    def _get_observation(self):
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        return obs

    def step(self, action):
        weights = np.clip(action, 0, 1)
        weights /= np.sum(weights) + 1e-8

        cash_weight = weights[-1]
        asset_weights = weights[:-1]

        # Extrahiere nur die Close-Preise (assumes second level of MultiIndex is "Close")
        old_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values
        self.current_step += 1
        done = self.current_step >= len(self.data)

        if done:
            reward = 0.0
            obs = np.zeros(self.data.shape[1], dtype=np.float32)  # dummy obs with correct shape
            if self.verbosity > 0:
                print("Episode finished!")

            return obs, reward, done, {}

        new_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values
        asset_returns = new_prices / (old_prices + 1e-15)
        portfolio_return = cash_weight * 1.0 + np.dot(asset_weights, asset_returns)

        self.balance *= portfolio_return
        reward = np.log(portfolio_return)

        obs = self.data.iloc[self.current_step].values.astype(np.float32)

        if self.verbosity > 0:
            print(f"Step: {self.current_step} | Reward: {reward:.4f} | Balance: {self.balance:.2f}")
            print(f"Action: {action} | Weights: {weights} | Prices: {new_prices}")
            print(f"Portfolio return: {portfolio_return:.4f} | Asset returns: {asset_returns}")
            print(f"Obs: {obs}")
            print(f"Observation shape: {obs.shape}")
        return obs, reward, done, {}


