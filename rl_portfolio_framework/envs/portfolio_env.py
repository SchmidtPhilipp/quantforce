import gymnasium as gym
from gymnasium import spaces
import numpy as np

class PortfolioEnv(gym.Env):
    def __init__(self, data, initial_balance=1_000):
        self.data = data
        self.assets = list(data.columns)
        self.n_assets = len(self.assets)
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=0,
            high=np.inf,
            shape=(self.n_assets,),
            dtype=np.float32
        )

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        return self._get_observation()

    def _get_observation(self):
        return self.data.iloc[self.current_step].values.astype(np.float32)

    def step(self, action):
        weights = np.clip(action, 0, 1)
        weights /= np.sum(weights) + 1e-8  # ensure sum = 1

        cash_weight = weights[-1]
        asset_weights = weights[:-1]

        old_prices = self.data.iloc[self.current_step].values
        self.current_step += 1
        done = self.current_step >= len(self.data)

        if done:
            return old_prices, 0.0, True, {}

        new_prices = self.data.iloc[self.current_step].values
        asset_returns = new_prices / old_prices
        portfolio_return = cash_weight * 1.0 + np.dot(asset_weights, asset_returns)

        self.balance *= portfolio_return
        reward = np.log(portfolio_return)

        return new_prices.astype(np.float32), reward, done, {}

