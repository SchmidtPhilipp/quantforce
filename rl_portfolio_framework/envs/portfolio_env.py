import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class PortfolioEnv(gym.Env):
    def __init__(self, data, initial_balance=1_000, verbosity=1, n_agents=1, shared_obs=True, shared_action=True):
        """
        Initialize the PortfolioEnv.

        Parameters:
            data (pd.DataFrame): The historical data for the assets.
            initial_balance (float): The initial balance for the portfolio.
            verbosity (int): Verbosity level for logging.
            n_agents (int): Number of agents.
            shared_obs (bool): Whether each agent sees the entire observation space.
            shared_action (bool): Whether each agent can trade on every asset.
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
        self.shared_obs = shared_obs
        self.shared_action = shared_action

        # Define action space
        if shared_action:
            self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        else:
            self.action_space = spaces.Box(low=0, high=1, shape=(1 + 1,), dtype=np.float32)  # One asset + cash

        # Define observation space
        if shared_obs:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.data.shape[1],),
                dtype=np.float32
            )
        else:
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.data.shape[1] // self.n_assets,),
                dtype=np.float32
            )

    def reset(self):
        self.current_step = 0
        self.balance = self.initial_balance
        return self._get_observation()

    def _get_observation(self):
        obs = self.data.iloc[self.current_step].values.astype(np.float32)
        if not self.shared_obs:
            obs = np.array_split(obs, self.n_agents)
        return obs

    def step(self, actions):
        if not self.shared_action:
            actions = np.array_split(actions, self.n_agents)

        rewards = []
        for i in range(self.n_agents):


            if self.n_agents > 1:
                if self.shared_action:
                    # If the agents share the action space then every agent can trade on every asset + cash
                    # which means that we get the same action vector shape from every agent 
                    # we therefore have to average their actions to get the final action vector
                    final_action = np.mean(actions, axis=0)

                    # normalize
                    final_action /= np.sum(final_action) + 1e-8

                    # assert
                    assert final_action.shape[0] == self.n_assets + 1, "Action vector must have the same shape as the number of assets + cash"
                    # Check the action vector sums to 1 use is close to avoid floating point errors
                    assert np.isclose(np.sum(final_action), 1), "Action vector must sum to 1"

                else: # If the agents do not share the action space then every agent can only trade on one asset + cash
                    # we therefore have to extract the first actions of the agents as the asset weight and the last action as the cash weight
                    # we then normalize the action vector to sum to 1
                    print(actions)
                    final_action = actions[i]

                    # normalize
                    final_action /= np.sum(final_action) + 1e-8

                    # assert
                    assert final_action.shape[0] == 2, "Action vector must have the shape of 2 (asset weight, cash weight)"
                    assert np.isclose(np.sum(final_action), 1), "Action vector must sum to 1"
            
            else:
                final_action = actions


            cash_weight = final_action[-1]
            asset_weights = final_action[:-1]


            # Extract only the Close prices (assumes second level of MultiIndex is "Close")
            old_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values
            self.current_step += 1
            done = self.current_step >= len(self.data)

            if done:
                reward = 0.0 if self.n_agents == 1 else [0.0] * self.n_agents
                obs = np.zeros(self.data.shape[1], dtype=np.float32)  # dummy obs with correct shape
                if self.verbosity > 0:
                    print("Episode finished!")
                return obs, reward, done, {}

            new_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values
            asset_returns = new_prices / (old_prices + 1e-15)

            if self.verbosity > 0:
                print(f"Old prices: {old_prices} | New prices: {new_prices} | Returns: {asset_returns}")
                print(f"Asset weights: {asset_weights} | Cash weight: {cash_weight}")

            portfolio_return = cash_weight * 1.0 + np.dot(asset_weights, asset_returns)

            self.balance *= (portfolio_return)
            reward = np.log(portfolio_return)
            rewards.append(reward)

            obs = self.data.iloc[self.current_step].values.astype(np.float32)
            if not self.shared_obs:
                obs = np.array_split(obs, self.n_agents)

            if self.verbosity > 0:
                print(f"Step: {self.current_step} | Reward: {reward:.4f} | Balance: {self.balance:.2f}")
                print(f"Action: {action} | Weights: {weights} | Prices: {new_prices}")
                print(f"Portfolio return: {portfolio_return:.4f} | Asset returns: {asset_returns}")
                print(f"Obs: {obs}")
                print(f"Observation shape: {obs.shape}")

        if self.n_agents == 1:
            return obs, rewards[0], done, {}
        else:
            return obs, rewards, done, {}

    def render(self, mode='human'):
        pass

    def close(self):
        pass


