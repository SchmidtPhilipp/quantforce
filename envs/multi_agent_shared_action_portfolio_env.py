from gymnasium import spaces
import numpy as np
from envs.base_portfolio_env import BasePortfolioEnv

class MultiAgentSharedActionPortfolioEnv(BasePortfolioEnv):
    """
    Multi-Agent Portfolio Environment with shared or non-shared observations and shared actions.

    Parameters:
        data (pd.DataFrame): Historical data for the assets.
        initial_balance (float): Starting balance for the portfolio.
        verbosity (int): Verbosity level for debug messages (0 = silent, 1 = detailed).
        n_agents (int): Number of agents in the environment.
        shared_obs (bool): Whether all agents share the same observation.
    """
    def __init__(self, data, initial_balance=1_000, verbosity=0, n_agents=2, shared_obs=True):
        super().__init__(data, initial_balance, verbosity, n_agents)
        self.shared_obs = shared_obs

        # Verify that n_assets == n_agents if shared_obs is False
        if not shared_obs and self.n_assets != self.n_agents:
            raise ValueError(
                f"Non-shared observations require n_assets == n_agents, but got n_assets={self.n_assets} and n_agents={self.n_agents}."
            )

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.data.shape[1],),
            dtype=np.float32
        )

        if self.verbosity > 0:
            print(f"MultiAgentSharedActionPortfolioEnv initialized with {self.n_agents} agents.")
            print(f"Shared Observations: {self.shared_obs}")

    def _get_observation(self):
        """
        Returns the current observation for the agents.

        Returns:
            obs (np.ndarray): If shared_obs is True, all agents receive the same observation (shape: [n_agents, n_assets]).
                              If shared_obs is False, each agent receives an individual observation corresponding to one asset
                              (shape: [n_agents, n_assets_per_agent]).
        """
        obs = self.data.iloc[self.current_step].values.astype(np.float32)

        if self.shared_obs:
            # All agents receive the same observation (all assets)
            shared_obs = np.tile(obs, (self.n_agents, 1))  # Repeat the same observation for all agents
            if self.verbosity > 0:
                print(f"Shared observation provided to all agents: {shared_obs}")
            return shared_obs
        else:
            # Each agent receives an observation corresponding to one asset
            asset_split = self.data.shape[1] // self.n_agents
            individual_obs = np.array([
                obs[i * asset_split:(i + 1) * asset_split] for i in range(self.n_agents)
            ])
            if self.verbosity > 0:
                print(f"Non-shared observations provided to agents: {individual_obs}")
            return individual_obs

    def step(self, actions):
        """
        Executes a step in the environment.

        Parameters:
            actions (np.ndarray): Array of actions from each agent (shape: [n_agents, action_dim]).

        Returns:
            obs (np.ndarray): Next observation(s) for the agents.
            rewards (np.ndarray): Rewards for each agent (shape: [n_agents]).
            done (bool): Whether the episode is finished.
            info (dict): Additional information.
        """
        # Compute the shared action as the mean of all agent actions
        final_action = np.mean(actions, axis=0)
        final_action = np.clip(final_action, 0, 1)
        final_action /= np.sum(final_action) + 1e-8

        cash_weight = final_action[-1]
        asset_weights = final_action[:-1]

        # Get prices for the current and next steps
        old_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values
        self.current_step += 1
        done = self.current_step >= len(self.data)

        if done:
            rewards = np.zeros(self.n_agents, dtype=np.float32)
            obs = np.zeros((self.n_agents, self.data.shape[1]), dtype=np.float32)
            if self.verbosity > 0:
                print("Episode finished!")
            return obs, rewards, done, {}

        new_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values

        # Calculate asset returns and portfolio return
        asset_returns = new_prices / (old_prices + 1e-15) - 1
        portfolio_return = cash_weight * 1.0 + np.dot(asset_weights, asset_returns)

        # Update balance and calculate rewards
        self.balance *= (1 + portfolio_return)
        reward = np.log(1 + portfolio_return) if portfolio_return > -1 else -np.inf
        rewards = np.full(self.n_agents, reward, dtype=np.float32)

        # Get the next observation(s)
        obs = self._get_observation()

        # Debugging information
        if self.verbosity > 0:
            print(f"Step: {self.current_step} | Rewards: {rewards} | Balance: {self.balance:.2f}")
            print(f"Actions: {actions} | Final Action: {final_action}")
            print(f"Old Prices: {old_prices} | New Prices: {new_prices}")
            print(f"Portfolio Return: {portfolio_return:.4f} | Asset Returns: {asset_returns}")
            print(f"Observation(s): {obs}")

        return obs, rewards, done, {}