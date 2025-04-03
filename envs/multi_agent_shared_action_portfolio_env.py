from gymnasium import spaces
import numpy as np
from envs.base_portfolio_env import BasePortfolioEnv

class MultiAgentSharedActionPortfolioEnv(BasePortfolioEnv):
    """
    Multi-Agent Portfolio Environment with shared actions and shared or non-shared observations.

    Parameters:
        data (pd.DataFrame): Historical data for the assets.
        initial_balance (float): Starting balance for the portfolio.
        verbosity (int): Verbosity level for debug messages (0 = silent, 1 = detailed).
        n_agents (int): Number of agents in the environment.
        shared_obs (bool): Whether all agents share the same observation.
    """
    def __init__(self, data, initial_balance=1_000, verbosity=0, n_agents=2, shared_obs=False, trade_cost_percent=0.0, trade_cost_fixed=0.0):
        super().__init__(data, initial_balance, verbosity, n_agents)
        self.shared_obs = shared_obs
        self.trade_cost_percent = trade_cost_percent
        self.trade_cost_fixed = trade_cost_fixed

        # Initialize shared cash and asset holdings
        self.cash = initial_balance
        self.asset_holdings = np.zeros(self.n_assets)
        self.balance = initial_balance

        # Set observation space based on shared_obs
        if self.shared_obs:
            # Shared observation: all assets and indicators
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(self.data.shape[1],),
                dtype=np.float32
            )
        else:
            # Non-shared observation: only one asset and its indicators per agent
            asset_split = self.data.shape[1] // self.n_agents
            self.observation_space = spaces.Box(
                low=-np.inf,
                high=np.inf,
                shape=(asset_split,),
                dtype=np.float32
            )

        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)

        if self.verbosity > 0:
            print(f"MultiAgentSharedActionPortfolioEnv initialized with {self.n_agents} agents.")
            print(f"Shared Observations: {self.shared_obs}")

    def _get_observation(self):
        """
        Returns the current observation for the agents.

        Returns:
            obs (list of np.ndarray): A list of observations, one for each agent.
        """
        obs = self.data.iloc[self.current_step].values.astype(np.float32)

        if self.shared_obs:
            # All agents receive the same observation
            return [obs for _ in range(self.n_agents)]  # List of identical observations
        else:
            # Each agent receives an observation corresponding to one asset
            asset_split = self.data.shape[1] // self.n_agents
            individual_obs = [
                obs[i * asset_split:(i + 1) * asset_split] for i in range(self.n_agents)
            ]
            return individual_obs  # List of individual observations

    def step(self, actions):
        """
        Executes a step in the environment.

        Parameters:
            actions (np.ndarray): Array of actions from each agent (shape: [n_agents, action_dim]).

        Returns:
            obs (list of np.ndarray): Next observation(s) for the agents.
            rewards (np.ndarray): Rewards for each agent (shape: [n_agents]).
            done (bool): Whether the episode is finished.
            info (dict): Additional information.
        """
        # Aggregate actions (e.g., average across all agents)
        collective_action = np.mean(actions, axis=0)
        max_action = np.max(collective_action)  # For numerical stability
        exp_action = np.exp(collective_action - max_action)  # Subtract max_action for stability
        weights = exp_action / np.sum(exp_action)  # Softmax normalization

        cash_weight = weights[-1]
        asset_weights = weights[:-1]

        # Get prices for the current and next steps
        old_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values
        self.current_step += 1
        done = self.current_step >= len(self.data)

        if done:
            reward = 0.0
            obs = [np.zeros(self.observation_space.shape, dtype=np.float32) for _ in range(self.n_agents)]
            if self.verbosity > 0:
                print("Episode finished!")
            return obs, [reward] * self.n_agents, done, {}

        new_prices = self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values

        # Calculate the current portfolio value
        current_portfolio_value = self.cash + np.sum(self.asset_holdings * new_prices)

        # Calculate target distribution
        target_cash = current_portfolio_value * cash_weight
        target_asset_values = current_portfolio_value * asset_weights

        # Calculate target asset numbers
        target_asset_numbers = np.floor(target_asset_values / new_prices)

        # Calculate differences and trade costs
        asset_differences = target_asset_numbers - self.asset_holdings
        buy_costs = np.sum(np.maximum(asset_differences, 0) * new_prices)
        sell_proceeds = np.sum(np.maximum(-asset_differences, 0) * new_prices)
        trade_costs_percent = np.sum(np.abs(asset_differences) * new_prices * self.trade_cost_percent)
        trade_costs_fixed = np.sum(asset_differences != 0) * self.trade_cost_fixed
        total_trade_costs = trade_costs_percent + trade_costs_fixed

        # Update cash and asset holdings
        self.cash += sell_proceeds - buy_costs - total_trade_costs
        self.asset_holdings = target_asset_numbers

        # Calculate portfolio value and reward
        portfolio_value = self.cash + np.sum(self.asset_holdings * new_prices)
        reward = portfolio_value - self.balance
        self.balance = portfolio_value

        # Debugging information
        if self.verbosity > 0:
            print(f"Step: {self.current_step} | Reward: {reward:.4f} | Balance: {self.balance:.2f}")
            print(f"Action: {collective_action} | Weights: {weights} | Prices: {new_prices}")
            print(f"Target asset numbers: {target_asset_numbers} | Current asset holdings: {self.asset_holdings}")
            print(f"Remaining cash: {self.cash:.2f} | Trade costs: {total_trade_costs:.4f}")

        # Get the next observation(s)
        obs = self._get_observation()

        return obs, np.array([reward] * self.n_agents), done, {}

    def reset(self):
        """
        Resets the environment and returns the first observation.
        """
        self.current_step = 0
        self.cash = self.initial_balance
        self.asset_holdings = np.zeros(self.n_assets)
        self.balance = self.initial_balance
        obs = self._get_observation()
        return obs