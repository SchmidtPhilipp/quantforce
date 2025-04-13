from gymnasium import spaces
import numpy as np
from envs.base_portfolio_env import BasePortfolioEnv
import torch

class MultiAgentPortfolioEnv(BasePortfolioEnv):
    """
    Multi-Agent Portfolio Environment with shared actions and shared or non-shared observations.

    Parameters:
        data (pd.DataFrame): Historical data for the assets.
        initial_balance (float): Starting balance for the portfolio.
        verbosity (int): Verbosity level for debug messages (0 = silent, 1 = detailed).
        n_agents (int): Number of agents in the environment.
        shared_obs (bool): Whether all agents share the same observation.
    """
    def __init__(self, data, initial_balance=1_000, verbosity=0, n_agents=2, trade_cost_percent=0.0, trade_cost_fixed=0.0):
        super().__init__(data, initial_balance, verbosity, n_agents)
        self.trade_cost_percent = trade_cost_percent
        self.trade_cost_fixed = trade_cost_fixed

        # Initialize shared cash and asset holdings
        self.cash = initial_balance
        self.asset_holdings = np.zeros(self.n_assets)
        self.balance = initial_balance

        # Initialize actor cash and asset holdings for each agent
        self.actor_cash = np.zeros(self.n_agents)
        self.actor_cash.fill(initial_balance//self.n_agents)
        self.actor_asset_holdings = np.zeros((self.n_agents, self.n_assets))
        self.actor_balance = np.zeros(self.n_agents)
        self.actor_balance.fill(initial_balance//self.n_agents)

        # Shared observation: all assets and indicators
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.data.shape[1],),
            dtype=np.float32
        )

        # Action space: weights for each asset and cash
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)

        if self.verbosity > 0:
            print(f"MultiAgentPortfolioEnv initialized with {self.n_agents} agents.")

    def _get_observation(self):
        """
        Returns the current observation for the agents.

        Returns:
            obs (list of np.ndarray): A list of observations, one for each agent.
        """
        obs = torch.tensor(self.data.iloc[self.current_step].values, dtype=torch.float32)

        # All agents receive the same observation
        return obs.repeat(self.n_agents, 1)  # List of identical observations


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
        actions = np.array(actions)
        # Normalize the actions
        actions = np.clip(actions, 0, 1)

        if self.n_agents > 1:
            actions = actions / np.sum(actions, axis=1, keepdims=True)
        else: 
            actions = actions / np.sum(actions)



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



        rewards = []
        for i in range(self.n_agents):
            if self.n_agents > 1:
                action = actions[i]
            else:
                action = actions[i]

            actor_cash_weight = action[-1]
            actor_asset_weight = action[:-1]

            current_actor_portfolio_value = self.actor_cash[i] + np.sum(self.actor_asset_holdings[i] * old_prices)

            # Calculate target distribution
            target_cash = current_actor_portfolio_value * actor_cash_weight
            target_asset_values = current_actor_portfolio_value * actor_asset_weight

            # Calculate target asset numbers
            target_asset_numbers = np.floor(target_asset_values / (new_prices + 1e-10))

            # Calculate differences and trade costs
            asset_differences = target_asset_numbers - self.actor_asset_holdings[i]
            buy_costs = np.sum(np.maximum(asset_differences, 0) * new_prices)
            sell_proceeds = np.sum(np.maximum(-asset_differences, 0) * new_prices)
            trade_costs_percent = np.sum(np.abs(asset_differences) * new_prices * self.trade_cost_percent)
            trade_costs_fixed = np.sum(asset_differences != 0) * self.trade_cost_fixed
            total_trade_costs = trade_costs_percent + trade_costs_fixed

            # Update cash and asset holdings
            self.actor_cash[i] += sell_proceeds - buy_costs - total_trade_costs
            self.actor_asset_holdings[i] = target_asset_numbers

            # Calculate portfolio value and reward
            portfolio_value = self.actor_cash[i] + np.sum(self.actor_asset_holdings[i] * new_prices)

            # Calculate the reward for each of the agents
            reward = portfolio_value - self.actor_balance[i]
            rewards.append(reward)
            self.actor_balance[i] = portfolio_value

        # Calculate the total balance and asset holdings
        self.cash = np.sum(self.actor_cash)
        self.asset_holdings = np.sum(self.actor_asset_holdings, axis=0)
        self.balance = np.sum(self.actor_balance)
        # Calculate the overall reward
        reward = np.mean(rewards)


        # Debugging information
        if self.verbosity > 0:
            print(f"Step: {self.current_step} | Reward: {reward:.4f} | Balance: {self.balance:.2f}")
            print("-" * 50)
            for i in range(self.n_agents):
                print(f"Agent {i} Actions: {actions[i]}")
                print(f"Agent {i}: Cash: {self.actor_cash[i]:.2f} | Asset Holdings: {self.actor_asset_holdings[i]}")
                print(f"Agent {i} Portfolio Value: {self.actor_balance[i]:.2f}")
                print(f"Agent {i} Reward: {rewards[i]:.4f}")
                print("-" * 50)

            print(f"Total Cash: {self.cash:.2f} | Total Asset Holdings: {self.asset_holdings}")
            print("-" * 50)


        # Get the next observation(s)
        obs = self._get_observation()

        return obs, np.array(rewards), done, {}

    def reset(self):
        """
        Resets the environment and returns the first observation.
        """
        self.current_step = 0
        self.cash = self.initial_balance
        self.asset_holdings = np.zeros(self.n_assets)
        self.balance = self.initial_balance

        # Reset actor cash and asset holdings for each agent
        self.actor_cash = np.zeros(self.n_agents)
        self.actor_cash.fill(self.initial_balance//self.n_agents)
        self.actor_asset_holdings = np.zeros((self.n_agents, self.n_assets))
        self.actor_balance = np.zeros(self.n_agents)
        self.actor_balance.fill(self.initial_balance//self.n_agents)
        

        obs = self._get_observation()
        return obs