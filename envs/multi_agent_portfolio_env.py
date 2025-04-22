from gymnasium import spaces
import numpy as np
import torch
import gymnasium as gym

class MultiAgentPortfolioEnv(gym.Env):
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

        self.data = data
        self.data_iterator = iter(self.data)

        self.n_assets = len(data.dataset.data.columns.get_level_values(0).unique())
        self.window_size = data.dataset.window_size
        self.initial_balance = initial_balance
        self.balance = initial_balance
        self.current_step = 0
        self.verbosity = verbosity
        self.n_agents = n_agents
    
        self.trade_cost_percent = trade_cost_percent
        self.trade_cost_fixed = trade_cost_fixed

        # Initialize shared cash and asset holdings
        self.cash = initial_balance
        self.asset_holdings = np.zeros(self.n_assets)
        self.balance = initial_balance

        # Initialize actor cash and asset holdings for each agent
        self.actor_cash = np.zeros((self.n_agents,1)).fill(initial_balance // self.n_agents)
        self.actor_asset_holdings = np.zeros((self.n_agents, self.n_assets))
        self.actor_balance = np.zeros(self.n_agents).fill(initial_balance // self.n_agents)

        # Initialize last actions (all cash initially)
        self.last_actions = np.zeros((self.n_agents, self.n_assets + 1))
        self.last_actions[:, -1] = 1  # Set the last entry (cash) to 1

        self.n_external_observables = len(self.data.dataset.data.columns)*self.window_size 

        # Shared observation: all assets and indicators + actions
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.n_external_observables + self.n_assets + 1 + self.n_assets + 1,), dtype=np.float32)

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
        # Get the current data for the timestep
        #current_observations = torch.tensor(self.data.iloc[self.current_step].values, dtype=torch.float32)

        current_observations = next(self.data_iterator).squeeze(0) # remove the batch dimension
        current_observations = current_observations.flatten()
        current_observations = current_observations.repeat(self.n_agents, 1)

        current_actions = torch.tensor(self.last_actions, dtype=torch.float32)

        current_holdings = torch.tensor(self.actor_asset_holdings, dtype=torch.float32)
        current_cash = torch.tensor(self.actor_cash, dtype=torch.float32).unsqueeze(0)

        # Combine the data with the last actions
        obs = torch.cat([current_observations, current_actions, current_holdings, current_cash], dim=1)

        # All agents receive the same observation
        return obs

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
        # Normalize the actions
        actions = np.clip(np.array(actions), 0, 1)

        if self.n_agents > 1:
            actions = actions / np.sum(actions, axis=1, keepdims=True)
        else:
            actions = actions / np.sum(actions)

        # Save the actions for the next observation
        self.last_actions = actions

        # Get prices for the current and next steps
        old_prices = self.data.dataset.data.xs("Close", axis=1, level=1).iloc[self.current_step+self.window_size-1].values

        self.current_step += 1
        done = self.current_step >= len(self.data)

        if done:
            reward = 0.0
            obs = [np.zeros(self.observation_space.shape, dtype=np.float32) for _ in range(self.n_agents)]
            if self.verbosity > 0:
                print("Episode finished!")
            return obs, np.array([reward] * self.n_agents), done, {}

        new_prices = self.data.dataset.data.xs("Close", axis=1, level=1).iloc[self.current_step+self.window_size-1].values

        rewards = []
        for i in range(self.n_agents):
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

        # Debugging information
        if self.verbosity > 0:
            print(f"Step: {self.current_step} | Rewards: {rewards} | Balance: {self.balance:.2f}")

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
        self.data_iterator = iter(self.data)

        # Reset actor cash and asset holdings for each agent
        self.actor_cash = np.zeros(self.n_agents)
        self.actor_cash.fill(self.initial_balance // self.n_agents)
        self.actor_asset_holdings = np.zeros((self.n_agents, self.n_assets))
        self.actor_balance = np.zeros(self.n_agents)
        self.actor_balance.fill(self.initial_balance // self.n_agents)

        # Reset last actions to all cash
        self.last_actions = np.zeros((self.n_agents, self.n_assets + 1))
        self.last_actions[:, -1] = 1  # Set the last entry (cash) to 1

        obs = self._get_observation()
        return obs

    def get_timesteps(self):
        """
        Returns the number of timesteps in the environment.
        """
        return len(self.data)