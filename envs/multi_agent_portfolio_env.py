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
    def __init__(self, data, initial_balance=1_000, verbosity=0, n_agents=2, trade_cost_percent=0.0, trade_cost_fixed=0.0, device="cpu"):

        self.data = data
        self.data_iterator = iter(self.data)
        self.device = device

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
            obs (torch.Tensor): A tensor of observations for all agents.
        """
        current_observations = next(self.data_iterator).squeeze(0).to(self.device)  # Move to MPS
        current_observations = current_observations.flatten()
        current_observations = current_observations.repeat(self.n_agents, 1)

        current_actions = torch.tensor(self.last_actions, dtype=torch.float32, device=self.device)
        current_holdings = torch.tensor(self.actor_asset_holdings, dtype=torch.float32, device=self.device)
        current_cash = torch.tensor(self.actor_cash, dtype=torch.float32, device=self.device).unsqueeze(1)

        # Combine the data with the last actions
        obs = torch.cat([current_observations, current_actions, current_holdings, current_cash], dim=1)

        return obs

    def step(self, actions):
        """
        Executes a step in the environment.

        Parameters:
            actions (torch.Tensor): Tensor of actions from each agent (shape: [n_agents, action_dim]).

        Returns:
            obs (torch.Tensor): Next observation(s) for the agents.
            rewards (torch.Tensor): Rewards for each agent (shape: [n_agents]).
            done (bool): Whether the episode is finished.
            info (dict): Additional information.
        """
        actions = torch.clamp(actions.to(self.device), 0, 1)  # Move actions to MPS

        if self.n_agents > 1:
            actions = actions / actions.sum(dim=1, keepdim=True)
        else:
            actions = actions / actions.sum()

        self.last_actions = actions.cpu().numpy()  # Save actions for the next observation

        # Get prices for the current and next steps
        old_prices = torch.tensor(
            self.data.dataset.data.xs("Close", axis=1, level=1).iloc[self.current_step + self.window_size - 1].values,
            dtype=torch.float32,
            device=self.device
        )

        self.current_step += 1
        done = self.current_step >= len(self.data)

        if done:
            reward = torch.zeros(self.n_agents, device=self.device)
            obs = torch.zeros((self.n_agents, *self.observation_space.shape), dtype=torch.float32, device=self.device)
            return obs, reward, done, {}

        new_prices = torch.tensor(
            self.data.dataset.data.xs("Close", axis=1, level=1).iloc[self.current_step + self.window_size - 1].values,
            dtype=torch.float32,
            device=self.device
        )

        rewards = torch.zeros(self.n_agents, device=self.device)
        for i in range(self.n_agents):
            action = actions[i]
            actor_cash_weight = action[-1]
            actor_asset_weight = action[:-1]

            # Convert actor_asset_holdings[i] to a tensor
            actor_asset_holdings_tensor = torch.tensor(self.actor_asset_holdings[i], dtype=torch.float32, device=self.device)

            current_actor_portfolio_value = self.actor_cash[i] + torch.sum(actor_asset_holdings_tensor * old_prices)

            # Calculate target distribution
            target_cash = current_actor_portfolio_value * actor_cash_weight
            target_asset_values = current_actor_portfolio_value * actor_asset_weight

            # Calculate target asset numbers
            target_asset_numbers = torch.floor(target_asset_values / (new_prices + 1e-10))

            # Calculate differences and trade costs
            asset_differences = target_asset_numbers - actor_asset_holdings_tensor
            buy_costs = torch.sum(torch.maximum(asset_differences, torch.tensor(0.0, device=self.device)) * new_prices)
            sell_proceeds = torch.sum(torch.maximum(-asset_differences, torch.tensor(0.0, device=self.device)) * new_prices)
            trade_costs_percent = torch.sum(torch.abs(asset_differences) * new_prices * self.trade_cost_percent)
            trade_costs_fixed = torch.sum((asset_differences != 0).float()) * self.trade_cost_fixed
            total_trade_costs = trade_costs_percent + trade_costs_fixed

            # Update cash and asset holdings
            self.actor_cash[i] += sell_proceeds - buy_costs - total_trade_costs
            self.actor_asset_holdings[i] = target_asset_numbers.cpu().numpy()  # Convert back to numpy for storage

            # Calculate portfolio value and reward
            portfolio_value = self.actor_cash[i] + torch.sum(target_asset_numbers * new_prices)
            reward = portfolio_value - self.actor_balance[i]
            rewards[i] = reward
            self.actor_balance[i] = portfolio_value

        # Update total balance and asset holdings
        self.cash = torch.sum(torch.tensor(self.actor_cash, dtype=torch.float32, device=self.device))
        self.asset_holdings = torch.sum(torch.tensor(self.actor_asset_holdings, dtype=torch.float32, device=self.device), dim=0)
        self.balance = torch.sum(torch.tensor(self.actor_balance, dtype=torch.float32, device=self.device))

        obs = self._get_observation()
        return obs, rewards, done, {}

    def reset(self):
        """
        Resets the environment and returns the first observation.
        """
        self.current_step = 0
        self.cash = self.initial_balance
        self.asset_holdings = np.zeros(self.n_assets, dtype=np.float32)
        self.balance = self.initial_balance
        self.data_iterator = iter(self.data)

        # Reset actor cash and asset holdings for each agent
        self.actor_cash = np.zeros(self.n_agents, dtype=np.float32)
        self.actor_cash.fill(self.initial_balance // self.n_agents)
        self.actor_asset_holdings = np.zeros((self.n_agents, self.n_assets), dtype=np.float32)
        self.actor_balance = np.zeros(self.n_agents, dtype=np.float32)
        self.actor_balance.fill(self.initial_balance // self.n_agents)

        # Reset last actions to all cash
        self.last_actions = np.zeros((self.n_agents, self.n_assets + 1), dtype=np.float32)
        self.last_actions[:, -1] = 1  # Set the last entry (cash) to 1

        obs = self._get_observation()
        return obs

    def get_timesteps(self):
        """
        Returns the number of timesteps in the environment.
        """
        return len(self.data)