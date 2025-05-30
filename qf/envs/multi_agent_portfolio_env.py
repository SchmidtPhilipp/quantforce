from gymnasium import spaces
import torch
import gymnasium as gym
import numpy as np

from qf.utils.tracker.tracker import Tracker


class MultiAgentPortfolioEnv(gym.Env):
    """
    Multi-Agent Portfolio Environment with shared actions and shared or non-shared observations.

    Parameters:
        data (pd.DataFrame): Historical data for the assets.
        initial_balance (float): Starting balance for the portfolio.
        verbosity (int): Verbosity level for debug messages (0 = silent, 1 = detailed).
        n_agents (int): Number of agents in the environment.
        trade_cost_percent (float): Proportional trading cost.
        trade_cost_fixed (float): Fixed trading cost per transaction.
        device (str): Device to use for computations ("cpu" or "cuda").
    """
    def __init__(self, data, initial_balance=1_000, verbosity=0, n_agents=2, trade_cost_percent=0.0, trade_cost_fixed=0.0, device="cpu", reward_function=None):
        self.data = data
        self.data_iterator = iter(self.data)
        self.device = device
        self.asset_labels = data.dataset.data.columns.get_level_values(0).unique().to_list() # Unique because we have multiple indicators
        self.n_assets = len(data.dataset.data.columns.get_level_values(0).unique())
        self.obs_window_size = data.dataset.window_size
        self.initial_balance = initial_balance
        self.current_step = 0
        self.verbosity = verbosity
        self.n_agents = n_agents
        self.trade_cost_percent = trade_cost_percent
        self.trade_cost_fixed = trade_cost_fixed

        # Reward function can be "linear_rate_of_return", "log_return", or "absolute_return", or "sharpe_ratio_wX" where X is the time horizon the sharpe ratio is calculated over
        self.reward_function = reward_function
        
        # check if the reward function begins with "sharpe_ratio_w"
        if self.reward_function is not None and self.reward_function.startswith("sharpe_ratio_w"):
            # Extract the time horizon from the reward function
            try:
                self.sharpe_time_horizon = int(self.reward_function.split("_w")[1])
            except ValueError:
                raise ValueError(f"Invalid reward function format: {self.reward_function}. Expected format 'sharpe_ratio_wX' where X is an integer.")
            self.reward_function = "sharpe_ratio"


        # Initialize actor cash and asset holdings for each agent as Torch Tensors
        self.current_cash_vector = torch.full((self.n_agents,), initial_balance / self.n_agents, dtype=torch.float32, device=self.device)
        self.current_portfolio_matrix = torch.zeros((self.n_agents, self.n_assets), dtype=torch.float32, device=self.device)
        self.current_portfolio_value = torch.full((self.n_agents,), initial_balance / self.n_agents, dtype=torch.float32, device=self.device)


        # Initialize last actions (all cash initially)
        self.last_actions = torch.zeros((self.n_agents, self.n_assets + 1), dtype=torch.float32, device=self.device)
        self.last_actions[:, -1] = 1  # Set the last entry (cash) to 1

        # Observation space: all assets and indicators + actions
        self.n_external_observables = len(self.data.dataset.data.columns) * self.obs_window_size
        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(self.n_external_observables + self.n_assets + 1 + self.n_assets + 1,),
            dtype=np.float32
        )

        # Action space: weights for each asset and cash
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32)

        if self.verbosity > 0:
            print(f"MultiAgentPortfolioEnv initialized with {self.n_agents} agents.")

        


        self.tracker = None

    def _get_observation(self):
        """
        Returns the current observation for the agents.

        Returns:
            obs (torch.Tensor): A tensor of observations for all agents.
        """
        current_observations = next(self.data_iterator).squeeze(0).to(self.device)  # Move to device
        current_observations = current_observations.flatten()
        current_observations = current_observations.repeat(self.n_agents, 1)

        current_actions = self.last_actions
        current_holdings = self.current_portfolio_matrix
        current_cash = self.current_cash_vector.unsqueeze(1)

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
        actions = torch.clamp(actions.to(self.device), 0, 1)  # Ensure actions are in [0, 1]
        actions = actions / actions.sum(dim=1, keepdim=True)  # Normalize actions

        self.last_actions = actions  # Save actions for the next observation

        # Get prices for the current and next steps
        old_prices = torch.tensor(
            self.data.dataset.data.xs("Close", axis=1, level=1).iloc[self.current_step + self.obs_window_size - 1].values,
            dtype=torch.float32,
            device=self.device
        )

        self.current_step += 1
        done = self.current_step >= len(self.data)
        

        if done:
            done = torch.full((self.n_agents, 1), float(done), dtype=torch.float32, device=self.device)
            rewards = torch.zeros(self.n_agents, device=self.device)
            obs = torch.zeros((self.n_agents, *self.observation_space.shape), dtype=torch.float32, device=self.device)
            return obs, rewards, done, {}

        new_prices = torch.tensor(
            self.data.dataset.data.xs("Close", axis=1, level=1).iloc[self.current_step + self.obs_window_size - 1].values,
            dtype=torch.float32,
            device=self.device
        )
        done = torch.full((self.n_agents, 1), float(done), dtype=torch.float32, device=self.device)

        # 1. Use the current portfolio value
        current_portfolio_value_t = self.current_portfolio_value

        # 2. Split actions into asset weights and cash weights
        asset_weights = actions[:, :-1]
        cash_weights = actions[:, -1]

        # 3. Calculate target cash and asset values
        target_cash = current_portfolio_value_t * cash_weights
        target_asset_values = current_portfolio_value_t.unsqueeze(1) * asset_weights

        # 4. Calculate target asset holdings
        target_holdings = torch.floor(target_asset_values / (old_prices.unsqueeze(0) + 1e-10))

        # 5. Calculate differences in holdings
        delta_holdings = target_holdings - self.current_portfolio_matrix

        # 6. Calculate buy and sell costs
        buy_costs = torch.sum(torch.clamp(delta_holdings, min=0) * old_prices.unsqueeze(0), dim=1)
        sell_proceeds = torch.sum(torch.clamp(-delta_holdings, min=0) * old_prices.unsqueeze(0), dim=1)

        # 7. Calculate trading costs
        trade_costs_percent = torch.sum(torch.abs(delta_holdings) * old_prices.unsqueeze(0) * self.trade_cost_percent, dim=1)
        trade_costs_fixed = torch.sum((delta_holdings != 0).float(), dim=1) * self.trade_cost_fixed
        total_trade_costs = trade_costs_percent + trade_costs_fixed

        # 8. Update cash and holdings
        self.current_cash_vector = self.current_cash_vector + sell_proceeds - buy_costs - total_trade_costs
        self.current_portfolio_matrix = target_holdings

        # 9. Calculate new portfolio value
        current_portfolio_value_t1 = self.current_cash_vector + torch.sum(self.current_portfolio_matrix * new_prices, dim=1)

        # Update the portfolio value property
        self.current_portfolio_value = current_portfolio_value_t1

        # End the episode if the portfolio value is zero or negative
        if torch.any(current_portfolio_value_t1 <= 0):
            print("Portfolio value is zero or negative. Ending episode.")   
            done = torch.full((self.n_agents, 1), float(True), dtype=torch.float32, device=self.device)
            current_portfolio_value_t1[current_portfolio_value_t1 <= 0] = 0  # Set negative values to zero


        # 10. Calculate rewards
        if self.reward_function == "linear_rate_of_return":
            rewards = current_portfolio_value_t1/(current_portfolio_value_t + 1e-10) - 1
        elif self.reward_function == "log_return":
            rewards = current_portfolio_value_t1 / (current_portfolio_value_t + 1e-10)
            if rewards < 0:
                raise ValueError("Negative reward detected. log_return is not defined for negative values. You may remove the costs of the transaction.")
            rewards = torch.log(rewards)
        elif self.reward_function == "absolute_return":
            rewards = current_portfolio_value_t1 - current_portfolio_value_t

        elif self.reward_function == "sharpe_ratio":
            # We calculate the Sharpe ratio based on the current portfolio and the past stock prices
            # Update the past portfolio val
            adjusted_current_step = self.current_step + self.obs_window_size - 1
            past_price_matrix = torch.tensor(
                self.data.dataset.data.xs("Close", axis=1, level=1).iloc[(adjusted_current_step - self.sharpe_time_horizon):adjusted_current_step].values,
                dtype=torch.float32,
                device=self.device
            ) # shape: [past_steps, n_assets]
            self.current_portfolio_matrix # shape: [n_agents, n_assets]
            past_portfolio_values = self.current_portfolio_matrix @ past_price_matrix.T # shape: [n_agents, past_steps]
            past_portfolio_linear_returns = past_portfolio_values[:, 1:] / (past_portfolio_values[:, :-1] + 1e-10) - 1
            past_portfolio_log_returns = torch.log(past_portfolio_values[:, 1:] / (past_portfolio_values[:, :-1] + 1e-10))

            past_returns = past_portfolio_linear_returns

            # Calculate the Sharpe ratio
            mean_return = torch.mean(past_returns, dim=-1) # mean of the time dimension
            std_return = torch.std(past_returns, dim=-1) # std of the time dimension
            sharpe_ratio = (mean_return) / (std_return + 1e-10)

            # if any entry of the sharpe ratio is NaN or inf, we set it to 0
            if torch.any(torch.isnan(sharpe_ratio)) or torch.any(torch.isinf(sharpe_ratio)):
                print("Sharpe ratio contains NaN or inf values.")

            rewards = sharpe_ratio

        else:
            raise ValueError(f"Unknown reward function: {self.reward_function}. Supported functions are 'linear_rate_of_return', 'log_return', and 'absolute_return'.")

        # Get the next observation
        obs = self._get_observation()
        return obs, rewards, done, {}

    def reset(self):
        """
        Resets the environment and returns the first observation.
        """
        self.current_step = 0
        self.data_iterator = iter(self.data)
        self.current_cash_vector = torch.full((self.n_agents,), self.initial_balance / self.n_agents, dtype=torch.float32, device=self.device)
        self.current_portfolio_matrix = torch.zeros((self.n_agents, self.n_assets), dtype=torch.float32, device=self.device)
        self.current_portfolio_value = torch.full((self.n_agents,), self.initial_balance / self.n_agents, dtype=torch.float32, device=self.device)

        self.last_actions = torch.zeros((self.n_agents, self.n_assets + 1), dtype=torch.float32, device=self.device)
        self.last_actions[:, -1] = 1  # Set the last entry (cash) to 1

        obs = self._get_observation()
        return obs

    def get_cash(self):
        """
        Calculates the total cash across all agents.

        Returns:
            torch.Tensor: Total cash (scalar).
        """
        return torch.sum(self.current_cash_vector)

    def get_portfolio(self):
        """
        Calculates the total asset holdings across all agents.

        Returns:
            torch.Tensor: Total asset holdings (vector of size [n_assets]).
        """
        return torch.sum(self.current_portfolio_matrix, dim=0)

    def get_portfolio_value(self):
        """
        Calculates the total portfolio balance across all agents.

        Parameters:
            prices (torch.Tensor): Current prices of the assets (shape: [n_assets]).

        Returns:
            torch.Tensor: Total portfolio balance (scalar).
        """
        return torch.sum(self.current_portfolio_value)
    
    def get_timesteps(self):
        """
        Returns the number of timesteps in the environment.
        """
        return len(self.data)
    
    def register_tracker(self, tracker=None):
        """
        Registers a tracker for logging and monitoring.

        Parameters:
            tracker (object): Tracker object for logging.
        """

        if self.tracker!=None:
            return

        if tracker is None:
            tracker = Tracker(timesteps=self.get_timesteps(), tensorboard_prefix=f"")

        self.tracker= tracker

        # Register tracked values with custom axis labels
        tracker.register_value("rewards", shape=(self.n_agents,), description="Rewards per agent", dimensions=["timesteps", "agents"], labels=[range(self.n_agents)])
        tracker.register_value("actions",shape=(self.n_agents, self.n_assets + 1),description="Actions per agent",dimensions=["timesteps", "agents", "assets"],labels=[range(self.n_agents), self.asset_labels + ["cash"]])
        #tracker.register_value("asset_holdings",shape=(self.n_agents, self.n_assets),description="Asset holdings per agent",dimensions=["timesteps", "agents", "assets"],labels=[range(self.n_agents), self.asset_labels])
        tracker.register_value("balance", shape=(1,), description="Environment balance", dimensions=["timesteps"], labels=[["balance"]])
        #tracker.register_value("actor_balance",shape=(self.n_agents,),description="Actor balances",dimensions=["timesteps", "agents"],labels=[range(self.n_agents)])

    def record_data(self, action=None, reward=None):

        if self.tracker is None:
            raise ValueError("Tracker is not registered. Please register a tracker before recording data.")

        values_to_record = self.tracker.tracked_values.keys()

        values =  {}

        if "rewards" in values_to_record and reward is not None:
            values["rewards"] = reward.to(self.device)
        if "actions" in values_to_record and action is not None:
            values["actions"] = action.to(self.device)
        if "asset_holdings" in values_to_record:
            values["asset_holdings"] = self.current_portfolio_matrix
        if "actor_balance" in values_to_record:
            values["actor_balance"] = self.current_portfolio_value
        if "balance" in values_to_record:
            values["balance"] = self.get_portfolio_value().unsqueeze(0)

        # Record the values
        self.tracker.record_step(**values)