from datetime import datetime
from typing import Tuple
from gymnasium import spaces
import torch
import numpy as np

import qf as qf

from qf.utils.logger import Logger
from qf.utils.tracker.tracker import Tracker
from qf.data.dataset import TimeBasedDataset
from qf.envs.tensor_env import TensorEnv


class MultiAgentPortfolioEnv(TensorEnv):
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
    def __init__(self, 
                tensorboard_prefix,
                config=None):
        
        super(MultiAgentPortfolioEnv, self).__init__(device=config.get("device", qf.DEFAULT_DEVICE))


        default_config = {
            "tickers": qf.DEFAULT_TICKERS, 
            "start_date": qf.DEFAULT_TRAIN_START,
            "end_date": qf.DEFAULT_TRAIN_END,
            "window_size": qf.DEFAULT_WINDOW_SIZE,
            "interval": qf.DEFAULT_INTERVAL,
            "indicators": qf.DEFAULT_INDICATORS,
            "cache_dir": qf.DEFAULT_CACHE_DIR,
            "initial_balance": qf.DEFAULT_INITIAL_BALANCE,
            "n_agents": qf.DEFAULT_N_AGENTS,
            "trade_cost_percent": qf.DEFAULT_TRADE_COST_PERCENT,
            "trade_cost_fixed": qf.DEFAULT_TRADE_COST_FIXED,
            "reward_function": qf.DEFAULT_REWARD_FUNCTION,
            "reward_scaling": qf.DEFAULT_REWARD_SCALING,
            "final_reward": qf.DEFAULT_FINAL_REWARD,
            "verbosity": qf.VERBOSITY,
            "log_dir": qf.DEFAULT_LOG_DIR,
            "config_name": qf.DEFUALT_CONFIG_NAME,
        }

        # If config is provided, update the default config with the provided config
        self.config = {**default_config, **(config or {})}

        self.dataset = TimeBasedDataset(tickers=self.config["tickers"],
                                        start_date=self.config["start_date"],
                                        end_date=self.config["end_date"],
                                        window_size=self.config["window_size"],
                                        indicators=self.config["indicators"],
                                        cache_dir=self.config["cache_dir"],
                                        interval=self.config["interval"])
        
        self.dataloader = self.dataset.get_dataloader()
        self.data_iterator = iter(self.dataloader)

        self.tickers = self.config["tickers"]
        self.n_assets = len(self.tickers)

        self.obs_window_size = self.config["window_size"]
        self.initial_balance = self.config["initial_balance"]
        self.current_step = 0
        self.verbosity = self.config["verbosity"]
        self.n_agents = self.config["n_agents"]
        self.trade_cost_percent = self.config["trade_cost_percent"]
        self.trade_cost_fixed = self.config["trade_cost_fixed"]

        # Reward function can be "linear_rate_of_return", "log_return", or "absolute_return", or "sharpe_ratio_wX" where X is the time horizon the sharpe ratio is calculated over
        self.reward_function = self.config["reward_function"]
        self.reward_scaling = self.config["reward_scaling"]
        self.final_reward = self.config["final_reward"]
        
        # check if the reward function begins with "sharpe_ratio_w"
        if self.reward_function is not None and self.reward_function.startswith("sharpe_ratio_w"):
            # Extract the time horizon from the reward function
            try:
                self.sharpe_time_horizon = int(self.reward_function.split("_w")[1])
            except ValueError:
                raise ValueError(f"Invalid reward function format: {self.reward_function}. Expected format 'sharpe_ratio_wX' where X is an integer.")
            self.reward_function = "sharpe_ratio"


        # Initialize actor cash and asset holdings for each agent as Torch Tensors
        self.current_cash_vector = torch.full((self.n_agents,), self.initial_balance / self.n_agents, dtype=torch.float32, device=self.device)
        self.current_portfolio_matrix = torch.zeros((self.n_agents, self.n_assets), dtype=torch.float32, device=self.device)
        self.current_portfolio_value = torch.full((self.n_agents,), self.initial_balance / self.n_agents, dtype=torch.float32, device=self.device)


        # Initialize last actions (all cash initially)
        self.last_actions = torch.zeros((self.n_agents, self.n_assets + 1), dtype=torch.float32, device=self.device)
        self.last_actions[:, -1] = 1  # Set the last entry (cash) to 1

        # Observation space: all assets and indicators + actions
        self.n_external_observables = self.dataset.get_width() * self.obs_window_size
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


        # Initialize the tracker
        self.tracker = Tracker(timesteps=self.get_timesteps(), tensorboard_prefix=f"{tensorboard_prefix}")
        self.register_tracker()

        # Initialize the logger
        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") 
        name = qf.generate_random_name()
        self.config_name = self.config["config_name"]
        self.log_dir = config["log_dir"]
        self.save_dir = self.log_dir + f"/{run_time}_{self.config_name}_{tensorboard_prefix}_{name}"
        self.logger = Logger(run_name=f"{run_time}_{self.config_name}_{tensorboard_prefix}_{name}", log_dir=self.log_dir)

        # Initialize Metrics
        self.metrics = qf.Metrics()
        self.balance = []


    def get_observation_space(self):
        """
        Returns the observation space of the environment.

        Returns:
            spaces.Box: The observation space.
        """
        return self.observation_space

    def get_action_space(self):
        """
        Returns the action space of the environment.

        Returns:
            spaces.Box: The action space.
        """
        return self.action_space

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

    def  step(self, actions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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
            self.dataset.data.xs("Close", axis=1, level=1).iloc[self.current_step + self.obs_window_size - 1].values,
            dtype=torch.float32,
            device=self.device
        )

        self.current_step += 1
        done = self.current_step >= len(self.dataloader)


        if done: 
            new_prices = old_prices # We use the last prices to calculate the balance a last time such that everything in the df has the same length
            obs = torch.zeros((self.n_agents, *self.observation_space.shape), dtype=torch.float32, device=self.device)
        else:
            new_prices = torch.tensor(
                self.dataset.data.xs("Close", axis=1, level=1).iloc[self.current_step + self.obs_window_size - 1].values,
                dtype=torch.float32,
                device=self.device
            )
        
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

        # Keep track of the total balance
        self.balance.append(torch.sum(self.current_portfolio_value))

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
            
            # We have to adjust the current step because we already incremented it. 
            adjusted_current_step = self.current_step - 1 

            # We build a price matrix that contains the (sharpe time horizon of past prices)x(n_assets)
            past_price_matrix = torch.tensor(
                self.dataset.data.xs("Close", axis=1, level=1).iloc[(adjusted_current_step - self.sharpe_time_horizon):(adjusted_current_step)].values,
                dtype=torch.float32,
                device=self.device
            ) # shape: [past_steps, n_assets]

            # We take the pastt portfolio matrix which contains the portfolio for each agent ([n_agents, n_assets]) 
            # and multiply it with the past price matrix to get the portfolio values for each agent in the past assuming 
            # that the portfolio was held constant during the past steps. This gives us a matrix of strictly positive values
            past_portfolio_values = self.current_portfolio_matrix @ past_price_matrix.T # shape: [n_agents, past_steps]

            # We calculate the past linear returns of this portfolio. 
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
        
        # Scale the rewards
        rewards = rewards * self.reward_scaling

        done = torch.full((self.n_agents, 1), float(done), dtype=torch.float32, device=self.device)

        # End the episode if the portfolio value is only a 1000th of the initial balance or less
        if torch.any(current_portfolio_value_t1 <= self.initial_balance / 1000):
            print("Episode ended due to portfolio value dropping below 0.1% of initial balance.")
            done = torch.full((self.n_agents, 1), float(True), dtype=torch.float32, device=self.device)
            current_portfolio_value_t1[current_portfolio_value_t1 <= 0] = 0  # Set negative values to zero
            obs = torch.zeros((self.n_agents, *self.observation_space.shape), dtype=torch.float32, device=self.device)


        self.record_data(action=actions, reward=rewards)
        
        if done.any():
            self._end_episode()

        # Get the next observation
        if not done:
            obs = self._get_observation()
        
        output = obs, rewards, done, {}
        return output
    
        
    def reset(self, *, seed=None, options=None):
        """
        Resets the environment and returns the first observation.
        """
        self.current_step = 0 + self.sharpe_time_horizon if self.reward_function == "sharpe_ratio" else 0
        self.data_iterator = iter(self.dataloader)

        if self.reward_function == "sharpe_ratio":
            # we throw away the first self.sharpe_time_horizon steps, because we need at least that many steps to calculate the sharpe ratio
            for _ in range(self.sharpe_time_horizon):
                next(self.data_iterator)


        self.current_cash_vector = torch.full((self.n_agents,), self.initial_balance / self.n_agents, dtype=torch.float32, device=self.device)
        self.current_portfolio_matrix = torch.zeros((self.n_agents, self.n_assets), dtype=torch.float32, device=self.device)
        self.current_portfolio_value = torch.full((self.n_agents,), self.initial_balance / self.n_agents, dtype=torch.float32, device=self.device)

        self.last_actions = torch.zeros((self.n_agents, self.n_assets + 1), dtype=torch.float32, device=self.device)
        self.last_actions[:, -1] = 1  # Set the last entry (cash) to 1

        obs = self._get_observation()

        return obs, {}
    
    
    def _end_episode(self):
        self.metrics.append(self.balance)
        self.tracker.end_episode()
        self.tracker.log(self.logger)

    def print_metrics(self):
        self.metrics.print_report()
        

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
        if self.reward_function == "sharpe_ratio":
            # We need to adjust the number of timesteps because we throw away the first self.sharpe_time_horizon steps
            return len(self.dataloader) - self.sharpe_time_horizon 
        else:
            return len(self.dataloader)

    def get_dataset(self):
        """
        Returns the dataset used in the environment.
        
        Returns:
            data (TimeBasedDataset): The dataset containing historical data.
        """
        return self.dataset
    
    def register_tracker(self):
        """
        Registers a tracker for logging and monitoring.

        Parameters:
            tracker (object): Tracker object for logging.
        """

        # Register tracked values with custom axis labels
        self.tracker.register_value("rewards", shape=(self.n_agents,), description="Rewards per agent", dimensions=["timesteps", "agents"], labels=[range(self.n_agents)])
        self.tracker.register_value("actions",shape=(self.n_agents, self.n_assets + 1),description="Actions per agent",dimensions=["timesteps", "agents", "assets"],labels=[range(self.n_agents), self.tickers + ["cash"]])
        #tracker.register_value("asset_holdings",shape=(self.n_agents, self.n_assets),description="Asset holdings per agent",dimensions=["timesteps", "agents", "assets"],labels=[range(self.n_agents), self.tickers])
        self.tracker.register_value("balance", shape=(1,), description="Environment balance", dimensions=["timesteps"], labels=[["balance"]])
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

    def save_and_close(self):
        import os
        # Close the loggers
        self.logger.close()

        # Save tracker data
        self.tracker.save(self.save_dir)

        # Save config data
        config_path = os.path.join(self.save_dir, "env_config.json")
        with open(config_path, "w") as f:
            import json
            json.dump(self.config, f, indent=4)

        # Save metrics data
        metrics_path = os.path.join(self.save_dir, "metrics")
        self.metrics.save(metrics_path)

    def get_save_path(self):
        """
        Returns the path where the environment data is saved.
        
        Returns:
            str: Path to the saved environment data.
        """
        return self.save_dir
        


