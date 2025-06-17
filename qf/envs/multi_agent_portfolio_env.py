from datetime import datetime
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from gymnasium import spaces

import qf as qf
from qf.data.utils.get_data import get_data
from qf.envs.reward_functions import (
    AbsoluteReturn,
    DifferentialSharpeRatio,
    LinearRateOfReturn,
    LogReturn,
    RewardFunction,
    SharpeRatio,
)
from qf.envs.tensor_env import TensorEnv
from qf.utils.logger import Logger
from qf.utils.tracker.tracker import Tracker


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

    def __init__(self, tensorboard_prefix, config=None):
        super(MultiAgentPortfolioEnv, self).__init__(
            device=(
                config["device"] if config and "device" in config else qf.DEFAULT_DEVICE
            )
        )

        default_config = {
            "tickers": qf.DEFAULT_TICKERS,
            "start": qf.DEFAULT_TRAIN_START,
            "end": qf.DEFAULT_TRAIN_END,
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
            "bad_reward": qf.DEFAULT_BAD_REWARD,
            "verbosity": qf.VERBOSITY,
            "log_dir": qf.DEFAULT_LOG_DIR,
            "config_name": qf.DEFUALT_CONFIG_NAME,
        }

        # If config is provided, update the default config with the provided config
        self.config = {**default_config, **(config or {})}

        # Get data directly using get_data
        self.data = get_data(
            tickers=self.config["tickers"],
            start=self.config["start"],
            end=self.config["end"],
            indicators=self.config["indicators"],
            cache_dir=self.config["cache_dir"],
            interval=self.config["interval"],
        )

        self.start = self.config["start"]
        self.end = self.config["end"]

        self.tickers = self.config["tickers"]
        self.n_assets = len(self.tickers)
        self.obs_window_size = self.config["window_size"]
        self.initial_balance = self.config["initial_balance"]
        self.current_step = 0
        self.verbosity = self.config["verbosity"]
        self.n_agents = self.config["n_agents"]
        self.trade_cost_percent = self.config["trade_cost_percent"]
        self.trade_cost_fixed = self.config["trade_cost_fixed"]

        # Reward function can be "linear_rate_of_return", "log_return", "absolute_return", "sharpe_ratio_wX", or "differential_sharpe_ratio"
        self.reward_function = self.config["reward_function"]
        self.reward_scaling = self.config["reward_scaling"]
        self.final_reward = self.config["final_reward"]
        self.bad_reward = self.config["bad_reward"]

        # Initialize the appropriate reward function
        if self.reward_function == "linear_rate_of_return":
            self.reward_calculator = LinearRateOfReturn(
                n_agents=self.n_agents,
                device=self.device,
                reward_scaling=self.reward_scaling,
                bad_reward=self.bad_reward,
            )
        elif self.reward_function == "log_return":
            self.reward_calculator = LogReturn(
                n_agents=self.n_agents,
                device=self.device,
                reward_scaling=self.reward_scaling,
                bad_reward=self.bad_reward,
            )
        elif self.reward_function == "absolute_return":
            self.reward_calculator = AbsoluteReturn(
                n_agents=self.n_agents,
                device=self.device,
                reward_scaling=self.reward_scaling,
                bad_reward=self.bad_reward,
            )
        elif self.reward_function.startswith("sharpe_ratio_w"):
            try:
                window_size = int(self.reward_function.split("_w")[1])
            except ValueError:
                raise ValueError(
                    f"Invalid reward function format: {self.reward_function}. Expected format 'sharpe_ratio_wX' where X is an integer."
                )
            self.reward_calculator = SharpeRatio(
                n_agents=self.n_agents,
                device=self.device,
                window_size=window_size,
                reward_scaling=self.reward_scaling,
                bad_reward=self.bad_reward,
            )
        elif self.reward_function == "differential_sharpe_ratio":
            self.reward_calculator = DifferentialSharpeRatio(
                n_agents=self.n_agents,
                device=self.device,
                reward_scaling=self.reward_scaling,
                bad_reward=self.bad_reward,
            )
        else:
            raise ValueError(
                f"Unknown reward function: {self.reward_function}. Supported functions are 'linear_rate_of_return', 'log_return', 'absolute_return', 'sharpe_ratio_wX', and 'differential_sharpe_ratio'."
            )

        # Initialize actor cash and asset holdings for each agent as Torch Tensors
        self.current_cash_vector = torch.full(
            (self.n_agents,),
            self.initial_balance / self.n_agents,
            dtype=torch.float32,
            device=self.device,
        )
        self.current_portfolio_matrix = torch.zeros(
            (self.n_agents, self.n_assets), dtype=torch.float32, device=self.device
        )
        self.current_portfolio_value = torch.full(
            (self.n_agents,),
            self.initial_balance / self.n_agents,
            dtype=torch.float32,
            device=self.device,
        )

        # Initialize last actions (all cash initially)
        self.last_actions = torch.zeros(
            (self.n_agents, self.n_assets + 1), dtype=torch.float32, device=self.device
        )
        self.last_actions[:, -1] = 1  # Set the last entry (cash) to 1

        # Calculate observation space dimensions
        self.n_external_observables = len(self.data.columns) * self.obs_window_size
        self.observation_space = spaces.Box(
            low=-float("inf"),
            high=float("inf"),
            shape=(
                self.n_external_observables + self.n_assets + 1 + self.n_assets + 1,
            ),
            dtype=np.float32,
        )

        # Action space: weights for each asset and cash
        self.action_space = spaces.Box(
            low=0, high=1, shape=(self.n_assets + 1,), dtype=np.float32
        )

        if self.verbosity > 0:
            print(f"MultiAgentPortfolioEnv initialized with {self.n_agents} agents.")

        # Initialize tracking components
        self._init_tracking(tensorboard_prefix)

        # Initialize Metrics
        self.metrics = qf.Metrics()
        self.balance = []

    def _init_tracking(self, tensorboard_prefix):
        """Initialize tracking components (logger and tracker)"""
        # Initialize the tracker
        self.tracker = Tracker(
            timesteps=self.get_n_succesive_states(),
            tensorboard_prefix=f"{tensorboard_prefix}",
        )
        self.register_tracker()

        # Initialize the logger
        run_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        name = qf.generate_random_name()
        self.config_name = self.config["config_name"]
        self.log_dir = self.config["log_dir"]
        self.save_dir = (
            self.log_dir + f"/{run_time}_{self.config_name}_{tensorboard_prefix}_{name}"
        )
        self.logger = Logger(
            run_name=f"{run_time}_{self.config_name}_{tensorboard_prefix}_{name}",
            log_dir=self.log_dir,
        )

    def __getstate__(self):
        """Return state values to be pickled."""
        state = self.__dict__.copy()
        # Remove non-picklable components
        state.pop("logger", None)
        state.pop("tracker", None)
        return {state}

    def get_n_succesive_states(self):
        """
        Returns the number of consecutive states in the environment.
        """
        return self.get_timesteps()

    def __setstate__(self, state):
        """Restore state from the unpickled state values."""
        self.__dict__.update(state)
        # Reinitialize tracking components
        self._init_tracking(self.config.get("tensorboard_prefix", "default"))

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
        # Get the window of data for the current step
        start_idx = self.current_step - self.obs_window_size + 1
        end_idx = self.current_step
        window_data = self.data.iloc[start_idx : end_idx + 1].values

        current_observations = torch.tensor(
            window_data.flatten(), dtype=torch.float32, device=self.device
        )

        current_observations = current_observations.repeat(self.n_agents, 1)

        current_actions = self.last_actions
        current_holdings = self.current_portfolio_matrix
        current_cash = self.current_cash_vector.unsqueeze(1)

        # Combine the data with the last actions
        obs = torch.cat(
            [current_observations, current_actions, current_holdings, current_cash],
            dim=1,
        )

        return obs

    def step(
        self, actions: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
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
        if type(actions) == np.ndarray:
            actions = torch.from_numpy(actions).to(self.device)
        elif type(actions) == torch.Tensor:
            actions = actions.to(self.device)
        else:
            raise ValueError(f"Invalid actions type: {type(actions)}")

        actions = torch.clamp(actions, 0, 1)  # Ensure actions are in [0, 1]
        actions = actions / actions.sum(dim=1, keepdim=True)  # Normalize actions

        self.last_actions = actions  # Save actions for the next observation

        # Get prices for the current and next steps
        old_prices = torch.tensor(
            self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values,
            dtype=torch.float32,
            device=self.device,
        )

        self.current_step += 1
        done = self.current_step >= len(self.data)

        if done:
            new_prices = old_prices  # We use the last prices to calculate the balance a last time
            obs = torch.zeros(
                (self.n_agents, *self.observation_space.shape),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            new_prices = torch.tensor(
                self.data.xs("Close", axis=1, level=1).iloc[self.current_step].values,
                dtype=torch.float32,
                device=self.device,
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
        target_holdings = torch.floor(
            target_asset_values / (old_prices.unsqueeze(0) + 1e-10)
        )

        # 5. Calculate differences in holdings
        delta_holdings = target_holdings - self.current_portfolio_matrix

        # 6. Calculate buy and sell costs
        buy_costs = torch.sum(
            torch.clamp(delta_holdings, min=0) * old_prices.unsqueeze(0), dim=1
        )
        sell_proceeds = torch.sum(
            torch.clamp(-delta_holdings, min=0) * old_prices.unsqueeze(0), dim=1
        )

        # 7. Calculate trading costs
        trade_costs_percent = torch.sum(
            torch.abs(delta_holdings)
            * old_prices.unsqueeze(0)
            * self.trade_cost_percent,
            dim=1,
        )
        trade_costs_fixed = (
            torch.sum((delta_holdings != 0).float(), dim=1) * self.trade_cost_fixed
        )
        total_trade_costs = trade_costs_percent + trade_costs_fixed

        # 8. Update cash and holdings
        self.current_cash_vector = (
            self.current_cash_vector + sell_proceeds - buy_costs - total_trade_costs
        )
        self.current_portfolio_matrix = target_holdings

        # 9. Calculate new portfolio value
        current_portfolio_value_t1 = self.current_cash_vector + torch.sum(
            self.current_portfolio_matrix * new_prices, dim=1
        )

        # Update the portfolio value property
        self.current_portfolio_value = current_portfolio_value_t1

        # Keep track of the total balance
        self.balance.append(torch.sum(self.current_portfolio_value))

        # Calculate rewards using the reward calculator
        rewards = self.reward_calculator.calculate(
            current_portfolio_value=current_portfolio_value_t1,
            old_portfolio_value=current_portfolio_value_t,
            portfolio_matrix=self.current_portfolio_matrix,
        )

        # Apply scaling and punishment
        rewards = self.reward_calculator.apply_scaling_and_punishment(
            rewards=rewards, portfolio_matrix=self.current_portfolio_matrix
        )

        done = torch.full(
            (self.n_agents, 1), float(done), dtype=torch.float32, device=self.device
        )

        # End the episode if the portfolio value is only a 1000th of the initial balance or less
        if torch.any(current_portfolio_value_t1 <= self.initial_balance / 1000):
            print(
                "Episode ended due to portfolio value dropping below 0.1% of initial balance."
            )
            done = torch.full(
                (self.n_agents, 1), float(True), dtype=torch.float32, device=self.device
            )
            current_portfolio_value_t1[current_portfolio_value_t1 <= 0] = (
                0  # Set negative values to zero
            )
            obs = torch.zeros(
                (self.n_agents, *self.observation_space.shape),
                dtype=torch.float32,
                device=self.device,
            )

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
        self.current_step = 0
        # Get the window size from the reward calculator if it's a SharpeRatio
        window_size = 0
        if isinstance(self.reward_calculator, SharpeRatio):
            window_size = self.reward_calculator.window_size
        self.current_step += max(window_size, self.obs_window_size - 1)

        self.current_cash_vector = torch.full(
            (self.n_agents,),
            self.initial_balance / self.n_agents,
            dtype=torch.float32,
            device=self.device,
        )
        self.current_portfolio_matrix = torch.zeros(
            (self.n_agents, self.n_assets), dtype=torch.float32, device=self.device
        )
        self.current_portfolio_value = torch.full(
            (self.n_agents,),
            self.initial_balance / self.n_agents,
            dtype=torch.float32,
            device=self.device,
        )

        # Reset reward calculator if it has a reset method
        if hasattr(self.reward_calculator, "reset"):
            self.reward_calculator.reset()

        self.last_actions = torch.zeros(
            (self.n_agents, self.n_assets + 1), dtype=torch.float32, device=self.device
        )
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
        # Get the window size from the reward calculator if it's a SharpeRatio
        window_size = 0
        if isinstance(self.reward_calculator, SharpeRatio):
            window_size = self.reward_calculator.window_size
        return len(self.data) - max(window_size, self.obs_window_size - 1)

    def register_tracker(self):
        """
        Registers a tracker for logging and monitoring.

        Parameters:
            tracker (object): Tracker object for logging.
        """

        # Register tracked values with custom axis labels
        self.tracker.register_value(
            "rewards",
            shape=(self.n_agents,),
            description="Rewards per agent",
            dimensions=["timesteps", "agents"],
            labels=[f"Actor_{i}" for i in range(self.n_agents)],
        )
        self.tracker.register_value(
            "actions",
            shape=(self.n_agents, self.n_assets + 1),
            description="Actions per agent",
            dimensions=["timesteps", "agents", "assets"],
            labels=[
                [f"Actor_{i}" for i in range(self.n_agents)],
                self.tickers + ["cash"],
            ],
        )
        # tracker.register_value("asset_holdings",shape=(self.n_agents, self.n_assets),description="Asset holdings per agent",dimensions=["timesteps", "agents", "assets"],labels=[range(self.n_agents), self.tickers])
        self.tracker.register_value(
            "balance",
            shape=(1,),
            description="Environment balance",
            dimensions=["timesteps"],
            labels=[["balance"]],
        )
        # tracker.register_value("actor_balance",shape=(self.n_agents,),description="Actor balances",dimensions=["timesteps", "agents"],labels=[range(self.n_agents)])
        self.tracker.register_value(
            "date",
            shape=(1,),
            description="Current date",
            dimensions=["timesteps", "Env"],
            labels=[["date"]],
        )

    def record_data(self, action=None, reward=None):

        if self.tracker is None:
            raise ValueError(
                "Tracker is not registered. Please register a tracker before recording data."
            )

        values_to_record = self.tracker.tracked_values.keys()

        values = {}

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
        if "date" in values_to_record:
            values["date"] = torch.tensor(
                [(self.data.index[self.current_step - 1]).timestamp()],
                dtype=torch.int32,
                device=self.device,
            )
        try:
            # Record the values
            self.tracker.record_step(**values)
        except Exception as e:
            raise ValueError(
                f"Current step is out of bounds. Please check the data index. {e}"
            )

        print(f"Current step: {self.current_step}")
        print(f"Data index: {self.data.index}")
        print(f"Window size: {self.obs_window_size}")
        print(f"Data length: {len(self.data)}")
        print(f"N Timesteps: {self.get_timesteps()}")

    def save_data(self, path=None):
        import os

        if path is None:
            path = self.save_dir

        # Save tracker data
        self.tracker.save(path)

        # Save config data
        config_path = os.path.join(path, "env_config.json")
        with open(config_path, "w") as f:
            import json

            json.dump(self.config, f, indent=4)

        # Save metrics data
        metrics_path = os.path.join(path, "metrics")
        self.metrics.save(metrics_path)

    def get_save_dir(self):
        """
        Returns the path where the environment data is saved.

        Returns:
            str: Path to the saved environment data.
        """
        return self.save_dir

    def log_metrics(self, logger=None, run_type=None):
        """
        Logs the metrics of the environment at a specific step.

        Parameters:
            logger (Logger): Logger instance to log the metrics.
            run_type (str): Type of run (e.g., "train", "eval").
        """
        if logger is None:
            logger = self.logger

        self.metrics.log(logger=logger, run_type=run_type)

    def get_logger(self):
        """
        Returns the logger instance for the environment.

        Returns:
            Logger: Logger instance.
        """
        return self.logger
