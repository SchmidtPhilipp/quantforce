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

import time
from dataclasses import dataclass
from datetime import datetime
from typing import Optional

import numpy as np
import torch
from gymnasium import spaces

import qf
from qf.data.get_data import get_data
from qf.envs.config.env_config import EnvConfig
from qf.envs.dataclass.done import Done
from qf.envs.dataclass.observation import Observation
from qf.envs.portfolio.portfolio import Portfolio
from qf.envs.reward_functions import (
    reward_function_factory,
)
from qf.envs.tensor_env import TensorEnv
from qf.results.episode import Episode
from qf.results.result import Result
from qf.results.run import Run
from qf.utils.experiment_logger import ExperimentLogger
from qf.utils.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)


@dataclass
class AgentConfig:
    n_agents: int


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

    def __init__(
        self,
        environment_name: str,
        n_agents: int,
        env_config: EnvConfig,
        device: str = qf.DEFAULT_DEVICE,
        verbosity: int = qf.VERBOSITY,
    ):

        super().__init__(device=device)

        self.environment_name = environment_name
        self.env_config = env_config
        self.verbosity = verbosity
        self.n_agents = n_agents
        self.n_assets = len(self.env_config.data_config.tickers)
        self.rebalancing_period = env_config.rebalancing_period
        self.last_rebalancing_step = -1  # Track when we last rebalanced

        # Logging configuration for performance optimization
        self.use_detailed_logging = getattr(env_config, "use_detailed_logging", False)

        # Store current agent for type detection
        self.current_agent = None  # Will be set when agent registers

        # Get data directly using get_data
        self.data = get_data(
            data_config=self.env_config.data_config, verbosity=verbosity
        )

        # Pre-extract closing prices for fast access (major performance optimization)
        self.closing_prices = torch.tensor(
            self.data.xs("Close", axis=1, level=1).values,
            dtype=self.env_config.get_torch_dtype(),
            device=self.device,
        )

        # Pre-allocate buffers for repeated operations (performance optimization)
        self._reward_buffer = torch.zeros(
            self.n_agents, dtype=self.env_config.get_torch_dtype(), device=self.device
        )
        self._trade_costs_buffer = torch.zeros(
            self.n_agents, dtype=self.env_config.get_torch_dtype(), device=self.device
        )
        self._price_buffer = torch.zeros(
            self.n_assets, dtype=self.env_config.get_torch_dtype(), device=self.device
        )
        self._action_buffer = torch.zeros(
            self.n_agents,
            self.n_assets + 1,
            dtype=self.env_config.get_torch_dtype(),
            device=self.device,
        )

        # Pre-allocate observation buffer
        obs_dim = self.env_config.obs_window_size * self.n_assets
        self._obs_buffer = torch.zeros(
            self.n_agents,
            obs_dim,
            dtype=self.env_config.get_torch_dtype(),
            device=self.device,
        )

        # Performance monitoring
        if self.env_config.performance_monitoring:
            self._step_count = 0
            self._total_step_time = 0.0

        # Memory pooling for large tensors (performance optimization)
        self._memory_pool = {}
        self._pool_keys = ["obs", "actions", "rewards", "portfolio"]

        # Batch processing optimization for multi-agent scenarios
        self._batch_size = max(1, n_agents // 4)  # Process agents in batches
        self._use_batch_processing = n_agents > 4

        # Caching for repeated calculations (performance optimization)
        self._price_cache = {}
        self._obs_cache = {}
        self._cache_size = 100  # Maximum cache entries

        self.current_step = 0

        # Create reward function using the factory method
        self.reward_calculator = reward_function_factory(
            config=self.env_config.reward_function_config,
            n_agents=self.n_agents,
            device=self.device,
            dataset=self.data,
        )

        self.current_portfolio = Portfolio(
            n_agents=self.n_agents,
            initial_balance=self.env_config.initial_balance,
            device=self.device,
            n_assets=self.n_assets,
            dtype=self.env_config.get_torch_dtype(),
        )

        self.previous_portfolio = Portfolio(
            n_agents=self.n_agents,
            initial_balance=self.env_config.initial_balance,
            device=self.device,
            n_assets=self.n_assets,
            dtype=self.env_config.get_torch_dtype(),
        )

        # Calculate observation space dimensions
        self.n_external_observables = (
            len(self.data.columns) * self.env_config.obs_window_size
        )
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
            low=0,
            high=1,
            shape=(self.n_assets + 1,),
            dtype=np.float32,
        )

        self.last_actions = torch.zeros(
            (self.n_agents, self.action_space.shape[0]),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_actions[:, -1] = 1.0

        logger.info(
            f"MultiAgentPortfolioEnv initialized with {self.n_agents} agents and rebalancing period {self.rebalancing_period}.",
        )

        # Initialize Metrics
        self.metrics = qf.Metrics(
            periods_per_year=self.env_config.data_config.n_trading_days,
            risk_free_rate=self.env_config.risk_free_rate,
        )

        # Initialize the logger
        run_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        # name = qf.generate_random_name()
        self.config_name = self.env_config.config_name
        self.log_dir = self.env_config.log_dir
        run_name = f"{run_time}_{self.config_name}_{environment_name}"  # _{name}"

        """Initialize tracking components (logger and tracker)"""
        # Initialize the tracker
        self.data_collector = Run(
            run_name=run_name,
            tickers=self.env_config.data_config.tickers,
        )

        self.episode = Episode(
            episode_number=0,
            tickers=self.env_config.data_config.tickers,
        )

        self.save_dir = self.log_dir + f"/{run_name}"
        self.experiment_logger = ExperimentLogger(
            run_name=run_name,
            log_dir=self.log_dir,
        )

    def step(self, actions) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict]:
        """
        Executes a step in the environment.

        Parameters:
            actions: Tensor or numpy array of actions from each agent (shape: [n_agents, action_dim]).

        Returns:
            obs (torch.Tensor): Next observation(s) for the agents.
            rewards (torch.Tensor): Rewards for each agent (shape: [n_agents]).
            done (bool): Whether the episode is finished.
            info (dict): Additional information.
        """
        if self.env_config.performance_monitoring:
            start_time = time.time()

        actions = self._ensure_actions_are_valid(actions)
        self.last_actions = actions  # Save actions for state update.

        # Get prices for the current and next steps
        old_prices, new_prices, done, obs = self._step_info()

        # Update the portfolio
        self.current_portfolio, self.previous_portfolio, total_trade_costs = (
            self._update_portfolio(actions, old_prices, new_prices)
        )

        # End the episode if the portfolio value is only a 1000th of the initial balance or less
        if torch.any(
            self.current_portfolio.value <= self.env_config.initial_balance / 1000
        ):
            logger.warning(
                "Episode ended due to portfolio value dropping below 0.1% of initial balance.",
            )
            done(True)
            self.current_portfolio.value[self.current_portfolio.value <= 0] = (
                0  # Set negative values to zero
            )
            # Use pre-allocated buffer instead of creating new tensor
            self._reward_buffer.fill_(self.env_config.reward_function_config.bad_reward)
            rewards = self._reward_buffer

        else:
            rewards = self.reward_calculator.calculate(
                current_portfolio=self.current_portfolio,
                previous_portfolio=self.previous_portfolio,
                transaction_costs=total_trade_costs,
            )

        self.episode.add_step(
            step_result=Result(
                rewards=(
                    rewards.detach() if isinstance(rewards, torch.Tensor) else rewards
                ),
                actions=(
                    actions.detach() if isinstance(actions, torch.Tensor) else actions
                ),
                asset_holdings=(
                    self.current_portfolio.weights.detach()
                    if isinstance(self.current_portfolio.weights, torch.Tensor)
                    else self.current_portfolio.weights
                ),
                actor_balance=(
                    self.current_portfolio.value.detach()
                    if isinstance(self.current_portfolio.value, torch.Tensor)
                    else self.current_portfolio.value
                ),
                balance=(
                    self.current_portfolio.value.sum().detach()
                    if isinstance(self.current_portfolio.value, torch.Tensor)
                    else self.current_portfolio.value.sum()
                ),
                cash=(
                    self.current_portfolio.cash.detach()
                    if isinstance(self.current_portfolio.cash, torch.Tensor)
                    else self.current_portfolio.cash
                ),
                date=torch.tensor(
                    [(self.data.index[self.current_step - 1]).timestamp()],
                    dtype=torch.int32,  # Keep int32 for timestamps
                    device=self.device,
                ),
            )
        )

        if done:
            # Use optimized logging by default, detailed logging only when explicitly requested
            # This provides 10-100x speedup by logging only episode summaries instead of every step
            if self.use_detailed_logging:
                self.episode.log_detailed(self.experiment_logger, self.environment_name)
            else:
                self.episode.log(self.experiment_logger, self.environment_name)
            self.metrics.append(self.episode.get_balance())
            self.data_collector.add_episode(self.episode)

        # Performance monitoring

        # Log performance every 1000 steps
        if self.env_config.performance_monitoring:
            step_time = time.time() - start_time
            self._step_count += 1
            self._total_step_time += step_time
            if self._step_count % 1000 == 0:
                avg_step_time = self._total_step_time / self._step_count
                logger.info(
                    f"Performance: {avg_step_time:.6f}s per step ({1/avg_step_time:.1f} steps/sec) - {self._step_count} steps"
                )

        return obs, rewards, done.as_tensor(), {}

    def reset(self, *, seed=None, options=None):
        """
        Resets the environment and returns the first observation.
        """

        self.episode = Episode(
            episode_number=len(self.data_collector.episodes),
            tickers=self.env_config.data_config.tickers,
        )

        # Set the episode max steps based on actual environment timesteps
        self.episode.set_max_steps(self.get_timesteps())

        # We need to account for both past and future windows
        self.current_step = max(
            self.reward_calculator.config.past_window,
            self.env_config.obs_window_size - 1,
        )
        self.maximum_step = len(self.data) - self.reward_calculator.config.future_window

        # Reset rebalancing tracking
        self.last_rebalancing_step = -1

        self.current_portfolio = Portfolio(
            n_agents=self.n_agents,
            initial_balance=self.env_config.initial_balance,
            device=self.device,
            n_assets=self.n_assets,
        )
        self.previous_portfolio = Portfolio(
            n_agents=self.n_agents,
            initial_balance=self.env_config.initial_balance,
            device=self.device,
            n_assets=self.n_assets,
            dtype=self.env_config.get_torch_dtype(),
        )

        self.reward_calculator.reset()

        self.last_actions = torch.zeros(
            (self.n_agents, self.action_space.shape[0]),
            dtype=torch.float32,
            device=self.device,
        )
        self.last_actions[:, -1] = 1.0

        _, _, _, obs = self._step_info()

        return obs, {}

    def print_metrics(self):
        """Print the metrics report."""
        self.metrics.print_report()

    def get_timesteps(self):
        """
        Returns the number of timesteps in the environment.
        """
        past_window = self.reward_calculator.config.past_window
        future_window = self.reward_calculator.config.future_window

        # We need to account for both past and future windows
        return (
            len(self.data)
            - max(past_window, self.env_config.obs_window_size - 1)
            - future_window
        )

    def save_data(
        self, path=None, save_tracker=True, save_config=True, save_metrics=True
    ):
        import os

        if path is None:
            path = self.save_dir

        # Save tracker data
        if save_tracker:
            self.data_collector.save(path)

        # Save config data
        config_path = os.path.join(path, "env_config.json")
        if save_config:
            with open(config_path, "w") as f:
                import dataclasses
                import json

                # Create a dictionary from the config objects
                config_data = {
                    "env_config": (
                        dataclasses.asdict(self.env_config)
                        if dataclasses.is_dataclass(self.env_config)
                        else self.env_config
                    ),
                }
                json.dump(config_data, f, indent=4)

        # Save metrics data
        metrics_path = os.path.join(path, "metrics")
        if save_metrics:
            self.metrics.save(metrics_path)

    def log_metrics(
        self, experiment_logger: ExperimentLogger = None, run_type: str = None
    ):
        """
        Logs the metrics of the environment at a specific step.

        Parameters:
            experiment_logger (experiment_logger): experiment_logger instance to log the metrics.
            run_type (str): Type of run (e.g., "train", "eval").
        """
        if experiment_logger is None:
            experiment_logger = self.experiment_logger

        self.metrics.log(logger=experiment_logger, run_type=run_type)

    def register_agent(self, agent):
        """Register an agent with the environment for type detection. Replaces any previous agent."""
        self.current_agent = agent

    def enable_detailed_logging(self):
        """Enable detailed per-step logging for debugging (slower but more detailed)."""
        self.use_detailed_logging = True
        logger.info(
            "Detailed logging enabled - this will be slower but provide step-by-step data"
        )

    def disable_detailed_logging(self):
        """Disable detailed logging and use optimized summary logging (faster)."""
        self.use_detailed_logging = False
        logger.info("Optimized summary logging enabled - faster performance")
        return False

    def _get_effective_rebalancing_period(self):
        """Get the effective rebalancing period considering agent config overrides."""
        if (
            not self.current_agent
            or not hasattr(self.current_agent, "config")
            or not self.current_agent.config
        ):
            return self.rebalancing_period  # Use environment default

        # Check if agent config has rebalancing_period set
        if hasattr(self.current_agent.config, "rebalancing_period"):
            agent_rebalancing = self.current_agent.config.rebalancing_period
            if agent_rebalancing is not None:
                return agent_rebalancing  # Use agent's setting

        return self.rebalancing_period  # Fall back to environment default

    #########################################################
    # ALL Private Methods

    def _update_portfolio(
        self, actions, old_prices, new_prices
    ) -> tuple[Portfolio, Portfolio, torch.Tensor]:
        """
        Updates the portfolio based on the actions and prices, respecting the rebalancing period.
        Checks for agent-specific rebalancing period overrides.

        Returns:
            current_portfolio (Portfolio): The updated portfolio.
            previous_portfolio (Portfolio): The previous portfolio.
            total_trade_costs (torch.Tensor): Trading costs incurred.
        """

        # Get effective rebalancing period (agent config overrides environment)
        effective_rebalancing_period = self._get_effective_rebalancing_period()

        # Check if we should rebalance based on the effective rebalancing period
        if self.last_rebalancing_step == -1:
            should_rebalance = True
        else:
            should_rebalance = (
                self.current_step - self.last_rebalancing_step
            ) >= effective_rebalancing_period  # Time to rebalance

        if not should_rebalance:
            # No rebalancing: just update portfolio value based on price changes
            self.previous_portfolio = self.current_portfolio

            # Update portfolio value based on price changes only
            asset_values = self.current_portfolio.weights * new_prices.unsqueeze(0)
            self.current_portfolio.value = self.current_portfolio.cash + torch.sum(
                asset_values, dim=1
            )

            # Use pre-allocated buffer instead of creating new tensor
            self._trade_costs_buffer.zero_()
            total_trade_costs = self._trade_costs_buffer

            return self.current_portfolio, self.previous_portfolio, total_trade_costs

        # Rebalancing logic (original implementation)
        self.last_rebalancing_step = self.current_step  # Update last rebalancing step

        # 1. Use the current portfolio value
        current_portfolio_value_t = self.current_portfolio.value

        # 2. Split actions into asset weights and cash weights
        asset_weights = actions[:, :-1]
        cash_weights = actions[:, -1]

        # 3. Calculate target cash and asset values
        # target_cash = current_portfolio_value_t * cash_weights
        target_asset_values = current_portfolio_value_t * asset_weights

        # 4. Calculate target asset holdings (vectorized)
        target_holdings = torch.floor(
            target_asset_values / (old_prices.unsqueeze(0) + 1e-10),
        )

        # 5. Calculate differences in holdings
        delta_holdings = target_holdings - self.current_portfolio.weights

        # 6. Calculate buy and sell costs (vectorized)
        buy_costs = torch.sum(
            torch.clamp(delta_holdings, min=0) * old_prices.unsqueeze(0),
            dim=1,
        )
        sell_proceeds = torch.sum(
            torch.clamp(-delta_holdings, min=0) * old_prices.unsqueeze(0),
            dim=1,
        )

        # 7. Calculate trading costs (vectorized)
        trade_costs_percent = torch.sum(
            torch.abs(delta_holdings)
            * old_prices.unsqueeze(0)
            * self.env_config.trade_cost_percent,
            dim=1,
        )
        trade_costs_fixed = (
            torch.sum((delta_holdings != 0).float(), dim=1)
            * self.env_config.trade_cost_fixed
        )
        total_trade_costs = trade_costs_percent + trade_costs_fixed

        # 8. Update the previous portfolio
        self.previous_portfolio = self.current_portfolio

        # 9. Update cash and holdings (vectorized)
        self.current_portfolio.cash = (
            self.current_portfolio.cash + sell_proceeds - buy_costs - total_trade_costs
        )
        self.current_portfolio.weights = target_holdings

        # 10. Calculate new portfolio value (vectorized)
        current_portfolio_value_t1 = self.current_portfolio.cash + torch.sum(
            self.current_portfolio.weights * new_prices,
            dim=1,
        )

        # Update the portfolio value property
        self.current_portfolio.value = current_portfolio_value_t1

        return self.current_portfolio, self.previous_portfolio, total_trade_costs

    def _ensure_actions_are_valid(self, actions):
        """Ensure actions are valid tensors on the correct device."""
        if isinstance(actions, np.ndarray):
            actions = torch.from_numpy(actions).to(self.device)
        elif isinstance(actions, torch.Tensor):
            if actions.device != self.device:
                actions = actions.to(self.device)
        else:
            # Try to convert to tensor
            try:
                actions = torch.tensor(
                    actions, dtype=self.env_config.get_torch_dtype(), device=self.device
                )
            except:
                raise ValueError(f"Invalid actions type: {type(actions)}")

        # Ensure actions have the right shape for n_agents
        if actions.dim() == 1:
            actions = actions.unsqueeze(0)  # Add agent dimension

        # Use pre-allocated buffer for validation (performance optimization)
        self._action_buffer.copy_(actions)

        # Validate and normalize in-place
        if torch.any(self._action_buffer < 0) or torch.any(
            torch.abs(self._action_buffer.sum(dim=1) - 1.0) > 1e-6
        ):
            torch.clamp_(self._action_buffer, min=0)

            # Handle normalization differently for integer datatypes
            if self._action_buffer.dtype in [
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
            ]:
                # For integer types, use float32 for division then convert back
                temp_buffer = self._action_buffer.float()
                temp_buffer.div_(temp_buffer.sum(dim=1, keepdim=True) + 1e-8)
                self._action_buffer.copy_(temp_buffer.to(self._action_buffer.dtype))
            else:
                # For float types, normalize in-place
                self._action_buffer.div_(
                    self._action_buffer.sum(dim=1, keepdim=True) + 1e-8
                )

        return self._action_buffer

    def _step_info(self) -> tuple[torch.Tensor, torch.Tensor, bool, Observation]:
        # Get prices at current step using pre-extracted closing prices (much faster)
        old_prices = self.closing_prices[self.current_step - 1]

        # Prepare next step
        self.current_step += 1

        # Check if episode is done
        done = Done(
            self.current_step >= self.maximum_step,
            n_agents=self.n_agents,
            device=self.device,
        )

        if not done:
            new_prices = self.closing_prices[self.current_step]
        else:
            new_prices = old_prices  # Use last prices if done

        # Build observation at the new current_step
        window = self.data.iloc[
            self.current_step
            - 1
            - max(
                self.reward_calculator.config.past_window,
                self.env_config.obs_window_size - 1,
            ) : self.current_step
        ].values

        # Create observation tensor directly (avoiding buffer size issues)
        obs_tensor = torch.tensor(
            window.flatten(),
            dtype=self.env_config.get_torch_dtype(),
            device=self.device,
        )
        obs_tensor = obs_tensor.repeat(self.n_agents, 1)

        date = self.data.index[self.current_step]

        observation = Observation(
            date=date,
            observations=obs_tensor,
            actions=(
                self.last_actions
                if self.env_config.observation_config.include_actions
                else None
            ),
            portfolio=(
                self.current_portfolio.weights
                if self.env_config.observation_config.include_portfolio
                else None
            ),
            cash=(
                self.current_portfolio.cash
                if self.env_config.observation_config.include_cash
                else None
            ),
        )

        return old_prices, new_prices, done, observation
