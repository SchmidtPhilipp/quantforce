import copy
from dataclasses import dataclass, field
from typing import Callable, List, Optional

import torch

from qf.utils.experiment_logger import ExperimentLogger


@dataclass
class Episode:
    """
    Holds the results of a single episode, which is a series of steps.
    Optimized with tensor-based storage for better performance.
    """

    episode_number: int = 0
    tickers: List[str] = field(default_factory=list)

    # Tensor-based storage for better performance
    _rewards: Optional[torch.Tensor] = None
    _actions: Optional[torch.Tensor] = None
    _asset_holdings: Optional[torch.Tensor] = None
    _actor_balance: Optional[torch.Tensor] = None
    _balance: Optional[torch.Tensor] = None
    _cash: Optional[torch.Tensor] = None
    _date: Optional[torch.Tensor] = None

    # Track current step for efficient tensor growth
    _current_step: int = 0
    _max_steps: int = 1000  # Pre-allocate for typical episode length

    # Cache for backward compatibility
    _steps_cache: Optional[List] = None
    _steps_cache_step_count: int = 0

    def __post_init__(self):
        """Initialize tensor storage with pre-allocated space."""
        # Use a more reasonable default that can be overridden
        self._max_steps = 1000  # Will be updated when first step is added

    def set_max_steps(self, timesteps: int):
        """Set the maximum number of steps based on environment timesteps."""
        self._max_steps = max(1000, timesteps + 100)  # Add buffer for safety

    def add_step(self, step_result):
        """Add a step result using efficient tensor storage."""
        # Initialize tensors on first step
        if self._current_step == 0:
            self._initialize_tensors(step_result)

        # Store tensors efficiently
        if step_result.rewards is not None:
            self._rewards[self._current_step] = step_result.rewards

        if step_result.actions is not None:
            self._actions[self._current_step] = step_result.actions

        if step_result.asset_holdings is not None:
            self._asset_holdings[self._current_step] = step_result.asset_holdings

        if step_result.actor_balance is not None:
            self._actor_balance[self._current_step] = step_result.actor_balance

        if step_result.balance is not None:
            self._balance[self._current_step] = step_result.balance

        if step_result.cash is not None:
            self._cash[self._current_step] = step_result.cash

        if step_result.date is not None:
            self._date[self._current_step] = step_result.date

        self._current_step += 1

        # Invalidate cache when new steps are added
        self._steps_cache = None
        self._steps_cache_step_count = 0

        # Expand tensors if needed (rare)
        if self._current_step >= self._max_steps:
            self._expand_tensors()

    def _initialize_tensors(self, step_result):
        """Initialize tensor storage based on first step result."""
        device = (
            step_result.rewards.device
            if step_result.rewards is not None
            else torch.device("cpu")
        )
        dtype = (
            step_result.rewards.dtype
            if step_result.rewards is not None
            else torch.float32
        )

        # Get shapes from first step
        if step_result.rewards is not None:
            self._rewards = torch.zeros(
                self._max_steps, *step_result.rewards.shape, dtype=dtype, device=device
            )
        if step_result.actions is not None:
            self._actions = torch.zeros(
                self._max_steps, *step_result.actions.shape, dtype=dtype, device=device
            )
        if step_result.asset_holdings is not None:
            self._asset_holdings = torch.zeros(
                self._max_steps,
                *step_result.asset_holdings.shape,
                dtype=dtype,
                device=device,
            )
        if step_result.actor_balance is not None:
            self._actor_balance = torch.zeros(
                self._max_steps,
                *step_result.actor_balance.shape,
                dtype=dtype,
                device=device,
            )
        if step_result.balance is not None:
            self._balance = torch.zeros(
                self._max_steps, *step_result.balance.shape, dtype=dtype, device=device
            )
        if step_result.cash is not None:
            self._cash = torch.zeros(
                self._max_steps, *step_result.cash.shape, dtype=dtype, device=device
            )
        if step_result.date is not None:
            self._date = torch.zeros(
                self._max_steps, *step_result.date.shape, dtype=dtype, device=device
            )

    def _expand_tensors(self):
        """Expand tensor storage if needed (rare operation)."""
        new_max_steps = self._max_steps * 2

        if self._rewards is not None:
            new_rewards = torch.zeros(
                new_max_steps,
                *self._rewards.shape[1:],
                dtype=self._rewards.dtype,
                device=self._rewards.device,
            )
            new_rewards[: self._current_step] = self._rewards[: self._current_step]
            self._rewards = new_rewards

        if self._actions is not None:
            new_actions = torch.zeros(
                new_max_steps,
                *self._actions.shape[1:],
                dtype=self._actions.dtype,
                device=self._actions.device,
            )
            new_actions[: self._current_step] = self._actions[: self._current_step]
            self._actions = new_actions

        if self._asset_holdings is not None:
            new_asset_holdings = torch.zeros(
                new_max_steps,
                *self._asset_holdings.shape[1:],
                dtype=self._asset_holdings.dtype,
                device=self._asset_holdings.device,
            )
            new_asset_holdings[: self._current_step] = self._asset_holdings[
                : self._current_step
            ]
            self._asset_holdings = new_asset_holdings

        if self._actor_balance is not None:
            new_actor_balance = torch.zeros(
                new_max_steps,
                *self._actor_balance.shape[1:],
                dtype=self._actor_balance.dtype,
                device=self._actor_balance.device,
            )
            new_actor_balance[: self._current_step] = self._actor_balance[
                : self._current_step
            ]
            self._actor_balance = new_actor_balance

        if self._balance is not None:
            new_balance = torch.zeros(
                new_max_steps,
                *self._balance.shape[1:],
                dtype=self._balance.dtype,
                device=self._balance.device,
            )
            new_balance[: self._current_step] = self._balance[: self._current_step]
            self._balance = new_balance

        if self._cash is not None:
            new_cash = torch.zeros(
                new_max_steps,
                *self._cash.shape[1:],
                dtype=self._cash.dtype,
                device=self._cash.device,
            )
            new_cash[: self._current_step] = self._cash[: self._current_step]
            self._cash = new_cash

        if self._date is not None:
            new_date = torch.zeros(
                new_max_steps,
                *self._date.shape[1:],
                dtype=self._date.dtype,
                device=self._date.device,
            )
            new_date[: self._current_step] = self._date[: self._current_step]
            self._date = new_date

        self._max_steps = new_max_steps

    def get_property(self, attr: str) -> torch.Tensor:
        """
        Get a specific property over all steps (optimized tensor access).
        """
        tensor_map = {
            "rewards": self._rewards,
            "actions": self._actions,
            "asset_holdings": self._asset_holdings,
            "actor_balance": self._actor_balance,
            "balance": self._balance,
            "cash": self._cash,
            "date": self._date,
        }

        tensor = tensor_map.get(attr)
        if tensor is None:
            return torch.tensor([])

        # Return only the valid steps (up to current_step)
        return tensor[: self._current_step]

    def reduce_property(
        self, attr: str, fn: Callable[[torch.Tensor], torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """
        Apply a reduction function to a property over all steps (optimized).
        """
        tensor = self.get_property(attr)
        if tensor.numel() == 0:
            return None
        return fn(tensor)

    def mean_property(self, attr: str) -> Optional[torch.Tensor]:
        return self.reduce_property(attr, torch.mean)

    def sum_property(self, attr: str) -> Optional[torch.Tensor]:
        return self.reduce_property(attr, torch.sum)

    # Convenience accessors (optimized)
    def get_balance(self) -> torch.Tensor:
        return self.get_property("balance")

    def get_rewards(self) -> torch.Tensor:
        return self.get_property("rewards")

    def get_actions(self) -> torch.Tensor:
        return self.get_property("actions")

    def get_asset_holdings(self) -> torch.Tensor:
        return self.get_property("asset_holdings")

    def get_actor_balance(self) -> torch.Tensor:
        return self.get_property("actor_balance")

    def get_cash(self) -> torch.Tensor:
        return self.get_property("cash")

    def get_date(self) -> torch.Tensor:
        return self.get_property("date")

    @property
    def steps(self) -> List:
        """
        Backward compatibility property that converts tensor storage back to list of Result objects.
        This is needed for the Run class which expects episode.steps to be a list.
        Uses caching for better performance when accessed multiple times.
        """
        # Return cached result if available and valid
        if (
            self._steps_cache is not None
            and self._steps_cache_step_count == self._current_step
        ):
            return self._steps_cache

        from qf.results.result import Result

        steps_list = []
        for i_step in range(self._current_step):
            step_result = Result(
                rewards=self._rewards[i_step] if self._rewards is not None else None,
                actions=self._actions[i_step] if self._actions is not None else None,
                asset_holdings=(
                    self._asset_holdings[i_step]
                    if self._asset_holdings is not None
                    else None
                ),
                actor_balance=(
                    self._actor_balance[i_step]
                    if self._actor_balance is not None
                    else None
                ),
                balance=self._balance[i_step] if self._balance is not None else None,
                cash=self._cash[i_step] if self._cash is not None else None,
                date=self._date[i_step] if self._date is not None else None,
            )
            steps_list.append(step_result)

        # Cache the result
        self._steps_cache = steps_list
        self._steps_cache_step_count = self._current_step

        return steps_list

    def copy(self):
        """Create a deep copy of the episode."""
        return copy.deepcopy(self)

    def log(self, experiment_logger: ExperimentLogger, run_type: str):
        """Log episode summary (optimized - no per-step logging)."""
        self._log_episode_summary(experiment_logger, run_type)

    def _log_episode_summary(self, experiment_logger: ExperimentLogger, run_type: str):
        """Log only episode summary metrics (much faster than per-step logging)."""
        if self._current_step == 0:
            return  # No data to log

        # Calculate episode summary metrics
        summary_metrics = {}

        # Total episode metrics
        if self._rewards is not None:
            episode_rewards = self._rewards[: self._current_step]
            summary_metrics[f"{run_type}_episode_total_reward"] = float(
                torch.sum(episode_rewards)
            )
            summary_metrics[f"{run_type}_episode_mean_reward"] = float(
                torch.mean(episode_rewards)
            )
            summary_metrics[f"{run_type}_episode_std_reward"] = float(
                torch.std(episode_rewards)
            )

        if self._balance is not None:
            episode_balance = self._balance[: self._current_step]
            summary_metrics[f"{run_type}_episode_final_balance"] = float(
                episode_balance[-1]
            )
            summary_metrics[f"{run_type}_episode_max_balance"] = float(
                torch.max(episode_balance)
            )
            summary_metrics[f"{run_type}_episode_min_balance"] = float(
                torch.min(episode_balance)
            )

        if self._actor_balance is not None:
            episode_actor_balance = self._actor_balance[: self._current_step]
            summary_metrics[f"{run_type}_episode_final_actor_balance_mean"] = float(
                torch.mean(episode_actor_balance[-1])
            )
            summary_metrics[f"{run_type}_episode_max_actor_balance_mean"] = float(
                torch.max(torch.mean(episode_actor_balance, dim=1))
            )

        # Episode length
        summary_metrics[f"{run_type}_episode_length"] = self._current_step

        # Log all summary metrics at once
        experiment_logger.log_metrics(summary_metrics)

        # Increment step counter once for the episode
        experiment_logger.next_step()

    def log_detailed(self, experiment_logger: ExperimentLogger, run_type: str):
        """Log all properties of the episode (detailed per-step logging - use sparingly)."""
        for i_step in range(self._current_step):
            # Create a temporary Result object for logging
            step_result = type(
                "Result",
                (),
                {
                    "rewards": (
                        self._rewards[i_step] if self._rewards is not None else None
                    ),
                    "actions": (
                        self._actions[i_step] if self._actions is not None else None
                    ),
                    "asset_holdings": (
                        self._asset_holdings[i_step]
                        if self._asset_holdings is not None
                        else None
                    ),
                    "actor_balance": (
                        self._actor_balance[i_step]
                        if self._actor_balance is not None
                        else None
                    ),
                    "balance": (
                        self._balance[i_step] if self._balance is not None else None
                    ),
                    "cash": self._cash[i_step] if self._cash is not None else None,
                    "date": self._date[i_step] if self._date is not None else None,
                    "log": lambda self, logger, run_type, tickers: None,  # Placeholder
                },
            )()

            # Import and use the actual Result class for logging
            from qf.results.result import Result

            result = Result(
                rewards=step_result.rewards,
                actions=step_result.actions,
                asset_holdings=step_result.asset_holdings,
                actor_balance=step_result.actor_balance,
                balance=step_result.balance,
                cash=step_result.cash,
                date=step_result.date,
            )
            result.log(experiment_logger, run_type, self.tickers)
            experiment_logger.next_step()
