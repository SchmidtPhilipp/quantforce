"""
Hamilton-Jacobi-Bellman Portfolio Agent with Costs Configuration Module.

This module provides the HJBPortfolioAgentWithCostsConfig class which defines the
configuration for Hamilton-Jacobi-Bellman (HJB) portfolio optimization agents
that explicitly account for transaction costs. These agents implement continuous-time
portfolio optimization with cost considerations.
"""

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import optuna

from qf.agents.config.base_agent_config import BaseAgentConfig


@dataclass
class HJBPortfolioAgentWithCostsConfig(BaseAgentConfig):
    """
    Configuration for Hamilton-Jacobi-Bellman Portfolio Agents with transaction costs.

    This class defines the configuration for HJB portfolio optimization agents
    that explicitly account for transaction costs in their optimization framework.
    The HJB approach with costs provides optimal portfolio strategies that balance
    expected returns against both risk and transaction costs.

    The HJB framework with costs extends traditional portfolio theory by considering
    continuous-time dynamics, optimal control solutions, and explicit transaction
    cost modeling. This leads to more realistic portfolio strategies that account
    for the impact of trading costs on performance.

    Note: Transaction costs (proportional_cost, fixed_cost) are taken from the
    environment configuration (env.config.trade_cost_percent, env.config.trade_cost_fixed)
    rather than being specified in this config to avoid duplication.

    Attributes:
        type (Literal["hjb_portfolio"]): Agent type identifier.
        rebalancing_period (Optional[int]): Number of steps between portfolio
            rebalancing. None = buy-and-hold, 1 = daily, 7 = weekly, etc.
            Default: np.inf (no rebalancing).
        risk_aversion (float): Relative risk aversion coefficient (γ).
            Higher values indicate more risk-averse preferences. Default: 2.0.
        time_horizon (float): Investment time horizon in days (T).
            The planning period for the optimization. Default: 252 (1 year).
        solver_method (Literal["analytical", "numerical"]): Method for solving
            the HJB equation. "analytical" uses closed-form solutions where
            available, "numerical" uses numerical methods. Default: "analytical".
        dt (float): Time step size for numerical solutions in days.
            Default: 1 (daily steps).
        tolerance (float): Convergence tolerance for numerical solver.
            Default: 1e-6.
        max_iterations (int): Maximum iterations for numerical solver.
            Default: 1000.
        allow_shorting (bool): Whether to allow short positions.
            Default: False (long-only portfolio).
        min_weight (float): Minimum weight per asset constraint.
            Default: 0.0 (no minimum).
        max_weight (float): Maximum weight per asset constraint.
            Default: 1.0 (no maximum).
        leverage_constraint (float): Maximum leverage constraint.
            Default: 1.0 (no leverage).
        estimation_method (str): Method for estimating asset parameters
            (mu and sigma). Default: "sample_cov".
        lookback_window (int): Lookback window for parameter estimation in days.
            Default: 252 (1 year).

    Example:
        >>> from qf.agents.config.modern_agents.hjb_portfolio_agent_with_costs_config import HJBPortfolioAgentWithCostsConfig
        >>>
        >>> # Create a conservative HJB configuration with costs
        >>> conservative_config = HJBPortfolioAgentWithCostsConfig(
        ...     risk_aversion=5.0,  # High risk aversion
        ...     time_horizon=252,   # 1 year horizon
        ...     allow_shorting=False,
        ...     solver_method="analytical",
        ...     rebalancing_period=7  # Weekly rebalancing
        ... )
        >>>
        >>> # Create an aggressive HJB configuration with costs
        >>> aggressive_config = HJBPortfolioAgentWithCostsConfig(
        ...     risk_aversion=1.0,  # Low risk aversion
        ...     time_horizon=126,   # 6 months horizon
        ...     allow_shorting=True,
        ...     solver_method="numerical",
        ...     rebalancing_period=1  # Daily rebalancing
        ... )
        >>>
        >>> # Create a buy-and-hold HJB configuration with costs
        >>> buy_hold_config = HJBPortfolioAgentWithCostsConfig(
        ...     rebalancing_period=None,  # No rebalancing
        ...     risk_aversion=3.0,
        ...     time_horizon=504  # 2 year horizon
        ... )
    """

    type: Literal["hjb_portfolio"] = "hjb_portfolio"
    rebalancing_period: Optional[int] = (
        np.inf
    )  # None = buy-and-hold, 1 = daily, 7 = weekly, etc.

    # Core HJB parameters
    risk_aversion: float = 2.0  # γ - relative risk aversion coefficient
    time_horizon: float = 252  # T - investment time horizon (days)

    # Numerical solution parameters
    solver_method: Literal["analytical", "numerical"] = "analytical"
    dt: float = 1  # 1 Day
    tolerance: float = 1e-6  # Convergence tolerance for numerical solver
    max_iterations: int = 1000  # Maximum iterations for numerical solver

    # Portfolio constraints
    allow_shorting: bool = False  # Whether to allow short positions
    min_weight: float = 0.0  # Minimum weight per asset
    max_weight: float = 1.0  # Maximum weight per asset
    leverage_constraint: float = 1.0  # Maximum leverage

    # Estimation method for asset parameters
    estimation_method: str = "sample_cov"  # Method for estimating mu and sigma
    lookback_window: int = 252  # Lookback window for parameter estimation

    @staticmethod
    def get_default_config() -> "HJBPortfolioAgentWithCostsConfig":
        """
        Get default configuration for HJB Portfolio Agent with costs.

        Returns a configuration with sensible defaults for HJB portfolio
        optimization with transaction costs, including moderate risk aversion
        and analytical solution method.

        Returns:
            HJBPortfolioAgentWithCostsConfig: Default configuration.

        Example:
            >>> config = HJBPortfolioAgentWithCostsConfig.get_default_config()
            >>> print(config.risk_aversion)  # 2.0
            >>> print(config.solver_method)  # "analytical"
        """
        return HJBPortfolioAgentWithCostsConfig()

    @staticmethod
    def get_hyperparameter_space(
        trial: optuna.Trial,
    ) -> "HJBPortfolioAgentWithCostsConfig":
        """
        Get hyperparameter space for Optuna optimization.

        Defines the search space for hyperparameter optimization using Optuna.
        This method is used by the hyperparameter optimization framework to
        automatically search for optimal parameter combinations.

        Args:
            trial (optuna.Trial): Optuna trial object for parameter suggestions.

        Returns:
            HJBPortfolioAgentWithCostsConfig: Configuration with suggested parameters.

        Example:
            >>> import optuna
            >>>
            >>> def objective(trial):
            ...     config = HJBPortfolioAgentWithCostsConfig.get_hyperparameter_space(trial)
            ...     # Use config for training/evaluation
            ...     return performance_score
            >>>
            >>> study = optuna.create_study(direction="maximize")
            >>> study.optimize(objective, n_trials=100)
        """
        return HJBPortfolioAgentWithCostsConfig(
            risk_aversion=trial.suggest_float("risk_aversion", 0.5, 10.0),
            time_horizon=trial.suggest_float("time_horizon", 0.25, 5.0),
            solver_method=trial.suggest_categorical(
                "solver_method", ["analytical", "numerical"]
            ),
            allow_shorting=trial.suggest_categorical("allow_shorting", [True, False]),
            estimation_method=trial.suggest_categorical(
                "estimation_method",
                ["sample_cov", "ledoit_wolf", "ML_brownian_motion_logreturn"],
            ),
            lookback_window=trial.suggest_int(
                "lookback_window", 63, 504
            ),  # 3 months to 2 years
        )
