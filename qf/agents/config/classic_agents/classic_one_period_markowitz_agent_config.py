"""
Classic One-Period Markowitz Agent Configuration Module.

This module provides the ClassicOnePeriodMarkowitzAgentConfig class which defines
the configuration for classic Markowitz portfolio optimization agents. These agents
implement the traditional mean-variance optimization framework developed by Harry
Markowitz, with various risk models and optimization targets.
"""

from dataclasses import dataclass
from typing import Literal

import optuna

from qf.agents.config.classic_agents.classic_agent_config import ClassicAgentConfig


@dataclass
class ClassicOnePeriodMarkowitzAgentConfig(ClassicAgentConfig):
    """
    Configuration for classic one-period Markowitz portfolio optimization agents.

    This class defines the configuration for traditional Markowitz portfolio
    optimization agents that implement mean-variance optimization. The agent
    can optimize for different targets (Tangency, MaxExpReturn, MinVariance)
    using various risk models for covariance estimation.

    The Markowitz framework is a cornerstone of modern portfolio theory,
    balancing expected return against portfolio risk through mean-variance
    optimization.

    Attributes:
        type (Literal["classic_one_period_markowitz"]): Agent type identifier.
        target (Literal["Tangency", "MaxExpReturn", "MinVariance"]): Optimization
            target for the portfolio. "Tangency" maximizes Sharpe ratio,
            "MaxExpReturn" maximizes expected return, "MinVariance" minimizes
            portfolio variance. Default: "Tangency".
        risk_model (Literal): Risk model for covariance estimation. Options include:
            - "sample_cov": Sample covariance matrix
            - "exp_cov": Exponentially weighted covariance
            - "ledoit_wolf": Ledoit-Wolf shrinkage estimator
            - "ledoit_wolf_constant_variance": Ledoit-Wolf with constant variance
            - "ledoit_wolf_single_factor": Ledoit-Wolf single factor model
            - "ledoit_wolf_constant_correlation": Ledoit-Wolf constant correlation
            - "oracle_approximating": Oracle approximating shrinkage
            - "ML_brownian_motion_logreturn": ML-based Brownian motion model
            Default: "sample_cov".
        risk_free_rate (float): Risk-free rate for Tangency portfolio optimization.
            Used when target="Tangency". Default: 0.0.
        log_returns (bool): Whether to use log returns for covariance estimation.
            Default: True.

    Example:
        >>> from qf.agents.config.classic_agents.classic_one_period_Markowitz_agent_config import ClassicOnePeriodMarkowitzAgentConfig
        >>>
        >>> # Create a Tangency portfolio configuration
        >>> tangency_config = ClassicOnePeriodMarkowitzAgentConfig(
        ...     target="Tangency",
        ...     risk_model="ledoit_wolf",
        ...     risk_free_rate=0.02  # 2% risk-free rate
        ... )
        >>>
        >>> # Create a minimum variance configuration
        >>> min_var_config = ClassicOnePeriodMarkowitzAgentConfig(
        ...     target="MinVariance",
        ...     risk_model="sample_cov"
        ... )
        >>>
        >>> # Create a maximum expected return configuration
        >>> max_return_config = ClassicOnePeriodMarkowitzAgentConfig(
        ...     target="MaxExpReturn",
        ...     risk_model="exp_cov"
        ... )
    """

    type: Literal["classic_one_period_markowitz"] = "classic_one_period_markowitz"
    target: Literal["Tangency", "MaxExpReturn", "MinVariance"] = (
        "Tangency"  # Optimization target
    )
    risk_model: Literal[
        "sample_cov",
        "exp_cov",
        "ledoit_wolf",
        "ledoit_wolf_constant_variance",
        "ledoit_wolf_single_factor",
        "ledoit_wolf_constant_correlation",
        "oracle_approximating",
        "ML_brownian_motion_logreturn",
    ] = "sample_cov"
    risk_free_rate: float = 0.0  # Risk-free rate for Tangency optimization
    log_returns: bool = False

    @staticmethod
    def get_default_config() -> "ClassicOnePeriodMarkowitzAgentConfig":
        """
        Get default configuration for Markowitz portfolio optimization.

        Returns a configuration with sensible defaults for Tangency portfolio
        optimization using sample covariance estimation.

        Returns:
            ClassicOnePeriodMarkowitzAgentConfig: Default configuration.

        Example:
            >>> config = ClassicOnePeriodMarkowitzAgentConfig.get_default_config()
            >>> print(config.target)  # "Tangency"
            >>> print(config.risk_model)  # "sample_cov"
        """
        return ClassicOnePeriodMarkowitzAgentConfig()

    @staticmethod
    def get_hyperparameter_space(
        trial: optuna.Trial,
    ) -> "ClassicOnePeriodMarkowitzAgentConfig":
        """
        Get hyperparameter space for Optuna optimization.

        Defines the search space for hyperparameter optimization using Optuna.
        This method is used by the hyperparameter optimization framework to
        automatically search for optimal parameter combinations.

        Args:
            trial (optuna.Trial): Optuna trial object for parameter suggestions.

        Returns:
            ClassicOnePeriodMarkowitzAgentConfig: Configuration with suggested parameters.

        Example:
            >>> import optuna
            >>>
            >>> def objective(trial):
            ...     config = ClassicOnePeriodMarkowitzAgentConfig.get_hyperparameter_space(trial)
            ...     # Use config for training/evaluation
            ...     return performance_score
            >>>
            >>> study = optuna.create_study(direction="maximize")
            >>> study.optimize(objective, n_trials=100)
        """
        return ClassicOnePeriodMarkowitzAgentConfig(
            target=trial.suggest_categorical(
                "target", ["Tangency", "MaxExpReturn", "MinVariance"]
            ),
            risk_model=trial.suggest_categorical(
                "risk_model",
                [
                    "sample_cov",
                    "exp_cov",
                    "ledoit_wolf",
                    "ledoit_wolf_constant_variance",
                    "ledoit_wolf_single_factor",
                    "ledoit_wolf_constant_correlation",
                    "oracle_approximating",
                    "ML_brownian_motion_logreturn",
                ],
            ),
            risk_free_rate=trial.suggest_float("risk_free_rate", 0.0, 0.1),
            log_returns=trial.suggest_categorical("log_returns", [True, False]),
        )
