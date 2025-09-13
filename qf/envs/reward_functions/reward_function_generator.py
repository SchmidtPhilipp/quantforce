from qf.envs.reward_functions.absolute_return import AbsoluteReturn
from qf.envs.reward_functions.config.reward_function_config import (
    AbsoluteReturnConfig,
    BaseRewardConfig,
    CostAdjustedSharpeRatioConfig,
    DifferentialSharpeRatioConfig,
    LinearRateOfReturnConfig,
    LogReturnConfig,
)
from qf.envs.reward_functions.cost_adjusted_sharpe_ratio import CostAdjustedSharpeRatio
from qf.envs.reward_functions.differential_sharpe_ratio import DifferentialSharpeRatio
from qf.envs.reward_functions.linear_rate_of_return import LinearRateOfReturn
from qf.envs.reward_functions.log_rate_of_return import LogReturn
from qf.envs.reward_functions.reward_function import RewardFunction


def reward_function_factory(
    config: BaseRewardConfig, n_agents: int, device: str, dataset=None
) -> RewardFunction:
    """
    Factory function to create reward function instances from configuration.

    Args:
        config: Configuration object for the reward function
        n_agents: Number of agents
        device: Device to use for computations
        dataset: Dataset for reward functions that need it (optional)

    Returns:
        RewardFunction: An instance of the specified reward function
    """
    if config.type == "linear_rate_of_return":
        assert isinstance(
            config, LinearRateOfReturnConfig
        ), f"Expected LinearRateOfReturnConfig, got {type(config)}"
        return LinearRateOfReturn(config, n_agents, device)

    elif config.type == "log_return":
        assert isinstance(
            config, LogReturnConfig
        ), f"Expected LogReturnConfig, got {type(config)}"
        return LogReturn(config, n_agents, device)

    elif config.type == "absolute_return":
        assert isinstance(
            config, AbsoluteReturnConfig
        ), f"Expected AbsoluteReturnConfig, got {type(config)}"
        return AbsoluteReturn(config, n_agents, device)

    elif config.type == "differential_sharpe_ratio":
        assert isinstance(
            config, DifferentialSharpeRatioConfig
        ), f"Expected DifferentialSharpeRatioConfig, got {type(config)}"
        return DifferentialSharpeRatio(config, n_agents, device)

    elif config.type == "cost_adjusted_sharpe_ratio":
        assert isinstance(
            config, CostAdjustedSharpeRatioConfig
        ), f"Expected CostAdjustedSharpeRatioConfig, got {type(config)}"
        if dataset is None:
            raise ValueError("CostAdjustedSharpeRatio requires a dataset")
        return CostAdjustedSharpeRatio(config, n_agents, device, dataset)

    else:
        raise ValueError(f"Unknown reward function type: {config.type}")
