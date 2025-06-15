from typing import Any, Dict, List, Type

import qf as qf

# Constants
DEFAULT_N_TRIALS = 10
DEFAULT_MAX_TIMESTEPS = 1_000_000
DEFAULT_EVAL_EPISODES = 1
DEFAULT_TRADE_COSTS = 1
DEFAULT_TRADE_COSTS_PERCENT = 0.01
DEFAULT_REWARD_FUNCTION = "sharpe_ratio_w10"


def main() -> None:
    """
    Main function to run hyperparameter optimization for SPQL agent with Sharpe ratio reward function.

    This function:
    1. Sets up the environment configurations
    2. Configures the SPQL agent
    3. Runs hyperparameter optimization
    4. Visualizes the results
    5. Compares with a classic Markowitz agent
    """
    # Environment configurations
    train_env_config: Dict[str, Any] = qf.DEFAULT_TRAIN_ENV_CONFIG
    eval_env_config: Dict[str, Any] = qf.DEFAULT_EVAL_ENV_CONFIG
    env_class: Type = qf.MultiAgentPortfolioEnv

    # Environment adjustments
    env_adjustments: Dict[str, Any] = {
        "reward_function": DEFAULT_REWARD_FUNCTION,
        "trade_costs": DEFAULT_TRADE_COSTS,
        "trade_costs_percent": DEFAULT_TRADE_COSTS_PERCENT,
    }

    # Update training and evaluation environment configurations
    train_env_config = {**train_env_config, **env_adjustments}
    eval_env_config = {**eval_env_config, **env_adjustments}

    # Agent configuration
    agent_config: Dict[str, Any] = qf.DEFAULT_SPQLAGENT_CONFIG

    # List of agent classes to optimize
    agent_classes: List[Type] = [qf.SPQLAgent]

    # Optimization configuration
    optim_config: Dict[str, Any] = {
        "objective": "avg_reward",
        "max_timesteps": DEFAULT_MAX_TIMESTEPS,
        "episodes": DEFAULT_EVAL_EPISODES,
    }

    # Optional: Environment hyperparameter space
    env_hyperparameter_space: Dict[str, Any] = (
        qf.DEFAULT_ENVIRONMENT_HYPERPARAMETER_SPACE_SINGLE_AGENT
    )

    # Create and run hyperparameter optimizer
    optimizer = qf.HyperparameterOptimizer(
        agent_classes=agent_classes,
        env_class=env_class,
        train_env_config=train_env_config,
        eval_env_config=eval_env_config,
        optim_config=optim_config,
        env_hyperparameter_space=env_hyperparameter_space,
    )

    # Run optimization
    results = optimizer.optimize(n_trials=DEFAULT_N_TRIALS)

    # Print results
    print("Best agent class:", results["best_agent_class"].__name__)
    print("Best configuration:", results["best_config"])
    print("Best reward:", results["best_reward"])

    # Visualize results
    optimizer.visualize_results()

    # Compare with classic Markowitz agent
    classic_agent = qf.ClassicOnePeriodMarkovitzAgent(
        env_class, qf.DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_CONFIG
    )
    classic_agent.train()  # Calculate correlation matrix based on risk model
    classic_agent.evaluate(eval_env_config, episodes=DEFAULT_EVAL_EPISODES)
    classic_agent.visualize()


if __name__ == "__main__":
    main()
