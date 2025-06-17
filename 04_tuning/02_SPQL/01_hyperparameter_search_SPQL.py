import os

# Import the root folder of this folder
import sys
from typing import Any, Dict, List, Type

# Add the root folder to the Python path
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), "../" * 2)))

import qf as qf
from qf.utils.logging_utils import setup_print_logging

# Constants
N_TRIALS = 20
MAX_TIMESTEPS = 1_000_000
EVAL_STEPS = 50_000
EVAL_EPISODES = 1
USE_TQDM = True
PRINT_EVAL_METRICS = True

# Debugging Constants
IS_DEBUG = True
if IS_DEBUG:
    N_TRIALS = 2
    MAX_TIMESTEPS = 2
    EVAL_STEPS = 1
    EVAL_EPISODES = 1
    USE_TQDM = True
    PRINT_EVAL_METRICS = True


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

    # Start tensorboard
    qf.start_tensorboard()

    # Setup print logging
    # setup_print_logging()

    # Environment adjustments
    env_adjustments: Dict[str, Any] = {
        "trade_cost_fixed": 1,
        "trade_cost_percent": 0.01,
        "reward_function": "sharpe_ratio_w20",
    }

    # Update training and evaluation environment configurations
    train_env_config = {**train_env_config, **env_adjustments}
    eval_env_config = {**eval_env_config, **env_adjustments}

    # List of agent classes to optimize
    agent_classes: List[Type] = [qf.SPQLAgent]

    # Optimization configuration
    optim_config: Dict[str, Any] = {
        "objective": "avg_reward",
        "max_timesteps": MAX_TIMESTEPS,  # 1 million steps
        "episodes": EVAL_EPISODES,  # 1 episode of evaluation in the end of every training run with the best agent
        "n_eval_steps": EVAL_STEPS,  # Evaluate the agent every 50000 steps during training
        "n_eval_episodes": EVAL_EPISODES,  # Evaluate the agent 1 time during training every n_eval_steps steps
        "use_tqdm": True,  # Use tqdm to show the training progress
        "print_eval_metrics": True,  # Print the evaluation metrics
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
    results = optimizer.optimize(n_trials=20)  # Try 20 different configurations

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
    classic_agent.evaluate(eval_env_config, episodes=1)
    classic_agent.visualize()


if __name__ == "__main__":
    main()
