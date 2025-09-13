import copy
import itertools
import os
from typing import Any, Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import qf
from qf.networks.default_networks import DefaultNetworks
from qf.utils.logging_config import get_logger

logger = get_logger(__name__)


class GridSearchOptimizer:
    """
    Grid Search Optimizer for hyperparameter optimization of reinforcement learning agents.

    This class performs grid search over hyperparameter spaces to find optimal configurations
    for agent training. It supports both agent and environment hyperparameter optimization.

    The optimizer evaluates all combinations of hyperparameters in the defined spaces and
    returns the configuration that achieves the best objective metric.

    Hyperparameter Space Format:
        Each hyperparameter space is defined as a dictionary with the following structure:

        For float parameters:
            {
                "type": "float",
                "low": float,      # Lower bound
                "high": float,     # Upper bound
                "n_points": int,   # Number of points to generate
                "round": int       # Decimal places to round to (optional)
            }

        For integer parameters:
            {
                "type": "int",
                "low": int,        # Lower bound
                "high": int,       # Upper bound
                "n_points": int,   # Number of points to generate
                "round": int       # Step size (optional)
            }

        For categorical parameters:
            {
                "type": "categorical",
                "choices": list    # List of possible values
            }

    Examples:
        # Basic learning rate grid search for SAC agent
        learning_rate_space = {
            "learning_rate": {
                "type": "float",
                "low": 1e-5,
                "high": 1e-2,
                "n_points": 3,
                "round": 6
            }
        }

        optimizer = GridSearchOptimizer(
            agent_classes=[qf.SACAgent],
            agent_config=[learning_rate_space],
            optim_config={
                "objective": "avg_reward",
                "max_timesteps": 1000,
                "eval_every_n_steps": 500,
                "episodes": 1
            }
        )
        results = optimizer.optimize()

        # Multi-parameter grid search
        agent_space = {
            "learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "n_points": 3},
            "batch_size": {"type": "categorical", "choices": [32, 64, 128]}
        }

        env_space = {
            "rebalancing_period": {"type": "int", "low": 1, "high": 5, "n_points": 3}
        }

        optimizer = GridSearchOptimizer(
            agent_classes=[qf.SACAgent],
            agent_config=[agent_space],
            env_hyperparameter_space=env_space,
            optim_config={"objective": "avg_reward", "max_timesteps": 1000}
        )
        results = optimizer.optimize()

        # Get results and visualize
        df = optimizer.get_results_dataframe()
        optimizer.visualize_results()
        optimizer.save_results()

        # Compare multiple agents
        optimizer = GridSearchOptimizer(
            agent_classes=[qf.SACAgent, qf.PPOAgent],
            agent_config=[
                {"learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "n_points": 3}},
                {"learning_rate": {"type": "float", "low": 1e-4, "high": 1e-2, "n_points": 3}}
            ],
            optim_config={"objective": "avg_reward", "max_timesteps": 1000}
        )
        results = optimizer.optimize()
    """

    def __init__(
        self,
        agent_classes,
        optim_config=None,
        env_hyperparameter_space=None,
        env_class=None,
        train_env_config=None,
        eval_env_config=None,
        agent_config=None,
    ):
        """
        Initialize the GridSearchOptimizer.

        Parameters:
            agent_classes (list): List of agent classes to optimize (e.g., [qf.SACAgent]).
            optim_config (dict, optional): Optimization configuration containing:
                - "objective": Objective metric ("avg_reward", "avg_reward - std_deviation", or callable)
                - "max_timesteps": Maximum training timesteps
                - "eval_every_n_steps": Evaluation frequency during training
                - "n_eval_episodes": Number of episodes for training evaluation
                - "episodes": Number of episodes for final evaluation
                - "use_tqdm": Whether to show progress bars
                - "print_eval_metrics": Whether to print evaluation metrics
            env_hyperparameter_space (dict, optional): Environment hyperparameter space.
            env_class (class, optional): Environment class (default: qf.MultiAgentPortfolioEnv).
            train_env_config (dict, optional): Default training environment configuration.
            eval_env_config (dict, optional): Default evaluation environment configuration.
            agent_config (list, optional): List of agent hyperparameter spaces, one per agent class.
                Each item should be a dict defining the hyperparameter space for that agent.
                If None, uses default configurations for all agents.
                If empty dict {}, disables hyperparameter search for that agent.

        Raises:
            ValueError: If agent_config length doesn't match agent_classes length.
        """
        self.agent_classes = agent_classes
        self.env_class = env_class or qf.MultiAgentPortfolioEnv
        self.train_env_config = train_env_config or qf.EnvConfig.get_default_train()
        self.eval_env_config = eval_env_config or qf.EnvConfig.get_default_eval()
        self.optim_config = optim_config or {"objective": "avg_reward"}
        self.agent_config = agent_config
        self.env_hyperparameter_space = env_hyperparameter_space or {}

        # Validate agent_config
        if agent_config is None:
            self.agent_config = [
                agent_class.get_default_config() for agent_class in self.agent_classes
            ]
        elif isinstance(self.agent_config, list):
            if len(self.agent_config) != len(self.agent_classes):
                raise ValueError(
                    "agent_config should be a list of dictionaries, one for each agent class."
                )
            for config in self.agent_config:
                if not isinstance(config, dict):
                    raise ValueError(
                        "Each item in agent_config should be a dictionary for the corresponding agent class."
                    )
        else:
            raise ValueError(
                "agent_config should be a list of dictionaries, one for each agent class."
            )

        self.results = {}
        self.best_results = {}

    def _generate_grid_points(self, hyperparameter_space):
        """
        Generate all possible combinations of hyperparameters for grid search.

        Parameters:
            hyperparameter_space (dict): Dictionary defining the hyperparameter space.

        Returns:
            list: List of dictionaries, each containing one combination of hyperparameters.
        """
        if not isinstance(hyperparameter_space, dict):
            raise ValueError(
                f"hyperparameter_space must be a dictionary, got {type(hyperparameter_space)}"
            )

        if not hyperparameter_space:
            return [{}]

        param_names = []
        param_values = []

        for param_name, param_space in hyperparameter_space.items():
            if not isinstance(param_space, dict):
                raise ValueError(
                    f"Parameter space for '{param_name}' must be a dictionary with 'type' key, "
                    f"got {type(param_space)}: {param_space}"
                )

            if "type" not in param_space:
                raise ValueError(
                    f"Parameter space for '{param_name}' must contain 'type' key, "
                    f"got keys: {list(param_space.keys())}"
                )

            param_names.append(param_name)

            if param_space["type"] == "float":
                # Generate evenly spaced values between low and high
                if "low" not in param_space or "high" not in param_space:
                    raise ValueError(
                        f"Float parameter '{param_name}' must have 'low' and 'high' keys"
                    )

                values = np.linspace(
                    param_space["low"],
                    param_space["high"],
                    param_space.get("n_points", 5),
                )
                # Round if specified
                if "round" in param_space:
                    values = np.round(values, param_space["round"])
                param_values.append(values)

            elif param_space["type"] == "int":
                # Generate evenly spaced integer values
                if "low" not in param_space or "high" not in param_space:
                    raise ValueError(
                        f"Int parameter '{param_name}' must have 'low' and 'high' keys"
                    )

                values = np.linspace(
                    param_space["low"],
                    param_space["high"],
                    param_space.get("n_points", 5),
                )
                values = np.round(values).astype(int)
                # Round to step if specified
                if "round" in param_space:
                    values = (
                        np.round(values / param_space["round"]) * param_space["round"]
                    )
                param_values.append(values)

            elif param_space["type"] == "categorical":
                if "choices" not in param_space:
                    raise ValueError(
                        f"Categorical parameter '{param_name}' must have 'choices' key"
                    )
                param_values.append(param_space["choices"])

            else:
                raise ValueError(f"Unsupported parameter type: {param_space['type']}")

        # Generate all combinations
        combinations = list(itertools.product(*param_values))
        grid_points = []

        for combo in combinations:
            grid_point = dict(zip(param_names, combo))
            grid_points.append(grid_point)

        return grid_points

    def _compute_objective(self, rewards, objective):
        """
        Compute the objective metric from rewards.

        Parameters:
            rewards (list): List of reward values.
            objective (str or callable): Objective function to use.

        Returns:
            float: The computed objective value.
        """
        if objective == "avg_reward":
            return np.mean(rewards)
        elif objective == "avg_reward - std_deviation":
            return np.mean(rewards) - np.std(rewards)
        elif callable(objective):
            return objective(rewards)
        else:
            raise ValueError(f"Unsupported objective: {objective}")

    def _evaluate_configuration(
        self,
        agent_class,
        agent_config,
        agent_hyperparameters,
        env_hyperparameters,
        trial_number,
    ):
        """
        Evaluate a specific hyperparameter configuration.

        Parameters:
            agent_class (class): The agent class to evaluate.
            agent_config (dict): Base agent configuration.
            agent_hyperparameters (dict): Agent hyperparameters to test.
            env_hyperparameters (dict): Environment hyperparameters to test.
            trial_number (int): Trial number for logging.

        Returns:
            float: The objective metric value.
        """
        # Merge configurations
        if isinstance(agent_config, dict):
            # agent_config is a hyperparameter space, use default config as base
            base_agent_config = agent_class.get_default_config()
            # Create a new config object with updated hyperparameters
            merged_agent_config = copy.deepcopy(base_agent_config)
            for key, value in agent_hyperparameters.items():
                if hasattr(merged_agent_config, key):
                    setattr(merged_agent_config, key, value)
        else:
            # agent_config is a config object, merge with hyperparameters
            merged_agent_config = copy.deepcopy(agent_config)
            for key, value in agent_hyperparameters.items():
                if hasattr(merged_agent_config, key):
                    setattr(merged_agent_config, key, value)

        # For environment config, we need to create new EnvConfig objects
        # since the environment expects EnvConfig objects, not dictionaries
        merged_env_config = copy.deepcopy(self.train_env_config)
        merged_eval_env_config = copy.deepcopy(self.eval_env_config)

        # Update config names with smart naming for complex objects
        agent_params_str = "_".join(
            [
                f"{k}-{self._get_short_name(v)}".replace("_", "-")
                for k, v in agent_hyperparameters.items()
            ]
        )
        env_params_str = "_".join(
            [
                f"{k}-{self._get_short_name(v)}".replace("_", "-")
                for k, v in env_hyperparameters.items()
            ]
        )

        if agent_params_str and env_params_str:
            config_name = f"{agent_class.__name__}_{agent_params_str}_{env_params_str}"
        elif agent_params_str:
            config_name = f"{agent_class.__name__}_{agent_params_str}"
        elif env_params_str:
            config_name = f"{agent_class.__name__}_{env_params_str}"
        else:
            config_name = f"{agent_class.__name__}"

        merged_env_config.config_name = config_name
        merged_eval_env_config.config_name = config_name

        # Apply environment hyperparameters if any
        for key, value in env_hyperparameters.items():
            if hasattr(merged_env_config, key):
                setattr(merged_env_config, key, value)
            if hasattr(merged_eval_env_config, key):
                setattr(merged_eval_env_config, key, value)

        # Create environment and agent
        env = self.env_class(
            environment_name=f"TRAIN", n_agents=1, env_config=merged_env_config
        )
        train_eval_env = self.env_class(
            environment_name=f"TRAIN_EVAL",
            n_agents=1,
            env_config=merged_eval_env_config,
        )
        agent = agent_class(env, config=merged_agent_config)

        # Train the agent
        agent.train(
            total_timesteps=self.optim_config.get("max_timesteps", 10000),
            use_tqdm=self.optim_config.get("use_tqdm", True),
            save_best=True,  # required
            eval_env=train_eval_env,
            eval_every_n_steps=(
                self.optim_config["eval_every_n_steps"]
                if "eval_every_n_steps" in self.optim_config
                else None
            ),
            n_eval_episodes=(
                self.optim_config["n_eval_episodes"]
                if "n_eval_episodes" in self.optim_config
                else None
            ),
            print_eval_metrics=(
                self.optim_config["print_eval_metrics"]
                if "print_eval_metrics" in self.optim_config
                else False
            ),
        )

        # Evaluate the best agent
        eval_env = self.env_class(
            environment_name=f"EVAL", n_agents=1, env_config=merged_eval_env_config
        )

        # Reload the best agent
        try:
            agent = agent_class.load_agent(
                os.path.join(env.save_dir, "best_model"), env=env
            )
        except Exception as e:
            logger.warning(
                f"Could not load the best agent: {e}. Using the last trained agent."
            )

        rewards = agent.evaluate(
            eval_env, episodes=self.optim_config.get("episodes", 1)
        )

        # Compute the objective metric
        return self._compute_objective(rewards, self.optim_config["objective"])

    def _get_short_name(self, value):
        """
        Generate a short, readable name for complex objects.

        Args:
            value: The value to generate a name for

        Returns:
            str: A short, readable name
        """
        # Handle basic types first (most specific checks)
        if isinstance(value, (int, float)):
            return str(value)
        elif isinstance(value, str):
            # Truncate long strings
            if len(value) > 20:
                return value[:20] + "..."
            return value
        elif isinstance(value, (list, tuple)):
            # For lists/tuples, use the first few elements
            if len(value) > 0:
                return f"list-{len(value)}"
            else:
                return "empty-list"

        # Handle NetworkConfig objects
        if hasattr(value, "__class__") and "NetworkConfig" in str(value.__class__):
            # Extract meaningful info from NetworkConfig
            if hasattr(value, "layers") and value.layers:
                # Get the first layer type as identifier
                first_layer = value.layers[0]
                if hasattr(first_layer, "type"):
                    layer_type = first_layer.type
                    if layer_type == "transformer":
                        # For transformers, include d_model and nhead
                        if hasattr(first_layer, "d_model") and hasattr(
                            first_layer, "nhead"
                        ):
                            return f"transformer-{first_layer.d_model}d-{first_layer.nhead}h"
                        else:
                            return "transformer"
                    elif layer_type == "linear":
                        # For linear layers, include output features
                        if hasattr(first_layer, "out_features"):
                            return f"linear-{first_layer.out_features}"
                        else:
                            return "linear"
                    elif layer_type == "lstm":
                        # For LSTM, include hidden size
                        if hasattr(first_layer, "hidden_size"):
                            return f"lstm-{first_layer.hidden_size}"
                        else:
                            return "lstm"
                    else:
                        return layer_type
                else:
                    return "network"
            else:
                return "empty-network"

        # Handle other complex objects
        if hasattr(value, "__class__"):
            class_name = value.__class__.__name__
            # Remove common prefixes/suffixes
            if class_name.endswith("Config"):
                class_name = class_name[:-6]  # Remove 'Config'
            return class_name.lower()

        # Fallback: use string representation but truncate
        str_repr = str(value)
        if len(str_repr) > 30:
            return str_repr[:30] + "..."
        return str_repr

    def optimize(self):
        """
        Conducts grid search optimization for all agent classes.

        This method evaluates all combinations of hyperparameters defined in the agent and
        environment hyperparameter spaces. For each combination, it trains an agent and
        evaluates its performance using the specified objective metric.

        The method returns the best configuration found across all agents and hyperparameter
        combinations.

        Returns:
            dict: Dictionary containing:
                - "best_agent_class": The agent class that achieved the best performance
                - "best_config": Dictionary of the best hyperparameter configuration
                - "best_reward": The objective metric value achieved by the best configuration

        Example:
            results = optimizer.optimize()
            print(f"Best agent: {results['best_agent_class'].__name__}")
            print(f"Best learning rate: {results['best_config']['learning_rate']}")
            print(f"Best reward: {results['best_reward']}")
        """
        best_agent_class = None
        best_config = None
        best_reward = float("-inf")

        for agent_class, agent_config in zip(self.agent_classes, self.agent_config):
            logger.info(f"Starting grid search for {agent_class.__name__}")

            # Handle different agent_config types
            if isinstance(agent_config, dict):
                # agent_config is a hyperparameter space dict
                if agent_config == {}:
                    agent_grid_points = [{}]  # Only use default config
                    logger.info(
                        "Agent hyperparameter search disabled - using default config only"
                    )
                else:
                    agent_grid_points = self._generate_grid_points(agent_config)
                    logger.info(
                        f"Generated {len(agent_grid_points)} agent hyperparameter combinations"
                    )
            else:
                # agent_config is a config object, use default hyperparameter space
                agent_grid_points = self._generate_grid_points(
                    agent_class.get_hyperparameter_space()
                )
                logger.info(
                    f"Using default hyperparameter space with {len(agent_grid_points)} combinations"
                )

            # Generate grid points for environment hyperparameters
            env_grid_points = self._generate_grid_points(self.env_hyperparameter_space)

            # If no environment hyperparameters, create a single empty config
            if not env_grid_points:
                env_grid_points = [{}]

            # Generate all combinations of agent and environment configurations
            all_combinations = list(
                itertools.product(agent_grid_points, env_grid_points)
            )

            logger.info(f"Total combinations to evaluate: {len(all_combinations)}")

            agent_results = []

            for i, (agent_params, env_params) in enumerate(all_combinations):
                logger.info(f"Evaluating combination {i+1}/{len(all_combinations)}")
                logger.info(f"Agent params: {agent_params}")
                logger.info(f"Env params: {env_params}")

                #                try:
                reward = self._evaluate_configuration(
                    agent_class, agent_config, agent_params, env_params, i
                )

                result = {
                    "agent_params": agent_params,
                    "env_params": env_params,
                    "reward": reward,
                    "combination_id": i,
                }
                agent_results.append(result)

                logger.info(f"Reward: {reward}")

                # Update best if this is better
                if reward > best_reward:
                    best_reward = reward
                    best_config = {**agent_params, **env_params}
                    best_agent_class = agent_class

            #                except Exception as e:
            #                    logger.error(f"Error evaluating combination {i}: {e}")
            #                    continue

            # Store results for this agent
            self.results[agent_class.__name__] = agent_results

            # Find best result for this agent
            if agent_results:
                best_agent_result = max(agent_results, key=lambda x: x["reward"])
                self.best_results[agent_class.__name__] = best_agent_result

                logger.info(f"Best result for {agent_class.__name__}:")
                logger.info(f"Reward: {best_agent_result['reward']}")
                logger.info(f"Parameters: {best_agent_result['agent_params']}")
                logger.info(f"Env Parameters: {best_agent_result['env_params']}")

        return {
            "best_agent_class": best_agent_class,
            "best_config": best_config,
            "best_reward": best_reward,
        }

    def get_results_dataframe(self, agent_class_name=None):
        """
        Convert results to a pandas DataFrame for analysis.

        This method converts the grid search results into a pandas DataFrame for easy
        analysis and visualization. The DataFrame includes all hyperparameter combinations
        and their corresponding performance metrics.

        Parameters:
            agent_class_name (str, optional): Specific agent class to get results for.
                                            If None, returns results for all agents.

        Returns:
            pd.DataFrame: DataFrame containing all results with columns:
                - "reward": The objective metric value for each configuration
                - "combination_id": Unique identifier for each configuration
                - "agent_*": Agent hyperparameters (e.g., "agent_learning_rate")
                - "env_*": Environment hyperparameters (e.g., "env_rebalancing_period")
                - "agent_class": Agent class name (if multiple agents)

        Example:
            # Get results for specific agent
            df = optimizer.get_results_dataframe("SACAgent")
            print(df[["agent_learning_rate", "reward"]].sort_values("reward", ascending=False))

            # Get all results
            df = optimizer.get_results_dataframe()
            print(df.groupby("agent_class")["reward"].mean())
        """
        if agent_class_name is not None:
            if agent_class_name not in self.results:
                raise ValueError(
                    f"No results found for agent class: {agent_class_name}"
                )
            agent_results = self.results[agent_class_name]
        else:
            # Combine results from all agents
            agent_results = []
            for agent_name, results in self.results.items():
                for result in results:
                    result_copy = result.copy()
                    result_copy["agent_class"] = agent_name
                    agent_results.append(result_copy)

        if not agent_results:
            return pd.DataFrame()

        # Convert to DataFrame
        df = pd.DataFrame(agent_results)

        # Expand agent_params and env_params into separate columns
        if "agent_params" in df.columns:
            agent_params_df = pd.json_normalize(df["agent_params"])
            agent_params_df.columns = [
                f"agent_{col}" for col in agent_params_df.columns
            ]
            df = pd.concat([df.drop("agent_params", axis=1), agent_params_df], axis=1)

        if "env_params" in df.columns:
            env_params_df = pd.json_normalize(df["env_params"])
            env_params_df.columns = [f"env_{col}" for col in env_params_df.columns]
            df = pd.concat([df.drop("env_params", axis=1), env_params_df], axis=1)

        return df

    def visualize_results(self, save_path=None, agent_class_name=None):
        """
        Visualizes the grid search results.

        Parameters:
            save_path (str, optional): Path to save the visualization plots.
            agent_class_name (str, optional): Specific agent class to visualize.
                                            If None, visualizes all agents.

        Returns:
            None: Saves the plots to the specified path.
        """
        if save_path is None:
            save_path = qf.DEFAULT_LOG_DIR + "/grid_search_results"

        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Get results DataFrame
        df = self.get_results_dataframe(agent_class_name)

        if df.empty:
            logger.warning("No results to visualize")
            return

        # Create visualizations
        self._create_heatmap(df, save_path, agent_class_name)
        self._create_scatter_plots(df, save_path, agent_class_name)
        self._create_summary_plots(df, save_path, agent_class_name)

    def _create_heatmap(self, df, save_path, agent_class_name):
        """Create heatmap visualization for 2D parameter combinations."""
        # Find numeric parameter columns
        param_cols = [
            col
            for col in df.columns
            if col.startswith(("agent_", "env_")) and col not in ["agent_class"]
        ]
        numeric_cols = df[param_cols].select_dtypes(include=[np.number]).columns

        if len(numeric_cols) >= 2:
            # Create heatmap for the first two numeric parameters
            param1, param2 = numeric_cols[0], numeric_cols[1]

            plt.figure(figsize=(10, 8))
            pivot_table = df.pivot_table(
                values="reward", index=param1, columns=param2, aggfunc="mean"
            )

            sns.heatmap(pivot_table, annot=True, fmt=".3f", cmap="viridis")
            plt.title(f"Grid Search Results Heatmap\n{param1} vs {param2}")
            plt.tight_layout()

            filename = f"heatmap_{param1}_vs_{param2}"
            if agent_class_name:
                filename = f"{agent_class_name}_{filename}"
            plt.savefig(
                os.path.join(save_path, f"{filename}.png"), dpi=300, bbox_inches="tight"
            )
            plt.close()

    def _create_scatter_plots(self, df, save_path, agent_class_name):
        """Create scatter plots for parameter relationships."""
        param_cols = [
            col
            for col in df.columns
            if col.startswith(("agent_", "env_")) and col not in ["agent_class"]
        ]
        numeric_cols = df[param_cols].select_dtypes(include=[np.number]).columns

        if len(numeric_cols) >= 2:
            # Create scatter plot matrix
            fig, axes = plt.subplots(
                len(numeric_cols), len(numeric_cols), figsize=(15, 15)
            )

            for i, param1 in enumerate(numeric_cols):
                for j, param2 in enumerate(numeric_cols):
                    if i == j:
                        # Histogram on diagonal
                        axes[i, j].hist(df[param1], bins=20, alpha=0.7)
                        axes[i, j].set_title(param1)
                    else:
                        # Scatter plot
                        scatter = axes[i, j].scatter(
                            df[param1], df[param2], c=df["reward"], cmap="viridis"
                        )
                        axes[i, j].set_xlabel(param1)
                        axes[i, j].set_ylabel(param2)

                        # Add colorbar
                        if i == 0 and j == len(numeric_cols) - 1:
                            cbar = plt.colorbar(scatter, ax=axes[i, j])
                            cbar.set_label("Reward")

            plt.suptitle("Parameter Relationships and Reward Distribution")
            plt.tight_layout()

            filename = "scatter_matrix"
            if agent_class_name:
                filename = f"{agent_class_name}_{filename}"
            plt.savefig(
                os.path.join(save_path, f"{filename}.png"), dpi=300, bbox_inches="tight"
            )
            plt.close()

    def _create_summary_plots(self, df, save_path, agent_class_name):
        """Create summary plots showing reward distribution and best configurations."""
        # Reward distribution
        plt.figure(figsize=(12, 8))

        plt.subplot(2, 2, 1)
        plt.hist(df["reward"], bins=20, alpha=0.7, edgecolor="black")
        plt.title("Reward Distribution")
        plt.xlabel("Reward")
        plt.ylabel("Frequency")

        # Top configurations
        plt.subplot(2, 2, 2)
        top_config = df.nlargest(10, "reward")
        plt.barh(range(len(top_config)), top_config["reward"])
        plt.yticks(
            range(len(top_config)), [f"Config {i+1}" for i in range(len(top_config))]
        )
        plt.title("Top 10 Configurations")
        plt.xlabel("Reward")

        # Parameter importance (correlation with reward)
        plt.subplot(2, 2, 3)
        param_cols = [
            col
            for col in df.columns
            if col.startswith(("agent_", "env_")) and col not in ["agent_class"]
        ]
        numeric_cols = df[param_cols].select_dtypes(include=[np.number]).columns

        if len(numeric_cols) > 0:
            correlations = []
            param_names = []
            for param in numeric_cols:
                corr = df[param].corr(df["reward"])
                if not np.isnan(corr):
                    correlations.append(abs(corr))
                    param_names.append(param)

            if correlations:
                plt.barh(range(len(correlations)), correlations)
                plt.yticks(range(len(correlations)), param_names)
                plt.title("Parameter Importance (|Correlation with Reward|)")
                plt.xlabel("|Correlation|")

        # Best configuration details
        plt.subplot(2, 2, 4)
        best_config = df.loc[df["reward"].idxmax()]
        plt.text(
            0.1,
            0.9,
            f"Best Reward: {best_config['reward']:.3f}",
            transform=plt.gca().transAxes,
            fontsize=12,
        )

        y_pos = 0.8
        for param in param_cols:
            if param in best_config and not pd.isna(best_config[param]):
                plt.text(
                    0.1,
                    y_pos,
                    f"{param}: {best_config[param]}",
                    transform=plt.gca().transAxes,
                    fontsize=10,
                )
                y_pos -= 0.05

        plt.title("Best Configuration Details")
        plt.axis("off")

        plt.tight_layout()

        filename = "summary_plots"
        if agent_class_name:
            filename = f"{agent_class_name}_{filename}"
        plt.savefig(
            os.path.join(save_path, f"{filename}.png"), dpi=300, bbox_inches="tight"
        )
        plt.close()

    def save_results(self, save_path=None):
        """
        Save the grid search results to CSV files.

        Parameters:
            save_path (str, optional): Path to save the results.
        """
        if save_path is None:
            save_path = qf.DEFAULT_LOG_DIR + "/grid_search_results"

        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Save results for each agent
        for agent_name, results in self.results.items():
            df = self.get_results_dataframe(agent_name)
            df.to_csv(
                os.path.join(save_path, f"{agent_name}_grid_search_results.csv"),
                index=False,
            )

        # Save combined results
        combined_df = self.get_results_dataframe()
        if not combined_df.empty:
            combined_df.to_csv(
                os.path.join(save_path, "all_agents_grid_search_results.csv"),
                index=False,
            )

        # Save best results summary
        best_summary = []
        for agent_name, best_result in self.best_results.items():
            summary = {
                "agent_class": agent_name,
                "best_reward": best_result["reward"],
                "combination_id": best_result["combination_id"],
            }
            # Add agent parameters
            for param, value in best_result["agent_params"].items():
                summary[f"agent_{param}"] = value
            # Add environment parameters
            for param, value in best_result["env_params"].items():
                summary[f"env_{param}"] = value
            best_summary.append(summary)

        if best_summary:
            best_df = pd.DataFrame(best_summary)
            best_df.to_csv(
                os.path.join(save_path, "best_configurations_summary.csv"), index=False
            )

        logger.info(f"Results saved to {save_path}")
