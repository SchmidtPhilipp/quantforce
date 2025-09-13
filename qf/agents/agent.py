import importlib
import json
import os
import sys
from typing import Optional

import numpy as np
from sympy import use
import torch
from tqdm.auto import tqdm

from qf.agents.config.base_agent_config import BaseAgentConfig
from qf.envs.multi_agent_portfolio_env import MultiAgentPortfolioEnv
from qf.utils.logging_config import get_logger

logger = get_logger(__name__)


from qf.envs.config.env_config import EnvConfig


class Agent:
    def __init__(self, env, config: Optional[BaseAgentConfig]):
        """
        Initializes the agent with the given environment.
        Parameters:

            env: The environment in which the agent will operate.
        """

        if hasattr(config, "seed") and config.seed is not None:
            torch.manual_seed(config.seed)
            np.random.seed(config.seed)

        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.device = env.device
        self.config = config
        self.verbosity = self.config.verbosity

        self.total_timesteps = None

        # Register with environment for type detection
        if hasattr(env, "register_agent"):
            env.register_agent(self)

    def train(
        self,
        total_timesteps: int,
        eval_env=None,
        eval_every_n_steps=None,
        n_eval_episodes=1,
        save_best=True,
        print_eval_metrics=True,
        use_tqdm=True,
        save_checkpoints=True,
    ) -> None:
        """Train the agent with optional evaluation.

        Args:
            total_timesteps: Total number of timesteps to train for
            eval_env: Environment to use for evaluation
            eval_every_n_steps: Number of steps between evaluations
            n_eval_episodes: Number of episodes to evaluate on
            save_best: Whether to save the best model (or final model if no evaluation)
            print_eval_metrics: Whether to print evaluation metrics
            use_tqdm: Whether to use tqdm progress bar
        """
        self.total_timesteps = total_timesteps

        best_model_path = os.path.join(self.env.save_dir, "best_model")
        if eval_env is None or eval_every_n_steps is None:
            # Train without evaluation
            self._train(total_timesteps, use_tqdm)
            # Always save model if save_best is True, even without evaluation
            if save_best:
                self.save(best_model_path)
                self.env.save_data(path=best_model_path)
                if print_eval_metrics:
                    logger.info(f"Final model saved after {total_timesteps} timesteps")
            return

        # Calculate number of evaluation steps
        n_evaluations = total_timesteps // eval_every_n_steps
        best_mean_reward = float("-inf")

        # Training loop with evaluation
        for i in range(n_evaluations):
            # Train for eval_every_n_steps
            self._train(eval_every_n_steps, use_tqdm)

            # Reset metrics of eval_env
            eval_env.metrics.reset()

            # Evaluate
            mean_reward = self.evaluate(
                eval_env=eval_env,
                episodes=n_eval_episodes,
                print_metrics=print_eval_metrics,
            ).mean()

            if save_checkpoints:
                path = os.path.join(eval_env.save_dir, f"checkpoint_{i}")
                self.save(path)
                eval_env.save_data(path=path, save_tracker=False)

            # Save best model if needed
            if save_best and mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                self.save(best_model_path)
                eval_env.save_data(path=best_model_path)
                if print_eval_metrics:
                    logger.info(
                        f"New best model saved with mean reward: {mean_reward:.2f}"
                    )

        # Train remaining steps
        remaining_steps = total_timesteps % eval_every_n_steps
        if remaining_steps > 0:
            self._train(remaining_steps, use_tqdm)

            # If we have evaluation data, check if final model is better
            if save_best and best_mean_reward > float("-inf"):

                # Reset metrics of eval_env
                eval_env.metrics.reset()

                # Evaluate final model to see if it's better
                final_mean_reward = self.evaluate(
                    eval_env=eval_env,
                    episodes=n_eval_episodes,
                    print_metrics=False,  # Don't print metrics for final evaluation
                ).mean()

                if final_mean_reward > best_mean_reward:
                    self.save(best_model_path)
                    eval_env.save_data(path=best_model_path)
                    if print_eval_metrics:
                        logger.info(
                            f"Final model saved as best with mean reward: {final_mean_reward:.2f}"
                        )
            elif save_best:
                # No evaluation was done, but save_best is True, so save the final model
                self.save(best_model_path)
                eval_env.save_data(path=best_model_path)
                if print_eval_metrics:
                    logger.info(f"Final model saved after {total_timesteps} timesteps")

    def evaluate(self, eval_env=None, episodes=1, use_tqdm=True, print_metrics=True):
        """
        Evaluates the agent for a specified number of episodes on the environment.
        Parameters:
            eval_env: The environment used for evaluation.
            episodes (int): Number of episodes to evaluate the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
        Returns:
            np.ndarray: Rewards matrix with shape (episodes, num_steps, n_agents).
        """
        if eval_env is None:
            raise ValueError("eval_env is required for evaluation.")

        # Register agent with evaluation environment for rebalancing period detection
        if hasattr(eval_env, "register_agent"):
            eval_env.register_agent(self)

        rewards_dict = {}  # Dictionary to store rewards for each episode
        max_steps = 0  # Track the maximum number of steps across episodes

        progress = (
            tqdm(
                range(episodes),
                desc=f"Evaluating {self.__class__.__name__}",
                file=sys.__stderr__,
            )
            if use_tqdm
            else range(episodes)
        )

        for episode in progress:
            state, info = eval_env.reset()
            done = False
            episode_reward = []
            step_count = 0

            # Create step progress bar for this episode
            if use_tqdm:
                total_steps = (
                    eval_env.get_timesteps()
                    if hasattr(eval_env, "get_timesteps")
                    else None
                )
                step_progress = tqdm(
                    total=total_steps,
                    desc=f"Episode {episode + 1}",
                    leave=False,
                    file=sys.__stderr__,
                    position=1,
                )

            while not done:
                action = self.act(state)
                output = eval_env.step(action)
                state, reward, done, info = output[0], output[1], output[2], output[-1]

                # rewards should be of [n_agents,]
                if isinstance(reward, np.ndarray) and reward.ndim == 0:
                    reward = reward[np.newaxis]
                elif hasattr(reward, "cpu"):  # Handle tensor rewards
                    reward = reward.cpu().numpy()

                episode_reward.append(reward)
                step_count += 1

                # Update step progress bar
                if use_tqdm:
                    step_progress.update(1)
                    step_progress.set_postfix(
                        {"Steps": step_count, "Reward": f"{sum(reward):.4f}"}
                    )

            # Close step progress bar
            if use_tqdm:
                step_progress.close()

            rewards_dict[episode] = episode_reward  # Store rewards in the dictionary
            max_steps = max(max_steps, len(episode_reward))  # Update max_steps

            if use_tqdm:
                progress.set_postfix(
                    {
                        "Episode": episode + 1,
                        "Total Steps": step_count,
                    }
                )

        # Convert dictionary to a padded matrix of shape (episodes, max_steps, n_agents)
        rewards_matrix = np.array(
            [
                np.pad(
                    # Convert tensor to CPU numpy if needed
                    (
                        rewards_dict[episode].cpu().numpy()
                        if hasattr(rewards_dict[episode], "cpu")
                        else rewards_dict[episode]
                    ),
                    (0, max_steps - len(rewards_dict[episode])),
                    mode="constant",
                    constant_values=0,
                )
                for episode in range(episodes)
            ]
        )
        if print_metrics:
            eval_env.print_metrics()

        avg_reward = np.mean(rewards_matrix)  # remember in rl is the reward R_t not G_t
        std_reward = np.std(rewards_matrix)
        logger.info(
            f"Average reward over {episodes} episodes: {avg_reward}, Standard Deviation: {std_reward}"
        )

        # End of evaluation save data
        eval_env.save_data()
        eval_env.log_metrics(
            experiment_logger=eval_env.experiment_logger,
            run_type=eval_env.environment_name,
        )

        # Save agent config
        if self.config is not None:
            # Save config as a json file
            with open(eval_env.save_dir + "/agent_config.json", "w") as f:
                json.dump(self.config.to_dict(), f, indent=4)

        # Dump agent
        self.save(eval_env.save_dir)

        return rewards_matrix

    def save(self, path):
        """
        Saves the agent's state to a file.
        Parameters:
            path (str): Path to save the agent's state.
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save agent config
        if self.config is not None:
            with open(os.path.join(path, "agent_config.json"), "w") as f:
                json.dump(self.config.to_dict(), f, indent=4)

        # Save environment config
        if hasattr(self.env, "config"):
            with open(os.path.join(path, "train_env_config.json"), "w") as f:
                json.dump(self.env.config, f, indent=4)

        # Save agent class name for loading
        with open(os.path.join(path, "agent_class.txt"), "w") as f:
            f.write(self.__class__.__name__)

        # Call subclass-specific save implementation
        self._save_impl(path)

    def _save_impl(self, path):
        """
        Implementation-specific save method to be overridden by subclasses.
        Parameters:
            path (str): Path to save the agent's state.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def load(self, path):
        """
        Loads the agent's state from a file.
        Parameters:
            path (str): Path to load the agent's state from.
        """
        # Load agent config
        config_path = os.path.join(path, "agent_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                config_dict = json.load(f)

            # Convert config dictionary back to config object
            if hasattr(self, "config") and self.config is not None:
                self.config = self.config.__class__.from_dict(config_dict)
            else:
                # If no config exists, try to create one from the default
                if hasattr(self.__class__, "get_default_config"):
                    default_config = self.__class__.get_default_config()
                    self.config = default_config.__class__.from_dict(config_dict)
                else:
                    self.config = config_dict

        # Call subclass-specific load implementation
        self._load_impl(path)

    def _load_impl(self, path):
        """
        Implementation-specific load method to be overridden by subclasses.
        Parameters:
            path (str): Path to load the agent's state from.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    def visualize(self):
        """
        Visualizes the agent's performance or learned policy.
        This method can be overridden by subclasses to provide specific visualizations.
        """
        return Warning("This method should be implemented by subclasses.")

    @staticmethod
    def get_default_config():
        """
        Returns the default configuration for the agent.
        This method can be overridden by subclasses to provide specific default configurations.
        """
        return Warning("This method should be implemented by subclasses.")

    @staticmethod
    def load_agent(path, env=None, device="cpu"):
        """
        Loads an agent from a specified path. Use this method to load an agent that was saved with the save method.
        Parameters:
            path (str): Path to the saved agent.
            env: The environment to associate with the loaded agent. If None, will be loaded from config.
            device (str): Device to load the agent on ("cpu" or "cuda").
        Returns:
            Agent: An instance of the loaded agent.
        """
        # Load environment config
        if env is None:
            env_config_path = os.path.join(path, "env_config.json")
            if not os.path.exists(env_config_path):
                raise FileNotFoundError(
                    f"Environment config not found at {env_config_path}"
                )

            with open(env_config_path, "r") as f:
                env_config = json.load(f)

            env = MultiAgentPortfolioEnv(
                tensorboard_prefix="LOAD_ENV", config=env_config
            )

        # Load agent class name
        agent_class_path = os.path.join(path, "agent_class.txt")
        if not os.path.exists(agent_class_path):
            raise FileNotFoundError(f"Agent class file not found at {agent_class_path}")

        with open(agent_class_path, "r") as f:
            agent_class_name = f.read().strip()

        # Import and instantiate agent
        agent_module = importlib.import_module("qf")
        agent_class = getattr(agent_module, agent_class_name)

        # Load agent config
        agent_config_path = os.path.join(path, "agent_config.json")
        if os.path.exists(agent_config_path):
            with open(agent_config_path, "r") as f:
                agent_config_dict = json.load(f)

            # Convert config dictionary back to config object
            agent_config = agent_class.get_default_config().from_dict(agent_config_dict)
        else:
            agent_config = None

        # Load agent
        agent = agent_class(env, config=agent_config)

        # Load agent state
        agent.load(path)

        return agent

    def multi_seeded_run(
        self,
        # Training parameters
        total_timesteps: int,
        eval_env_config=None,
        eval_every_n_steps: int = None,
        n_eval_episodes=1,
        # Evaluation parameters
        val_env_config=None,
        val_episodes=1,
        seeds: list[int] = [1, 2, 3],
        save_best=True,
        print_eval_metrics=True,
        use_tqdm=True,
    ):
        """
        Runs the agent with multiple seeds for robustness testing.

        Parameters:
            seeds: List of seeds to run
            total_timesteps: Total number of timesteps for training
            eval_every_n_steps: Number of steps between evaluations
            eval_env: Environment to use for evaluation
            n_eval_episodes: Number of episodes to evaluate on
            val_env: Validation environment (if different from eval_env)
            val_episodes: Number of validation episodes
            save_best: Whether to save the best model
            print_eval_metrics: Whether to print evaluation metrics
            use_tqdm: Whether to use tqdm for progress tracking
            save_checkpoints: Whether to save checkpoints during training

        Returns:
            tuple: (trained_agents, train_runs, eval_runs)
                - trained_agents: List of trained agent instances
                - train_runs: List of training run data
                - eval_runs: List of evaluation run data
        """
        trained_agents = []
        train_runs = []
        train_eval_runs = []
        eval_runs = []

        # configs
        train_env_config = self.env.env_config

        progress = tqdm(seeds, desc="Multi-seed training") if use_tqdm else seeds

        for seed in progress:
            # Set random seeds for reproducibility
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Create seed-specific agent config
            seed_agent_config = self.config.copy()
            seed_agent_config.seed = seed

            train_env_config.config_name = (
                f"Agent_{self.__class__.__name__}_seed_{seed}"
            )
            val_env_config.config_name = f"Agent_{self.__class__.__name__}_seed_{seed}"
            eval_env_config.config_name = f"Agent_{self.__class__.__name__}_seed_{seed}"

            # Reinstantiate training environment for each seed
            train_env = MultiAgentPortfolioEnv(
                environment_name=f"TRAIN",
                n_agents=self.env.n_agents,
                env_config=train_env_config,
            )

            # Reinstantiate evaluation environment for each seed
            eval_env = MultiAgentPortfolioEnv(
                environment_name=f"EVAL",
                n_agents=self.env.n_agents,
                env_config=eval_env_config,
            )

            # Create agent with fresh environment
            agent = self.__class__(train_env, config=seed_agent_config)

            # Train the agent
            agent.train(
                total_timesteps=total_timesteps,
                eval_env=eval_env,
                eval_every_n_steps=eval_every_n_steps,
                n_eval_episodes=n_eval_episodes,
                save_best=save_best,
                print_eval_metrics=print_eval_metrics,
                use_tqdm=use_tqdm,
            )

            # Reload the best agent
            best_model_path = os.path.join(train_env.save_dir, "best_model")
            trained_agent = self.__class__.load_agent(best_model_path, env=train_env)

            if val_env_config is not None:
                # Reinstantiate evaluation environment for each seed
                val_env = MultiAgentPortfolioEnv(
                    environment_name=f"VAL",
                    n_agents=self.env.n_agents,
                    env_config=val_env_config,
                )

                # Evaluate the trained agent
                trained_agent.evaluate(
                    eval_env=val_env,
                    episodes=val_episodes,
                    print_metrics=False,
                )

                eval_runs.append(val_env.data_collector)

            # Collect results
            trained_agents.append(trained_agent)
            train_runs.append(train_env.data_collector)

            train_eval_runs.append(eval_env.data_collector)

            if use_tqdm:
                progress.set_postfix(
                    {
                        "seed": seed,
                        "final_reward": "N/A",  # Could be enhanced to show actual metrics
                    }
                )

        return trained_agents, train_runs, train_eval_runs, eval_runs

    def hyperparameter_optimization_run(
        self,
        # Training parameters
        total_timesteps: int,
        eval_env_config=None,
        eval_every_n_steps: int = None,
        n_eval_episodes=1,
        # Evaluation parameters
        val_env_config=None,
        val_episodes=1,
        # Hyperparameter optimization parameters
        n_trials: int = 20,
        seeds: list[int] = [1],
        save_best=True,
        print_eval_metrics=True,
        use_tqdm=True,
        objective_metric="avg_reward",
        study_name=None,
    ):
        """
        Runs hyperparameter optimization with multiple seeds for each trial.

        Parameters:
            total_timesteps: Total number of timesteps for training
            eval_every_n_steps: Number of steps between evaluations
            eval_env_config: Environment configuration for evaluation
            n_eval_episodes: Number of episodes to evaluate on
            val_env_config: Validation environment configuration
            val_episodes: Number of validation episodes
            n_trials: Number of hyperparameter optimization trials
            seeds: List of seeds to run for each trial
            save_best: Whether to save the best model
            print_eval_metrics: Whether to print evaluation metrics
            use_tqdm: Whether to use tqdm for progress tracking
            objective_metric: Metric to optimize ("avg_reward", "avg_reward - std_deviation", or callable)
            study_name: Name for the Optuna study

        Returns:
            dict: Optimization results containing best parameters, best reward, and study object
        """
        import optuna
        from qf.utils.logging_config import get_logger

        logger = get_logger(__name__)

        if study_name is None:
            study_name = f"{self.__class__.__name__}_hyperparameter_optimization"

        # Create Optuna study
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
        )

        def objective(trial):
            """Objective function for Optuna optimization."""
            # Get hyperparameter space from agent class
            if hasattr(self.__class__, "get_hyperparameter_space"):
                # Use the agent's hyperparameter space method
                trial_config = self.__class__.get_hyperparameter_space(trial)
            else:
                raise NotImplementedError(
                    f"get_hyperparameter_space not implemented for agent {self.__class__.__name__}"
                )

            # Merge with base config
            merged_config = self.config.copy()
            for key, value in trial_config.__dict__.items():
                if hasattr(merged_config, key):
                    setattr(merged_config, key, value)

            # Run multi-seeded training for this hyperparameter configuration
            all_rewards = []

            for seed in seeds:
                # Set random seeds for reproducibility
                np.random.seed(seed)
                torch.manual_seed(seed)

                # Create seed-specific config
                seed_config = merged_config.copy()
                seed_config.seed = seed

                # Create environments with trial-specific names
                train_env_config = self.env.env_config.copy()
                train_env_config.config_name = (
                    f"Agent_{self.__class__.__name__}_Trial_{trial.number}_Seed_{seed}"
                )

                if eval_env_config is not None:
                    eval_env_config.config_name = f"Agent_{self.__class__.__name__}_Trial_{trial.number}_Seed_{seed}"

                if val_env_config is not None:
                    val_env_config.config_name = f"Agent_{self.__class__.__name__}_Trial_{trial.number}_Seed_{seed}"

                # Create training environment
                train_env = MultiAgentPortfolioEnv(
                    environment_name="TRAIN",
                    n_agents=self.env.n_agents,
                    env_config=train_env_config,
                )

                # Create evaluation environment
                if eval_env_config is not None:
                    eval_env = MultiAgentPortfolioEnv(
                        environment_name="EVAL",
                        n_agents=self.env.n_agents,
                        env_config=eval_env_config,
                    )
                else:
                    eval_env = None

                # Create agent with trial configuration
                agent = self.__class__(train_env, config=seed_config)

                # Train the agent
                agent.train(
                    total_timesteps=total_timesteps,
                    eval_env=eval_env,
                    eval_every_n_steps=eval_every_n_steps,
                    n_eval_episodes=n_eval_episodes,
                    save_best=save_best,
                    print_eval_metrics=print_eval_metrics,
                    use_tqdm=use_tqdm,  # Disable tqdm for individual trials
                )

                # Evaluate the best agent
                if val_env_config is not None:
                    val_env = MultiAgentPortfolioEnv(
                        environment_name="VAL",
                        n_agents=self.env.n_agents,
                        env_config=val_env_config,
                    )

                    # Reload the best agent
                    best_model_path = os.path.join(train_env.save_dir, "best_model")
                    trained_agent = self.__class__.load_agent(
                        best_model_path, env=train_env
                    )

                    # Evaluate
                    rewards = trained_agent.evaluate(
                        eval_env=val_env,
                        episodes=val_episodes,
                        print_metrics=False,
                    )

                    all_rewards.extend(rewards)

            # Compute objective metric
            if objective_metric == "avg_reward":
                return np.mean(all_rewards)
            elif objective_metric == "avg_reward - std_deviation":
                return np.mean(all_rewards) - np.std(all_rewards)
            elif callable(objective_metric):
                return objective_metric(all_rewards)
            else:
                raise ValueError(f"Unsupported objective metric: {objective_metric}")

        # Run optimization
        logger.info(f"Starting hyperparameter optimization with {n_trials} trials")
        study.optimize(objective, n_trials=n_trials, show_progress_bar=use_tqdm)

        # Log results
        logger.info(f"Best trial: {study.best_trial.value}")
        logger.info(f"Best parameters: {study.best_trial.params}")

        return {
            "best_params": study.best_trial.params,
            "best_reward": study.best_trial.value,
            "study": study,
            "all_trials": study.trials,
        }

    @staticmethod
    def get_hyperparameter_space():
        """
        Returns the hyperparameters of the agent.
        This method can be overridden by subclasses to provide specific hyperparameters.
        """
        return Warning("This method should be implemented by subclasses.")

    @staticmethod
    def grid_search_run(
        agents_to_compare: list,
        train_env_config: EnvConfig,
        # Training parameters
        total_timesteps: int,
        eval_env_config=None,
        eval_every_n_steps: int = None,
        n_eval_episodes=1,
        # Evaluation parameters
        val_env_config=None,
        val_episodes=1,
        # Grid search parameters
        seeds: list[int] = [1, 2, 3],
        save_best=True,
        print_eval_metrics=True,
        use_tqdm=True,
        save_results=True,
    ):
        """
        Runs grid search through different agent configurations with multiple seeds.

        Parameters:
            agents_to_compare: List of agent configurations to compare.
                Each item should be a tuple: (name, agent_class, config)
            total_timesteps: Total number of timesteps for training
            eval_every_n_steps: Number of steps between evaluations
            eval_env_config: Environment configuration for evaluation
            n_eval_episodes: Number of episodes to evaluate on
            val_env_config: Validation environment configuration
            val_episodes: Number of validation episodes
            seeds: List of seeds to run for each configuration
            save_best: Whether to save the best model
            print_eval_metrics: Whether to print evaluation metrics
            use_tqdm: Whether to use tqdm for progress tracking
            save_results: Whether to save results to file

        Returns:
            dict: Grid search results containing all configurations, their results, and best configuration

        Example:
            >>> import qf
            >>>
            >>> # Define different agent configurations to compare
            >>> agents_to_compare = [
            ...     ("SAC_flat_network", qf.SACAgent, qf.SACConfig.get_default_config()),
            ...     ("SAC_medium_network", qf.SACAgent, qf.SACConfig(
            ...         feature_extractor_config=qf.DefaultNetworks.get_medium_transformer_feature_extractor()
            ...     )),
            ...     ("SAC_large_network", qf.SACAgent, qf.SACConfig(
            ...         feature_extractor_config=qf.DefaultNetworks.get_large_transformer_feature_extractor()
            ...     )),
            ... ]
            >>>
            >>> # Environment configurations
            >>> train_env_config = qf.EnvConfig.get_default_train(trade_cost_percent=0, trade_cost_fixed=0)
            >>> eval_env_config = qf.EnvConfig.get_default_eval(trade_cost_percent=0, trade_cost_fixed=0)
            >>> val_env_config = qf.EnvConfig.get_default_eval(trade_cost_percent=0, trade_cost_fixed=0)
            >>>
            >>> # Run grid search
            >>> results = qf.Agent.grid_search_run(
            ...     agents_to_compare=agents_to_compare,
            ...     total_timesteps=200_000,
            ...     eval_env_config=eval_env_config,
            ...     eval_every_n_steps=10_000,
            ...     n_eval_episodes=1,
            ...     val_env_config=val_env_config,
            ...     val_episodes=1,
            ...     seeds=[1, 2, 3],
            ...     save_best=True,
            ...     print_eval_metrics=True,
            ...     use_tqdm=True,
            ...     save_results=True,
            ... )
            >>>
            >>> # Access results
            >>> print(f"Best configuration: {results['grid_search_summary']['best_configuration']}")
            >>> print(f"Best mean reward: {results['grid_search_summary']['best_mean_reward']:.4f}")
            >>> print(f"Total configurations tested: {results['grid_search_summary']['total_configurations']}")
            >>>
            >>> # Access detailed results for each configuration
            >>> for config_name, config_results in results['configurations'].items():
            ...     print(f"{config_name}: mean_reward={config_results['mean_reward']:.4f}, "
            ...           f"std_reward={config_results['std_reward']:.4f}")
            >>>
            >>> # Example output:
            >>> # Best configuration: SAC_medium_network
            >>> # Best mean reward: 0.1234
            >>> # Total configurations tested: 3
            >>> # SAC_flat_network: mean_reward=0.0987, std_reward=0.0123
            >>> # SAC_medium_network: mean_reward=0.1234, std_reward=0.0098
            >>> # SAC_large_network: mean_reward=0.1156, std_reward=0.0145
        """
        from qf.utils.logging_config import get_logger
        import json
        from datetime import datetime

        logger = get_logger(__name__)

        all_results = {}
        best_config_name = None
        best_reward = float("-inf")
        best_agents = []
        all_train_runs = []
        all_train_eval_runs = []
        all_eval_runs = []

        # Calculate total iterations for progress bar
        total_iterations = len(agents_to_compare) * len(seeds)
        progress = (
            tqdm(total=total_iterations, desc="Grid search") if use_tqdm else None
        )

        for config_name, agent_class, config in agents_to_compare:
            logger.info(f"\n{'='*50}")
            logger.info(f"Testing configuration: {config_name}")
            logger.info(f"{'='*50}")

            config_results = {
                "config_name": config_name,
                "agent_class": agent_class.__name__,
                "config": (
                    config.__dict__ if hasattr(config, "__dict__") else str(config)
                ),
                "seeds": {},
                "mean_reward": 0.0,
                "std_reward": 0.0,
                "best_seed": None,
                "best_reward": float("-inf"),
            }

            config_agents = []
            config_train_runs = []
            config_train_eval_runs = []
            config_eval_runs = []

            for seed in seeds:
                # Set random seeds for reproducibility
                np.random.seed(seed)
                torch.manual_seed(seed)

                # Create seed-specific config
                seed_config = config.copy()
                seed_config.seed = seed

                # Create environments with config-specific names
                # Get environment config from the first agent in the list
                first_agent_class, first_config = (
                    agents_to_compare[0][1],
                    agents_to_compare[0][2],
                )
                temp_env = MultiAgentPortfolioEnv(
                    "TEMP", n_agents=1, env_config=train_env_config
                )
                temp_agent = first_agent_class(temp_env, config=first_config)

                train_env_config_copy = train_env_config.copy()
                train_env_config_copy.config_name = f"{config_name}_seed_{seed}"

                eval_env_config_copy = (
                    eval_env_config.copy() if eval_env_config is not None else None
                )
                if eval_env_config_copy is not None:
                    eval_env_config_copy.config_name = f"{config_name}_seed_{seed}"

                val_env_config_copy = (
                    val_env_config.copy() if val_env_config is not None else None
                )
                if val_env_config_copy is not None:
                    val_env_config_copy.config_name = f"{config_name}_seed_{seed}"

                # Create training environment
                train_env = MultiAgentPortfolioEnv(
                    environment_name="TRAIN",
                    n_agents=temp_agent.env.n_agents,
                    env_config=train_env_config_copy,
                )

                # Create evaluation environment
                if eval_env_config_copy is not None:
                    eval_env = MultiAgentPortfolioEnv(
                        environment_name="EVAL",
                        n_agents=temp_agent.env.n_agents,
                        env_config=eval_env_config_copy,
                    )
                else:
                    eval_env = None

                # Create agent with configuration
                agent = agent_class(train_env, config=seed_config)

                # Train the agent
                agent.train(
                    total_timesteps=total_timesteps,
                    eval_env=eval_env,
                    eval_every_n_steps=eval_every_n_steps,
                    n_eval_episodes=n_eval_episodes,
                    save_best=save_best,
                    print_eval_metrics=print_eval_metrics,
                    use_tqdm=True,
                )

                # Evaluate the best agent
                if val_env_config_copy is not None:
                    val_env = MultiAgentPortfolioEnv(
                        environment_name="VAL",
                        n_agents=temp_agent.env.n_agents,
                        env_config=val_env_config_copy,
                    )

                    # Reload the best agent
                    best_model_path = os.path.join(train_env.save_dir, "best_model")
                    trained_agent = agent_class.load_agent(
                        best_model_path, env=train_env
                    )

                    # Evaluate
                    rewards = trained_agent.evaluate(
                        eval_env=val_env,
                        episodes=val_episodes,
                        print_metrics=False,
                    )

                    seed_reward = np.mean(rewards)
                    config_eval_runs.append(val_env.data_collector)
                else:
                    # If no validation, use training metrics
                    seed_reward = (
                        0.0  # Placeholder - could be enhanced to extract from training
                    )
                    config_eval_runs.append(None)

                # Store seed results
                config_results["seeds"][seed] = {
                    "reward": seed_reward,
                    "rewards": rewards if val_env_config_copy is not None else [],
                }

                # Update best seed for this configuration
                if seed_reward > config_results["best_reward"]:
                    config_results["best_reward"] = seed_reward
                    config_results["best_seed"] = seed

                # Store agent and runs
                config_agents.append(trained_agent)
                config_train_runs.append(train_env.data_collector)
                config_train_eval_runs.append(
                    eval_env.data_collector if eval_env else None
                )

                if progress:
                    progress.update(1)
                    progress.set_postfix(
                        {
                            "config": config_name,
                            "seed": seed,
                            "reward": f"{seed_reward:.4f}",
                        }
                    )

            # Calculate statistics for this configuration
            seed_rewards = [config_results["seeds"][seed]["reward"] for seed in seeds]
            config_results["mean_reward"] = np.mean(seed_rewards)
            config_results["std_reward"] = np.std(seed_rewards)

            # Update best configuration overall
            if config_results["mean_reward"] > best_reward:
                best_reward = config_results["mean_reward"]
                best_config_name = config_name

            # Store configuration results
            all_results[config_name] = config_results
            best_agents.extend(config_agents)
            all_train_runs.extend(config_train_runs)
            all_train_eval_runs.extend(config_train_eval_runs)
            all_eval_runs.extend(config_eval_runs)

            logger.info(f"Configuration {config_name} results:")
            logger.info(f"  Mean reward: {config_results['mean_reward']:.4f}")
            logger.info(f"  Std reward: {config_results['std_reward']:.4f}")
            logger.info(
                f"  Best seed: {config_results['best_seed']} ({config_results['best_reward']:.4f})"
            )

        if progress:
            progress.close()

        # Create summary results
        summary_results = {
            "grid_search_summary": {
                "total_configurations": len(agents_to_compare),
                "total_seeds": len(seeds),
                "best_configuration": best_config_name,
                "best_mean_reward": best_reward,
                "timestamp": datetime.now().isoformat(),
            },
            "configurations": all_results,
            "agents": best_agents,
            "train_runs": all_train_runs,
            "train_eval_runs": all_train_eval_runs,
            "eval_runs": all_eval_runs,
        }

        # Log final results
        logger.info(f"\n{'='*60}")
        logger.info("GRID SEARCH COMPLETED")
        logger.info(f"{'='*60}")
        logger.info(f"Best configuration: {best_config_name}")
        logger.info(f"Best mean reward: {best_reward:.4f}")
        logger.info(f"Total configurations tested: {len(agents_to_compare)}")
        logger.info(f"Total seeds per configuration: {len(seeds)}")

        # Save results if requested
        if save_results:
            results_dir = "grid_search_results"
            os.makedirs(results_dir, exist_ok=True)

            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(
                results_dir, f"grid_search_results_{timestamp}.json"
            )

            # Convert numpy types to native Python types for JSON serialization
            def convert_numpy_types(obj):
                if isinstance(obj, np.integer):
                    return int(obj)
                elif isinstance(obj, np.floating):
                    return float(obj)
                elif isinstance(obj, np.ndarray):
                    return obj.tolist()
                elif isinstance(obj, dict):
                    return {
                        key: convert_numpy_types(value) for key, value in obj.items()
                    }
                elif isinstance(obj, list):
                    return [convert_numpy_types(item) for item in obj]
                else:
                    return obj

            serializable_results = convert_numpy_types(summary_results)

            with open(results_file, "w") as f:
                json.dump(serializable_results, f, indent=2, default=str)

            logger.info(f"Results saved to: {results_file}")

        return summary_results
