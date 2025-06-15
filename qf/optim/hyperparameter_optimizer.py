import numpy as np
import optuna
from optuna.visualization import plot_optimization_history, plot_param_importances

import qf


class HyperparameterOptimizer:
    def __init__(
        self,
        agent_classes,
        optim_config=None,
        env_class=None,
        train_env_config=None,
        eval_env_config=None,
        agent_config=None,
        env_hyperparameter_space=None,  # New: Optional hyperparameter space for the environment
    ):
        """
        Initializes the HyperparameterOptimizer.

        Parameters:
            agent_classes (list): List of agent classes to optimize.
            optim_config (dict, optional): Optimization configuration, e.g., objective metric.
            env_class (class, optional): Environment class to use.
            train_env_config (dict, optional): Default training environment configuration.
            eval_env_config (dict, optional): Default evaluation environment configuration.
            agent_config (list, optional): List of agent configurations.
            env_hyperparameter_space (dict, optional): Hyperparameter space for the environment.
        """
        self.agent_classes = agent_classes
        self.env_class = env_class or qf.MultiAgentPortfolioEnv
        self.train_env_config = train_env_config or qf.DEFAULT_TRAIN_ENV_CONFIG
        self.eval_env_config = eval_env_config or qf.DEFAULT_EVAL_ENV_CONFIG
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

        self.best_study = None
        self.all_studies = None

    def _objective(self, trial, agent_class, agent_config):
        """
        Objective function for Optuna.

        Parameters:
            trial (optuna.Trial): Optuna trial object for hyperparameter search.
            agent_class (class): The agent class being optimized.

        Returns:
            float: The objective metric to optimize.
        """
        # Sample agent hyperparameters
        agent_hyperparameters = {}
        for param_name, param_space in agent_class.get_hyperparameter_space().items():
            if param_space["type"] == "float":
                value = trial.suggest_float(
                    param_name, param_space["low"], param_space["high"]
                )
                # Round float values if specified
                if "round" in param_space:
                    value = round(value, param_space["round"])
                agent_hyperparameters[param_name] = value
            elif param_space["type"] == "int":
                value = trial.suggest_int(
                    param_name, param_space["low"], param_space["high"]
                )
                # Round int values if specified
                if "round" in param_space:
                    value = round(value / param_space["round"]) * param_space["round"]
                agent_hyperparameters[param_name] = value
            elif param_space["type"] == "categorical":
                agent_hyperparameters[param_name] = trial.suggest_categorical(
                    param_name, param_space["choices"]
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_space['type']}")

        # Sample environment hyperparameters (if provided)
        env_hyperparameters = {}
        for param_name, param_space in self.env_hyperparameter_space.items():
            if param_space["type"] == "float":
                value = trial.suggest_float(
                    param_name, param_space["low"], param_space["high"]
                )
                # Round float values if specified
                if "round" in param_space:
                    value = round(value, param_space["round"])
                env_hyperparameters[param_name] = value
            elif param_space["type"] == "int":
                value = trial.suggest_int(
                    param_name, param_space["low"], param_space["high"]
                )
                # Round int values if specified
                if "round" in param_space:
                    value = round(value / param_space["round"]) * param_space["round"]
                env_hyperparameters[param_name] = value
            elif param_space["type"] == "categorical":
                env_hyperparameters[param_name] = trial.suggest_categorical(
                    param_name, param_space["choices"]
                )
            else:
                raise ValueError(f"Unsupported parameter type: {param_space['type']}")

        # Merge configurations
        merged_agent_config = {**agent_config, **agent_hyperparameters}
        merged_env_config = {**self.train_env_config, **env_hyperparameters}

        # Set the environment's config_name to reflect the current hyperparameter sweep
        merged_env_config["config_name"] = f"{agent_class.__name__}_" + "_".join(
            [
                f"{k}_{v}"
                for k, v in {**agent_hyperparameters, **env_hyperparameters}.items()
            ]
        )
        self.eval_env_config["config_name"] = merged_env_config["config_name"]

        # Create environment and agent
        env = self.env_class(tensorboard_prefix="TRAIN", config=merged_env_config)
        agent = agent_class(env, config=merged_agent_config)

        # Train the agent
        agent.train(
            total_timesteps=self.optim_config.get(
                "max_timesteps", qf.DEFAULT_MAX_TIMESTEPS
            ),
            use_tqdm=self.optim_config.get("use_tqdm", True),
        )

        # Evaluate the agent
        eval_env = self.env_class(
            tensorboard_prefix="EVAL", config=self.eval_env_config
        )
        rewards = agent.evaluate(
            eval_env, episodes=self.optim_config.get("episodes", 10)
        )

        # Compute the objective metric
        if self.optim_config["objective"] == "avg_reward":
            return np.mean(rewards)
        elif self.optim_config["objective"] == "avg_reward - std_deviation":
            return np.mean(rewards) - np.std(rewards)
        elif callable(self.optim_config["objective"]):
            return self.optim_config["objective"](rewards)
        else:
            raise ValueError(f"Unsupported objective: {self.optim_config['objective']}")

    def optimize(self, n_trials=50):
        """
        Conducts hyperparameter optimization for all agent classes.

        Parameters:
            n_trials (int): Number of optimization trials.

        Returns:
            dict: Best agent class, best hyperparameter configuration, and the corresponding reward.
        """
        best_agent_class = None
        best_config = None
        best_reward = float("-inf")
        best_study = None
        all_studies = []

        for agent_class, agent_config in zip(self.agent_classes, self.agent_config):
            study = optuna.create_study(
                direction="maximize",
                study_name=f"{agent_class.__name__}_hyperparameter_optimization",
            )
            study.optimize(
                lambda trial: self._objective(trial, agent_class, agent_config),
                n_trials=n_trials,
            )

            # Append the study to the list of all studies
            all_studies.append(study)

            # Best trial for the current agent class
            trial = study.best_trial
            print(f"Best trial for {agent_class.__name__}:")
            print(f"Value: {trial.value}")
            print(f"Parameters: {trial.params}")

            # Update the best study if the current study has a higher reward
            if trial.value > best_reward:
                best_reward = trial.value
                best_config = trial.params
                best_agent_class = agent_class
                best_study = study

        self.best_study = best_study
        self.all_studies = all_studies

        return {
            "best_agent_class": best_agent_class,
            "best_config": best_config,
            "best_reward": best_reward,
        }

    def visualize_results(self, renderer="vscode", save_path=None):
        """
        Visualizes the results of the best Optuna study and all studies combined.

        Parameters:
            best_study (optuna.Study): The best Optuna study containing optimization results.
            all_studies (list): List of all Optuna studies containing optimization results.

        Returns:
            None: Displays the plots for optimization history and parameter importance.
        """
        import plotly.io as pio

        pio.renderers.default = "png"
        import os

        if save_path is None:
            save_path = qf.DEFAULT_LOG_DIR + "/hyperparameter_optimization_results"

        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)

        # Plot optimization history for the best study
        opt_history_fig = plot_optimization_history(self.best_study)
        opt_history_fig.write_image(
            os.path.join(save_path, "opt_history_best_study.png")
        )

        # Plot parameter importance for the best study
        param_importance_fig = plot_param_importances(self.best_study)
        param_importance_fig.write_image(
            os.path.join(save_path, "param_importance_best_study.png")
        )

        # print("Parameter Importance Across All Studies:")
        for study in self.all_studies:
            opt_history_fig_all = plot_optimization_history(study)
            opt_history_fig_all.write_image(
                os.path.join(save_path, f"opt_history_study_{study.study_name}.png")
            )

            param_importance_fig_all = plot_param_importances(study)
            param_importance_fig_all.write_image(
                os.path.join(
                    save_path, f"param_importance_study_{study.study_name}.png"
                )
            )
