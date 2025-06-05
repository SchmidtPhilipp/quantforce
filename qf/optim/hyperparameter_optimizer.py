import optuna
import numpy as np
import qf
from optuna.visualization import plot_optimization_history, plot_param_importances


class HyperparameterOptimizer:
    def __init__(self, agent_classes, optim_config=None, env_class=None, train_env_config=None, eval_env_config=None):
        """
        Initialisiert die Hyperparameter-Optimierung.

        Parameters:
            agent_classes (list): Liste der Agent-Klassen, die optimiert werden sollen.
            train_env_config (dict): Konfiguration für die Trainingsumgebung.
            eval_train_env_config (dict): Konfiguration für die Evaluierungsumgebung.
            optim_config (dict, optional): Konfiguration für die Optimierung, z.B. Zielmetrik.
        """
        self.agent_classes = agent_classes
        self.env_class = env_class or qf.MultiAgentPortfolioEnv
        self.train_env_config = train_env_config or qf.DEFAULT_TRAIN_ENV_CONFIG
        self.eval_env_config = eval_env_config or qf.DEFAULT_EVAL_ENV_CONFIG
        self.optim_config = optim_config or {"objective": "avg_reward"}  # Standard-Zielmetrik: Durchschnittliche Belohnung

        self.best_study = None
        self.all_studies = None

    def _objective(self, trial, agent_class, use_tqdm=True):
        """
        Objective-Funktion für Optuna.

        Parameters:
            trial (optuna.Trial): Optuna-Trial-Objekt für die Hyperparameter-Suche.
            agent_class (class): Die Agent-Klasse, die optimiert wird.

        Returns:
            float: Zielmetrik, die optimiert wird.
        """
        # Hyperparameter-Sampling
        hyperparameters = {}
        for param_name, param_space in agent_class.get_hyperparameter_space().items():
            if param_space["type"] == "float":
                hyperparameters[param_name] = trial.suggest_float(param_name, param_space["low"], param_space["high"])
            elif param_space["type"] == "int":
                hyperparameters[param_name] = trial.suggest_int(param_name, param_space["low"], param_space["high"])
            elif param_space["type"] == "categorical":
                hyperparameters[param_name] = trial.suggest_categorical(param_name, param_space["choices"])
            else:
                raise ValueError(f"Unsupported parameter type: {param_space['type']}")

        # Agent-Konfiguration erstellen
        default_config = agent_class.get_default_config()
        merged_config = {**default_config, **hyperparameters}

        # Set the environment's config_name to reflect the current hyperparameter sweep
        self.train_env_config["config_name"] = f"{agent_class.__name__}{'_'.join([f'{k}_{v}' for k, v in hyperparameters.items()])}"
        self.eval_env_config["config_name"] = self.train_env_config["config_name"]

        env = self.env_class(tensorboard_prefix="TRAIN", config=self.train_env_config)
        agent = agent_class(env, config=merged_config)

        # Agent trainieren
        agent.train(total_timesteps=self.optim_config.get("max_timesteps", qf.DEFAULT_MAX_TIMESTEPS), use_tqdm=use_tqdm)

        # Agent evaluieren
        eval_env = self.env_class(tensorboard_prefix="EVAL", config=self.eval_env_config)
        rewards = agent.evaluate(eval_env, episodes=self.optim_config.get("episodes", 10), use_tqdm=use_tqdm)

        # Zielmetrik berechnen
        if self.optim_config["objective"] == "avg_reward":
            return np.mean(rewards)
        elif self.optim_config["objective"] == "avg_reward - std_deviation":
            return np.mean(rewards) - np.std(rewards)
        elif callable(self.optim_config["objective"]):
            return self.optim_config["objective"](rewards)
        else:
            raise ValueError(f"Unsupported objective: {self.optim_config['objective']}")

    def optimize(self, n_trials=50, use_tqdm=True):
        """
        Conducts hyperparameter optimization for all agent classes.

        Parameters:
            n_trials (int): Number of optimization trials.

        Returns:
            dict: Best agent class, best hyperparameter configuration, and the corresponding reward.
            optuna.Study: The best Optuna study with the highest reward.
            list: List of all Optuna studies for all agent classes.
        """
        best_agent_class = None
        best_config = None
        best_reward = float('-inf')
        best_study = None
        all_studies = []

        for agent_class in self.agent_classes:
            study = optuna.create_study(direction="maximize", study_name=f"{agent_class.__name__}_hyperparameter_optimization")
            study.optimize(lambda trial: self._objective(trial, agent_class), n_trials=n_trials)

            # Append the study to the list of all studies
            all_studies.append(study)

            # Best trial for the current agent class
            trial = study.best_trial
            print(f"Beste Trial für {agent_class.__name__}:")
            print(f"Wert: {trial.value}")
            print(f"Parameter: {trial.params}")

            # Update the best study if the current study has a higher reward
            if trial.value > best_reward:
                best_reward = trial.value
                best_config = trial.params
                best_agent_class = agent_class
                best_study = study

        self.best_study = best_study
        self.all_studies = all_studies

        return {"best_agent_class": best_agent_class, "best_config": best_config, "best_reward": best_reward}

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
        pio.renderers.default = 'png'
        import os

        if save_path is None:
            save_path = qf.DEFAULT_LOG_DIR + "/hyperparameter_optimization_results"
            
        # Ensure the directory exists
        os.makedirs(save_path, exist_ok=True)
        
        # Plot optimization history for the best study
        opt_history_fig = plot_optimization_history(self.best_study)
        opt_history_fig.write_image(os.path.join(save_path, "opt_history_best_study.png"))

        # Plot parameter importance for the best study
        param_importance_fig = plot_param_importances(self.best_study)
        param_importance_fig.write_image(os.path.join(save_path, "param_importance_best_study.png"))

        #print("Parameter Importance Across All Studies:")
        for study in self.all_studies:
            opt_history_fig_all = plot_optimization_history(study)
            opt_history_fig_all.write_image(os.path.join(save_path, f"opt_history_study_{study.study_name}.png"))

            param_importance_fig_all = plot_param_importances(study)
            param_importance_fig_all.write_image(os.path.join(save_path, f"param_importance_study_{study.study_name}.png"))

