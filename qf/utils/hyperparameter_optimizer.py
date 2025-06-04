import optuna
import numpy as np


class HyperparameterOptimizer:
    def __init__(self, agent_classes, env_config, eval_env_config, optim_config=None):
        """
        Initialisiert die Hyperparameter-Optimierung.

        Parameters:
            agent_classes (list): Liste der Agent-Klassen, die optimiert werden sollen.
            env_config (dict): Konfiguration für die Trainingsumgebung.
            eval_env_config (dict): Konfiguration für die Evaluierungsumgebung.
            optim_config (dict, optional): Konfiguration für die Optimierung, z.B. Zielmetrik.
        """
        self.agent_classes = agent_classes
        self.env_config = env_config
        self.eval_env_config = eval_env_config
        self.optim_config = optim_config or {"objective": "avg_reward"}  # Standard-Zielmetrik: Durchschnittliche Belohnung

    def _objective(self, trial, agent_class):
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
        for param_name, param_space in agent_class.hyperparameter_space.items():
            if param_space["type"] == "float":
                hyperparameters[param_name] = trial.suggest_float(param_name, param_space["low"], param_space["high"])
            elif param_space["type"] == "int":
                hyperparameters[param_name] = trial.suggest_int(param_name, param_space["low"], param_space["high"])
            elif param_space["type"] == "categorical":
                hyperparameters[param_name] = trial.suggest_categorical(param_name, param_space["choices"])
            else:
                raise ValueError(f"Unsupported parameter type: {param_space['type']}")

        # Agent-Konfiguration erstellen
        default_config = agent_class.default_config
        merged_config = {**default_config, **hyperparameters}

        # Trainingsumgebung initialisieren
        self.env_config["config_name"] = f"{agent_class.__name__}_{trial.number}"
        env = agent_class.env_class(tensorboard_prefix="TRAIN", config=self.env_config)
        agent = agent_class(env, config=merged_config)

        # Agent trainieren
        agent.train(total_timesteps=self.optim_config.get("max_timesteps", 5000))

        # Agent evaluieren
        eval_env = agent_class.env_class(tensorboard_prefix="EVAL", config=self.eval_env_config)
        rewards = agent.evaluate(eval_env, episodes=self.optim_config.get("episodes", 10))

        # Zielmetrik berechnen
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
        Führt die Hyperparameter-Optimierung durch.

        Parameters:
            n_trials (int): Anzahl der Optimierungsversuche.

        Returns:
            dict: Beste Agent-Klasse, beste Hyperparameter-Konfiguration und die zugehörige Belohnung.
            optuna.Study: Die Optuna-Studie mit allen Optimierungsergebnissen.
        """
        best_agent_class = None
        best_config = None
        best_reward = float('-inf')
        best_study = None

        for agent_class in self.agent_classes:
            study = optuna.create_study(direction="maximize")
            study.optimize(lambda trial: self._objective(trial, agent_class), n_trials=n_trials)

            # Beste Trial abrufen
            trial = study.best_trial
            print(f"Beste Trial für {agent_class.__name__}:")
            print(f"Wert: {trial.value}")
            print(f"Parameter: {trial.params}")

            if trial.value > best_reward:
                best_reward = trial.value
                best_config = trial.params
                best_agent_class = agent_class
                best_study = study

        return {"best_agent_class": best_agent_class, "best_config": best_config, "best_reward": best_reward}, best_study