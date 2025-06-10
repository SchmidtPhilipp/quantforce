# Import the root folder of this folder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import qf as qf
from itertools import product



def main():
    log_dir = "01_classic_agents/runs"

    qf.start_tensorboard(logdir=log_dir, port=6001)
    
    # Make product over all configs of the hyperparameterspace dict
    # qf.DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZ_HYPERPARAMETER_SPACE
    param_names = list(qf.DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZ_HYPERPARAMETER_SPACE.keys())
    param_values = [qf.DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZ_HYPERPARAMETER_SPACE[name]["choices"] for name in param_names]

    # Erstelle Kombinationen als Dictionaries
    for combination in (dict(zip(param_names, values)) for values in product(*param_values)):
        # Erstelle den Konfigurationsnamen basierend auf der Kombination
        config_name = "ClassicOnePeriodMarkovitzAgent_" + "_".join([f"{key}={value}" for key, value in combination.items()])

        # Aktualisiere die Konfigurationswerte
        qf.DEFAULT_EVAL_ENV_CONFIG["log_dir"] = log_dir
        qf.DEFAULT_TRAIN_ENV_CONFIG["log_dir"] = log_dir
        qf.DEFAULT_TRAIN_ENV_CONFIG["config_name"] = config_name 
        qf.DEFAULT_EVAL_ENV_CONFIG["config_name"] = qf.DEFAULT_TRAIN_ENV_CONFIG["config_name"]

        # Erstelle die Umgebung
        env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="TRAIN", config=qf.DEFAULT_TRAIN_ENV_CONFIG)

        # Kombiniere die Standardkonfiguration mit der aktuellen Kombination
        config_dict = {**qf.DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZ_HYPERPARAMETER_SPACE, **combination}

        # Erstelle den Agenten
        agent = qf.ClassicOnePeriodMarkovitzAgent(env=env, config=config_dict)

        # Führe das Training durch
        agent.train(total_timesteps=500_000, use_tqdm=True)

        # Evaluierung
        eval_env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="EVAL", config=qf.DEFAULT_EVAL_ENV_CONFIG)
        agent.evaluate(eval_env, episodes=1)#




if __name__ == "__main__":
    main()