import qf

def main():
    # Training und Evaluierungsumgebungen
    train_env_config = qf.DEFAULT_TRAIN_ENV_CONFIG
    eval_env_config = qf.DEFAULT_EVAL_ENV_CONFIG

    DQN = False         # Set to True if you want to include DQNAgent in the search, False only the default config will be run.
    SAC = False         # Set to True if you want to include SACAgent in the search, False only the default config will be run. 
    TD3 = False         # Set to True if you want to include TD3Agent in the search, False only the default config will be run.
    Tangency = False    # Set to True if you want to include TangencyAgent in the search, False only the default config will be run.

    Classic = True      # Set to True if you want to include ClassicAgent in the search, False only the default config will be run.

    # Hyperparameter-Suchräume für verschiedene Agentenklassen
    agent_classes_with_param_grids = {
        #qf.DQNAgent: qf.DEFAULT_DQN_HYPERPARAMETER_SPACE if DQN else {},    
        #qf.SACAgent: qf.DEFAULT_SAC_HYPERPARAMETER_SPACE if SAC else {}, 
        #qf.TD3Agent: qf.DEFAULT_TD3_HYPERPARAMETER_SPACE if TD3 else {},
        #qf.TangencyAgent: qf.DEFAULT_TANGENCY_HYPERPARAMETER_SPACE if Tangency else {},
        #qf.MaxExpReturnAgent: qf.DEFAULT_MAX_EXP_RETURN_HYPERPARAMETER_SPACE if MaxExpReturn else {},
        #qf.MinVarianceAgent: qf.DEFAULT_MIN_VARIANCE_HYPERPARAMETER_SPACE if MinVariance else {},
        qf.ClassicOnePeriodMarkovitzAgent: qf.DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZ_HYPERPARAMETER_SPACE if Classic else {},
    }

    # Führe die Hyperparameter-Suche durch
    results = qf.hyperparameter_search(
        env_config=train_env_config,
        agent_classes_with_param_grids=agent_classes_with_param_grids,
        eval_env_config=eval_env_config,
        max_timesteps=50000,
        episodes=1
    )

    print("Beste Agentenklasse:", results["best_agent_class"].__name__)
    print("Beste Konfiguration:", results["best_config"])
    print("Beste Belohnung:", results["best_reward"])


if __name__ == "__main__":
    qf.start_tensorboard()
    main()