import qf

def main():
    qf.start_tensorboard()

    # Training und Evaluierungsumgebungen
    train_env_config = qf.DEFAULT_TRAIN_ENV_CONFIG
    eval_env_config = qf.DEFAULT_EVAL_ENV_CONFIG

    DQN = True         # Set to True if you want to include DQNAgent in the search, False only the default config will be run.
    Classic = True      # Set to True if you want to include ClassicAgent in the search, False only the default config will be run.

    DQN_SPACE = qf.DEFAULT_DQNAGENT_HYPERPARAMETER_SPACE
    
    Classic_SPACE = {
        "target": ["MaxExpReturn"],
        "risk_free_rate": [0.0],
        "risk_model": ["sample_cov"],
        "log_returns": [True]
    }

    # Hyperparameter-Suchräume für verschiedene Agentenklassen
    agent_classes_with_param_grids = {
        qf.DQNAgent: DQN_SPACE if DQN else {},
        qf.ClassicOnePeriodMarkovitzAgent: Classic_SPACE if Classic else {}
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
    main()