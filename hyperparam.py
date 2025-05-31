import qf

def main():
    # Training und Evaluierungsumgebungen
    train_env_config = qf.DEFAULT_TRAIN_ENV_CONFIG
    eval_env_config = qf.DEFAULT_EVAL_ENV_CONFIG

    # Hyperparameter-Suchräume für verschiedene Agentenklassen
    agent_classes_with_param_grids = {
        qf.SACAgent: {
            #"learning_rate": [1e-5, 1e-4, 1e-3],
            #"buffer_size": [100000, 500000],
            #"batch_size": [64, 128],
            #"gamma": [0.95, 0.99],
            #"tau": [0.005, 0.01],
        },
        qf.TD3Agent: {
            #"learning_rate": [1e-4, 1e-3],
            #"buffer_size": [500000, 1000000],
            #"batch_size": [128, 256],
            #"gamma": [0.9, 0.99],
            #"noise_std": [0.1, 0.2],
            #"noise_clip": [0.3, 0.5],
        },
        qf.TangencyAgent: {
        },
        qf.DQNAgent: {
            #"learning_rate": [1e-3, 1e-4],
            #"gamma": [0.95, 0.99],
            #"batch_size": [32, 64],
            #"buffer_max_size": [50000, 100000],
            #"epsilon_start": [0.1, 0.4],
        },
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