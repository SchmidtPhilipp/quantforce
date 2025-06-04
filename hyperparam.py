import qf as qf
qf.setDebugMode(True)

def main():
    # Trainings- und Evaluierungsumgebungskonfigurationen
    train_env_config = qf.DEFAULT_TRAIN_ENV_CONFIG
    eval_env_config = qf.DEFAULT_EVAL_ENV_CONFIG

    # Liste der Agent-Klassen zur Optimierung
    agent_classes = [qf.ClassicOnePeriodMarkovitzAgent] #[qf.DQNAgent, qf.SACAgent, qf.TD3Agent, qf.ClassicOnePeriodMarkovitzAgent]
    agent_classes = [qf.SACAgent]
    env_class = qf.MultiAgentPortfolioEnv

    # Optimierungskonfiguration
    optim_config = {
        "objective": "avg_reward - std_deviation",  # Zielmetrik: Durchschnittliche Belohnung minus Standardabweichung
        "max_timesteps": 10,  # Maximale Anzahl an Trainings-Timesteps
        "episodes": 1  # Anzahl der Evaluierungs-Episoden
    }

    # Hyperparameter-Optimierung durchführen
    optimizer = qf.HyperparameterOptimizer(agent_classes, env_class, train_env_config, eval_env_config, optim_config)

    # Optuna-Optimierung starten
    results = optimizer.optimize(n_trials=5)  

    # Ergebnisse ausgeben
    print("Beste Agentenklasse:", results["best_agent_class"].__name__)
    print("Beste Konfiguration:", results["best_config"])
    print("Beste Belohnung:", results["best_reward"])

    # Visualisierung der Ergebnisse
    print("\nVisualisierung der Optimierungsergebnisse:")
    optimizer.visualize_results()


if __name__ == "__main__":
    main()