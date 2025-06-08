import qf as qf

def main():
    # Trainings- und Evaluierungsumgebungskonfigurationen    
    train_env_config = qf.DEFAULT_TRAIN_ENV_CONFIG
    eval_env_config = qf.DEFAULT_EVAL_ENV_CONFIG

    # Hier können Sie die Konfigurationen anpassen, z.B.:
    env_ajustments = {
        "reward_function": "sharpe_ratio_w10",  # Belohnungsfunktion auf Sharpe Ratio setzen
        "trade_costs": 1,  # Handelskosten auf 0.1% setzen
        "trade_costs_percent": 0.01,  # Handelskosten in Prozent
    }

    # Aktualisieren der Trainings- und Evaluierungsumgebungskonfigurationen
    train_env_config = {**train_env_config, **env_ajustments}
    eval_env_config = {**eval_env_config, **env_ajustments}

    # Agenten-Configuration 
    dqn_agent_config = qf.DEFAULT_SPQLAGENT_CONFIG

    # Liste der Agent-Klassen zur Optimierung
    agent_classes = [qf.SPQLAgent]
    #agent_classes = [qf.SACAgent]
    env_class = qf.MultiAgentPortfolioEnv

    # Optimierungskonfiguration
    optim_config = {
        "objective": "avg_reward",  # Zielmetrik: Durchschnittliche Belohnung minus Standardabweichung
        "max_timesteps": 1_000_000,  # Maximale Anzahl an Trainings-Timesteps
        "episodes": 1  # Anzahl der Evaluierungs-Episoden
    }

    # Hyperparameter-Optimierung durchführen
    optimizer = qf.HyperparameterOptimizer(agent_classes, 
                                           env_class=env_class, 
                                           train_env_config=train_env_config, 
                                           eval_env_config=eval_env_config,
                                           optim_config=optim_config)

    # Optuna-Optimierung starten
    results = optimizer.optimize(n_trials=10)  

    # Ergebnisse ausgeben
    print("Beste Agentenklasse:", results["best_agent_class"].__name__)
    print("Beste Konfiguration:", results["best_config"])
    print("Beste Belohnung:", results["best_reward"])

    # Visualisierung der Ergebnisse
    optimizer.visualize_results()


    # Zum Vergleich noch die Performance vom Classic One Period Markoviz Agent
    classic_agent = qf.ClassicOnePeriodMarkovitzAgent(env_class, qf.DEFAULT_CLASSIC_ONE_PERIOD_MARKOVITZAGENT_CONFIG)
    classic_agent.train() # Calculates the correlaction matrix based on the risk model defined in the config
    classic_agent.evaluate(eval_env_config, episodes=1)
    classic_agent.visualize()


if __name__ == "__main__":
    main()