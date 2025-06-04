from qf.utils.hyperparameter_optimizer import HyperparameterOptimizer
import qf
from optuna.visualization import plot_optimization_history, plot_param_importances


def main():
    # Trainings- und Evaluierungsumgebungskonfigurationen
    train_env_config = qf.DEFAULT_TRAIN_ENV_CONFIG
    eval_env_config = qf.DEFAULT_EVAL_ENV_CONFIG

    # Liste der Agent-Klassen zur Optimierung
    agent_classes = [qf.DQNAgent, qf.SACAgent, qf.TD3Agent]

    # Optimierungskonfiguration
    optim_config = {
        "objective": "avg_reward - std_deviation",  # Zielmetrik: Durchschnittliche Belohnung minus Standardabweichung
        "max_timesteps": 50000,  # Maximale Anzahl an Trainings-Timesteps
        "episodes": 10  # Anzahl der Evaluierungs-Episoden
    }

    # Hyperparameter-Optimierung durchführen
    optimizer = HyperparameterOptimizer(agent_classes, train_env_config, eval_env_config, optim_config)
    results, study = optimizer.optimize(n_trials=3)  # Anzahl der Optimierungsversuche

    # Ergebnisse ausgeben
    print("Beste Agentenklasse:", results["best_agent_class"].__name__)
    print("Beste Konfiguration:", results["best_config"])
    print("Beste Belohnung:", results["best_reward"])

    # Visualisierung der Ergebnisse
    print("\nVisualisierung der Optimierungsergebnisse:")
    plot_optimization_history(study).show()  # Zeigt die Optimierungshistorie
    plot_param_importances(study).show()  # Zeigt die Wichtigkeit der Hyperparameter


if __name__ == "__main__":
    main()