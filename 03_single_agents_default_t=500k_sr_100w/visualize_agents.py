# Import the root folder of this folder
import sys
import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import matplotlib.pyplot as plt
import seaborn as sns
import qf

def visualize_agent_weights(folders, names, env, output_folder="weights_visualization"):
    """
    Lädt die Agenten und visualisiert ihre Gewichtsmatrizen.

    :param folders: Liste von Ordnern, die die gespeicherten Agenten enthalten.
    :param names: Liste von Namen der Agenten.
    :param output_folder: Ordner, in dem die Visualisierungen gespeichert werden.
    """
    os.makedirs(output_folder, exist_ok=True)

    for folder, name in zip(folders, names):
        print(f"🔄 Loading agent: {name} from {folder}")
        
        # Lade den Agenten
        agent = qf.SPQLAgent(env=env)
        agent = agent.load(folder)

        # Extrahiere die Gewichtsmatrizen (z. B. aus der Policy oder Q-Funktion)
        if hasattr(agent, "policy"):
            weights = agent.policy.state_dict()
        elif hasattr(agent, "q_network"):
            weights = agent.q_network.state_dict()
        else:
            print(f"⚠️ No weights found for agent: {name}")
            continue

        # Visualisiere die Gewichtsmatrizen
        for layer_name, weight_matrix in weights.items():
            if "weight" in layer_name:  # Nur Gewichtsmatrizen visualisieren
                plt.figure(figsize=(10, 8))
                sns.heatmap(weight_matrix.cpu().numpy(), cmap="viridis", annot=False)
                plt.title(f"{name} - {layer_name}")
                plt.xlabel("Inputs")
                plt.ylabel("Outputs")
                plt.savefig(os.path.join(output_folder, f"{name}_{layer_name}.png"))
                plt.close()

        print(f"✅ Visualization saved for agent: {name}")

if __name__ == "__main__":
    folders = ["./runs/2025-06-10_03-50-48_SPQLAgent_default_config_t=500k_sr_100w_EVAL_Kyle",
               #"./runs/2025-06-10_02-46-42_PPOAgent_default_config_t=500k_sr_100w_EVAL_Yasmine",
               #"./runs/2025-06-10_02-20-35_TD3Agent_default_config_t=500k_sr_100w_EVAL_Mona",
               #"./runs/2025-06-09_23-32-50_DDPGAgent_default_config_t=500k_sr_100w_EVAL_Bianca",
               #"./runs/2025-06-10_01-07-04_SACAgent_default_config_t=500k_sr_100w_EVAL_Adrian",
               #"./runs/2025-06-10_03-50-59_ClassicOnePeriodMarkovitzAgent_default_config_t=500k_sr_100w_EVAL_Steve"
               ]
               
    names = ["SPQL"]#, "PPO", "TD3", "DDPG", "SAC", "Tangency"]
    env = qf.MultiAgentPortfolioEnv(tensorboard_prefix="Train")
    visualize_agent_weights(folders, names, env)