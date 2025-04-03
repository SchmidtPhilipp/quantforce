from .dqn_agent import DQNAgent
from .maddpg_agent import MADDPGAgent
from .random_agent import RandomAgent
from .ppo_agent import PPOAgent
from .base_agent import BaseAgent


def create_agent(agent_type, env, config):




    agent_mapping = {
        "DQNAgent": (DQNAgent, {
            "model_config": config.get("model_config", None),
            "lr": config.get("lr", 1e-3),
            "gamma": config.get("gamma", 0.99),
            "batch_size": config.get("batch_size", 32)
        }),
        "MADDPGAgent": (MADDPGAgent, {"n_agents": config.get("n_agents", 2)}),
        "RandomAgent": (RandomAgent, {}),
        "PPOAgent": (PPOAgent, {
            "model_config": config.get("model_config", None),
            "lr": config.get("lr", 3e-4),
            "gamma": config.get("gamma", 0.99),
            "eps_clip": config.get("eps_clip", 0.2),
            "k_epochs": config.get("k_epochs", 4)
        })
    }

    if agent_type not in agent_mapping:
        raise ValueError(f"Unknown agent type: {agent_type}")

    agent_class, agent_params = agent_mapping[agent_type]
    return agent_class(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        **agent_params
    )