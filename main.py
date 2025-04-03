import json
import argparse
import os

from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from agents.maddpg_agent import MADDPGAgent
from agents.ppo_agent import PPOAgent
from envs.portfolio_agent_generator import create_portfolio_env
from trainer.train import train_agent
from eval.evaluate import evaluate_agent
from utils.start_tensorboard import start_tensorboard
from utils.safari import focus_tensorboard_tab
import warnings
from data.downloader import get_data


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


# Start with:  python main.py --config configs/dqn_msft.json
# Test with: python -m unittest discover -s tests -p "test_*.py"

def main():
    # CLI arg: --config configs/dqn_msft.json
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file")
    args = parser.parse_args()

    if args.config:
        config_path = args.config
    else:
        config_path = "configs/dqn_msft.json"
        print(f"Keine Konfigurationsdatei angegeben. Verwende Standardkonfiguration: {config_path}")

    config = load_config(config_path)

    tickers = config["tickers"]
    run_name = f"{config['agent']}_{'-'.join(tickers)}_{config['train_episodes']}ep"

    start_tensorboard(mode="safari", port=6005)

    ##################################
    # Training setup
    train_data = get_data(tickers, config["train_start"], config["train_end"], indicators=config["indicators"])

    train_env = create_portfolio_env(
        data=train_data,
        initial_balance=config.get("initial_balance", 100_000),
        verbosity=config.get("verbosity", 0),
        n_agents=config.get("n_agents", 1),
        shared_obs=config.get("shared_obs", True),
        shared_action=config.get("shared_action", True),
        trade_cost_percent=config.get("trade_cost_percent", 0.01),
        trade_cost_fixed=config.get("trade_cost_fixed", 1.0)
    )

    if config["agent"] == "DQNAgent":
        train_agent_instance = DQNAgent(
            obs_dim=train_env.observation_space.shape[0],
            act_dim=train_env.action_space.shape[0]
        )
    elif config["agent"] == "MADDPGAgent":
        train_agent_instance = MADDPGAgent(
            obs_dim=train_env.observation_space.shape[0],
            act_dim=train_env.action_space.shape[0],
            n_agents=config.get("n_agents", 1)
        )
    elif config["agent"] == "RandomAgent":
        train_agent_instance = RandomAgent(
            obs_dim=train_env.observation_space.shape[0],
            act_dim=train_env.action_space.shape[0]
        )
    elif config["agent"] == "PPOAgent":
        train_agent_instance = PPOAgent(
            obs_dim=train_env.observation_space.shape[0],
            act_dim=train_env.action_space.shape[0],
            model_config=config.get("model_config", None),
            lr=config.get("lr", 3e-4),
            gamma=config.get("gamma", 0.99),
            eps_clip=config.get("eps_clip", 0.2),
            k_epochs=config.get("k_epochs", 4)
        )
    else:
        raise ValueError(f"Unknown agent type: {config['agent']}")

    load_path = train_agent(
        env=train_env,
        agent=train_agent_instance,
        n_episodes=config["train_episodes"],
        run_name=run_name
    )
    ##################################


    ##################################
    # Evaluation setup
    eval_data = get_data(tickers, config["eval_start"], config["eval_end"], indicators=config["indicators"])

    eval_env = create_portfolio_env(
        data=eval_data,
        initial_balance=config.get("initial_balance", 100_000),
        verbosity=config.get("verbosity", 0),
        n_agents=config.get("n_agents", 1),
        shared_obs=config.get("shared_obs", True),
        shared_action=config.get("shared_action", True)
    )

    if config["agent"] == "DQNAgent":
        eval_agent_instance = DQNAgent(
            obs_dim=eval_env.observation_space.shape[0],
            act_dim=eval_env.action_space.shape[0]
        )
    elif config["agent"] == "MADDPGAgent":
        eval_agent_instance = MADDPGAgent(
            obs_dim=eval_env.observation_space.shape[0],
            act_dim=eval_env.action_space.shape[0],
            n_agents=config.get("n_agents", 1)
        )
    elif config["agent"] == "RandomAgent":
        eval_agent_instance = RandomAgent(
            obs_dim=eval_env.observation_space.shape[0],
            act_dim=eval_env.action_space.shape[0]
        )
    elif config["agent"] == "PPOAgent":
        eval_agent_instance = PPOAgent(
            obs_dim=eval_env.observation_space.shape[0],
            act_dim=eval_env.action_space.shape[0],
            model_config=config.get("model_config", None),
            lr=config.get("lr", 3e-4),
            gamma=config.get("gamma", 0.99),
            eps_clip=config.get("eps_clip", 0.2),
            k_epochs=config.get("k_epochs", 4)
        )
    else:
        raise ValueError(f"Unknown agent type: {config["agent"]}")

    eval_agent_instance.load(load_path)

    evaluate_agent(env=eval_env, 
                   agent=eval_agent_instance, 
                   config=config, 
                   run_name=run_name,
                   n_runs=config["eval_episodes"])
    
    ##################################

    # END
    focus_tensorboard_tab()


if __name__ == "__main__":
    main()




