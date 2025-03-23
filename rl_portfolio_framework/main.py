import json
import argparse
import os

from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from envs.portfolio_env import PortfolioEnv
from data.downloader import download_data
from data.preprocessor import add_technical_indicators
from train import train_agent
from evaluate import evaluate_agent
from utils.start_tensorboard import start_tensorboard
from utils.safari import focus_tensorboard_tab
from utils.save_config import save_config


def load_config(path):
    with open(path, "r") as f:
        return json.load(f)


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
    run_name = f"{config['agent']}_{'-'.join(tickers)}_{config['episodes']}ep"

    start_tensorboard(mode="safari")

    ##################################
    # Training setup
    train_data = download_data(tickers, config["train_start"], config["train_end"])
    train_data = add_technical_indicators(train_data)  

    train_env = PortfolioEnv(train_data)

    train_agent_instance = DQNAgent(
        obs_dim=train_env.observation_space.shape[0],
        act_dim=train_env.action_space.shape[0]
    )

    load_path = train_agent(
        env=train_env,
        agent=train_agent_instance,
        n_episodes=config["episodes"],
        run_name=run_name
    )
    ##################################

    ##################################
    # Evaluation setup
    eval_data = download_data(tickers, config["eval_start"], config["eval_end"])
    eval_data = add_technical_indicators(eval_data)  # ✅ preprocess

    eval_env = PortfolioEnv(eval_data)

    eval_agent_instance = DQNAgent(
        obs_dim=eval_env.observation_space.shape[0],
        act_dim=eval_env.action_space.shape[0]
    )
    eval_agent_instance.load(load_path)

    eval_agent_instance = RandomAgent(
        act_dim=eval_env.action_space.shape[0]
    )

    evaluate_agent(env=eval_env, agent=eval_agent_instance, config=config, run_name=run_name)
    ##################################

    # END
    focus_tensorboard_tab()


if __name__ == "__main__":
    main()



# TODOS: 
# finish the Data feting and preprocessing and testing functions
# finsish the visualization functions

# Agent classes:
# renew the Agent classes such that the observalbes can have any number of features
# but the actions are limited to the number of assets + 1 (cash)

# Implement Multi-Agent training and evaluation
# Implement the PortfolioEnv such that it can handle multiple agents at the same time
# Implement the evaluation such that it can handle multiple agents at the same time

# Implement the MADDPG algorithm
# Implement the MADDPG agent with CPPI and TIPP and compare the results

# Implement the MADDPG agent with other risk management strategies and compare the results

# Implement classical agents 
