import json
import argparse
import os

from agents.dqn_agent import DQNAgent
from agents.random_agent import RandomAgent
from envs.portfolio_env import PortfolioEnv
from data.data_loader import download_data
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
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    config = load_config(args.config)

    tickers = config["tickers"]
    run_name = f"{config['agent']}_{'-'.join(tickers)}_{config['episodes']}ep"

    start_tensorboard(mode="safari")

    ##################################
    # Training setup
    train_data = download_data(tickers, config["train_start"], config["train_end"])
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
    eval_env = PortfolioEnv(eval_data)

    eval_agent_instance = DQNAgent(
        obs_dim=eval_env.observation_space.shape[0],
        act_dim=eval_env.action_space.shape[0]
    )
    eval_agent_instance.load(load_path)

    evaluate_agent(env=eval_env, agent=eval_agent_instance, config=config, run_name=run_name)
    ##################################


    ##################################
    # END
    focus_tensorboard_tab()


if __name__ == "__main__":
    main()
