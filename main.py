import json
import argparse

from config.config import Config
from envs.portfolio_agent_generator import create_portfolio_env
from train.train import train_agent
from eval.evaluate import evaluate_agent
from utils.start_tensorboard import start_tensorboard
from utils.safari import focus_tensorboard_tab
from data.downloader import get_data
from agents.create_agent import create_agent

def main():
    # Load and validate the configuration
    config = Config()

    if config.get("enable_tensorboard"):
        start_tensorboard(mode="safari", port=6005)

    ##################################
    # Training setup
    train_data = get_data(config["tickers"], config["train_start"], config["train_end"], indicators=config["indicators"])

    train_env = create_portfolio_env(
        data=train_data,
        initial_balance=config["initial_balance"],
        verbosity=config["verbosity"],
        n_agents=config["n_agents"],
        shared_obs=config["shared_obs"],
        shared_action=config["shared_action"],
        trade_cost_percent=config["trade_cost_percent"],
        trade_cost_fixed=config["trade_cost_fixed"]
    )

    train_agent_instance = create_agent(config["agent"], train_env, config.data)

    load_path = train_agent(
        env=train_env,
        agent=train_agent_instance,
        n_episodes=config["train_episodes"],
        run_name=config.run_name
    )
    ##################################

    ##################################
    # Evaluation setup
    eval_data = get_data(config["tickers"], config["eval_start"], config["eval_end"], indicators=config["indicators"])

    eval_env = create_portfolio_env(
        data=eval_data,
        initial_balance=config["initial_balance"],
        verbosity=config["verbosity"],
        n_agents=config["n_agents"],
        shared_obs=config["shared_obs"],
        shared_action=config["shared_action"]
    )

    eval_agent_instance = create_agent(config["agent"], eval_env, config.data)

    eval_agent_instance.load(load_path)

    evaluate_agent(env=eval_env, 
                   agent=eval_agent_instance, 
                   config=config.data, 
                   run_name=config.run_name,
                   n_runs=config["eval_episodes"])
    ##################################

    # END
    if config.get("enable_tensorboard"):
        focus_tensorboard_tab()


if __name__ == "__main__":
    main()




