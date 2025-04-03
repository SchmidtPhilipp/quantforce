import json
import argparse
import os
import shutil

from config.config import Config
from envs.portfolio_agent_generator import create_portfolio_env
from train.train import train_agent
from eval.evaluate import evaluate_agent
from utils.start_tensorboard import start_tensorboard
from utils.safari import focus_tensorboard_tab
from data.downloader import get_data
from agents.create_agent import create_agent


def process_config(config_path, processed_folder):
    """
    Processes a single configuration file and runs training and evaluation.

    Parameters:
        config_path (str): Path to the configuration file.
        processed_folder (str): Folder to move the processed configuration file.
    """
    print(f"Processing configuration: {config_path}")
    config = Config(config_path)

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

    # Move the processed config to the processed folder
    shutil.move(config_path, os.path.join(processed_folder, os.path.basename(config_path)))
    print(f"Configuration {config_path} processed and moved to {processed_folder}.")

    # END
    if config.get("enable_tensorboard"):
        focus_tensorboard_tab()


def main():
    # CLI arg: --config_folder configs/pending --processed_folder configs/processed
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_folder", type=str, default="configs/pending", help="Folder containing pending configuration files")
    parser.add_argument("--processed_folder", type=str, default="configs/processed", help="Folder to move processed configuration files")
    args = parser.parse_args()

    config_folder = args.config_folder
    processed_folder = args.processed_folder

    # Ensure the processed folder exists
    os.makedirs(processed_folder, exist_ok=True)

    # Get all JSON files in the config folder
    config_files = [os.path.join(config_folder, f) for f in os.listdir(config_folder) if f.endswith(".json")]

    if not config_files:
        print(f"No configuration files found in {config_folder}. Exiting.")
        return

    print(f"Found {len(config_files)} configuration files in {config_folder}.")

    # Process each configuration file
    for config_path in config_files:
        try:
            process_config(config_path, processed_folder)
        except Exception as e:
            print(f"Error processing configuration {config_path}: {e}")
            continue

    print("All configurations processed.")


if __name__ == "__main__":
    main()




