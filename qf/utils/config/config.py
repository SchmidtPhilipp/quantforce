import json
import os
import inspect

import random
import time

from datetime import datetime

from qf.train.scheduler.epsilon_scheduler import LinearEpsilonScheduler, InverseSigmoidEpsilonScheduler, ExponentialEpsilonScheduler, PeriodicEpsilonScheduler
from qf import ModelBuilder, DQNAgent, MADDPGAgent
from qf.utils.helper_functions import generate_random_name


class Config:
    VALID_KEYS = {
        "initial_balance",
        "verbosity",
        "n_agents",
        "trade_cost_percent",
        "trade_cost_fixed",
        "enable_tensorboard",
        "time_window_size",
        "tickers",
        "device",
        "AGENT_CONFIGS",
        "EXPLORATION_CONFIGS",
    }

    def __init__(self, config_path):
        """
        Loads and validates the configuration from a JSON file.

        Parameters:
            config_path (str): Path to the configuration file.
        """
        self.config_path = config_path
        self.data = self._load_config()

        agent_config = self.data.get("AGENT_CONFIGS", {})
        agent_params = agent_config.get("params", {})
        self.data["n_agents"] = agent_params.get("n_agents", 1)

        self._set_defaults()
        
        #self._validate_config()
        self.run_name = self._generate_run_name()

    def _load_config(self):
        """Loads the configuration from a JSON file."""
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _set_defaults(self):
        """Sets default values for missing configuration keys and prints a report."""
        defaults = {
            "initial_balance": 1_000_000,
            "verbosity": 0,
            "n_agents": 1,
            "trade_cost_percent": 0.00,
            "trade_cost_fixed": 0.0,
            "enable_tensorboard": True,
            "time_window_size": 1,  # Time window size for the dataset
            "tickers": ["AAPL", "GOOGL", "MSFT"],  # Default tickers
            "device": "cpu",  # Default device
            "reward_function": None,
        }

        print("-" * 50)
        print("✅ Config:")
        for key, default_value in defaults.items():
            if key not in self.data:
                print(f'{key} not set; default value "{default_value}" applied.')
                self.data[key] = default_value
            else:
                print(f'{key} set to "{self.data[key]}".')


        if self.data["tickers"] == "NASDAQ100":
            from qf.data.tickers.tickers import NASDAQ100
            self.data["tickers"] = NASDAQ100
            print(f"Tickers set to NASDAQ100: {self.data['tickers']}")
        elif self.data["tickers"] == "SNP500":
            from qf.data.tickers.tickers import SNP_500
            self.data["tickers"] = SNP_500
            print(f"Tickers set to S&P 500: {self.data['tickers']}")
        elif self.data["tickers"] == "DOWJONES":
            from qf.data.tickers.tickers import DOWJONES
            self.data["tickers"] = DOWJONES
            print(f"Tickers set to Dow Jones: {self.data['tickers']}")


        # Ensure tickers are unique
        if "tickers" in self.data:
            self.data["tickers"] = list(set(self.data["tickers"]))
            print(f"Tickers after removing duplicates: {self.data['tickers']}")

        print("-" * 50)

    def _validate_config(self):
        """Validate that all keys in the config are valid."""
        invalid_keys = set(self.data.keys()) - self.VALID_KEYS
        if invalid_keys:
            raise ValueError(f"Invalid configuration keys: {invalid_keys}")
        
    def _generate_run_name(self):
        """
        Generates a unique run name based on the configuration file name and a timestamp.
        """
        # Extract the base name of the configuration file (without extension)
        config_name = os.path.splitext(os.path.basename(self.config_path))[0]

        # Generate a timestamp in the format YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Generate a random name using the `names` library

        name = generate_random_name()

        # Combine the config name and timestamp to create the run name
        return f"{config_name}_{timestamp}_{name}"

    def get(self, key, default=None):
        """Gets a configuration value with an optional default."""
        return self.data.get(key, default)

    def __getitem__(self, key):
        """Allows dictionary-like access to the configuration."""
        return self.data[key]

    def load_scheduler(self):
        """
        Dynamically loads and initializes the epsilon scheduler from the config.

        Returns:
            object or None: An instance of the specified scheduler class, or None if no scheduler is specified.
        """
        exploration_config = self.data.get("EXPLORATION_CONFIGS", {})
        scheduler_class_name = exploration_config.get("scheduler_class")
        scheduler_params = exploration_config.get("params", {})

        # Wenn kein Scheduler angegeben ist, einfach None zurückgeben
        if not scheduler_class_name:
            print("ℹ️ No scheduler specified in the config. Skipping scheduler initialization.")
            return None

        # Dynamically import the scheduler class (assuming it's in the `train.scheduler.epsilon_scheduler` module)
        try:
            scheduler_class = globals()[scheduler_class_name]
        except KeyError:
            raise ImportError(f"Scheduler class '{scheduler_class_name}' not found.")

        # Validate that all required parameters are provided
        required_params = inspect.signature(scheduler_class).parameters
        missing_params = [
            param for param in required_params
            if param not in scheduler_params and required_params[param].default == inspect.Parameter.empty
        ]
        if missing_params:
            raise ValueError(f"Missing required parameters for '{scheduler_class_name}': {missing_params}")

        # Instantiate the scheduler with the provided parameters
        return scheduler_class(**scheduler_params)
    
    def save(self, save_path=None):
        """
        Saves the configuration to a JSON file.

        Parameters:
            save_path (str): Path to save the configuration file.
        """
        if save_path is None:
            save_path = os.path.join(self.run_name, "config.json")
        with open(save_path, "w") as f:
            json.dump(self.data, f, indent=4)

    def load_agent(self, obs_dim, act_dim):
        """
        Dynamically loads and initializes the agent from the config.

        Parameters:
            obs_dim (int): Observation space dimension.
            act_dim (int): Action space dimension.

        Returns:
            object: An instance of the specified agent class.
        """
        agent_config = self.data.get("AGENT_CONFIGS", {})
        agent_class_name = agent_config.get("agent_class")
        agent_params = agent_config.get("params", {})

        if not agent_class_name:
            raise ValueError("Agent class name is not specified in the config.")

        # Dynamically import the agent class (assuming it's in the `agents` module)
        try:
            agent_class = globals()[agent_class_name]
        except KeyError:
            raise ImportError(f"Agent class '{agent_class_name}' not found.")
        
        n_agents = self.data["n_agents"]

        # Replace placeholders in the network architecture with actual dimensions
        if "actor_config" in agent_params:
            for layer in agent_params["actor_config"]:
                if "params" in layer:
                    if "in_features" in layer["params"] and layer["params"]["in_features"] == "obs_dim":
                        layer["params"]["in_features"] = obs_dim
                    if "out_features" in layer["params"] and layer["params"]["out_features"] == "act_dim":
                        layer["params"]["out_features"] = act_dim

        if "critic_config" in agent_params:
            for layer in agent_params["critic_config"]:
                if "params" in layer:
                    if "in_features" in layer["params"] and layer["params"]["in_features"] == "obs_dim * n_agents + act_dim * n_agents":
                        layer["params"]["in_features"] = obs_dim * n_agents + act_dim * n_agents
                    elif "in_features" in layer["params"] and layer["params"]["in_features"] == "obs_dim + act_dim":
                        layer["params"]["in_features"] = obs_dim + act_dim
                    if "out_features" in layer["params"] and layer["params"]["out_features"] == "obs_dim":
                        layer["params"]["out_features"] = 1
                        
                        

        # Instantiate the agent with the provided parameters
        return agent_class(obs_dim=obs_dim, act_dim=act_dim, **agent_params)



