import json
import os
import inspect
from datetime import datetime
from train.scheduler.epsilon_scheduler import LinearEpsilonScheduler
from train.scheduler.epsilon_scheduler import InverseSigmoidEpsilonScheduler
from train.scheduler.epsilon_scheduler import ExponentialEpsilonScheduler
from train.scheduler.epsilon_scheduler import PeriodicEpsilonScheduler

from agents.dqn_agent import DQNAgent  # Add other agents as needed
from agents.maddpg_agent import MADDPGAgent  # Add other agents as needed
from agents.model_builder import ModelBuilder
from agents.base_agent import BaseAgent
from agents.up_agent import UniversalPortfolioAgent

class Config:
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
        self._validate_config()
        self.run_name = self._generate_run_name()

    def _load_config(self):
        """Loads the configuration from a JSON file."""
        with open(self.config_path, "r") as f:
            return json.load(f)

    def _set_defaults(self):
        """Sets default values for missing configuration keys and prints a report."""
        defaults = {
            "initial_balance": 100_000,
            "verbosity": 0,
            "n_agents": 1,
            "trade_cost_percent": 0.00,
            "trade_cost_fixed": 0.0,
            "enable_tensorboard": True,
            "tau": 0.01, # Target network update rate
            "gamma": 0.99, # Discount factor
            "batch_size": 64, # Replay buffer size
            "time_window_size": 365, # Time window size for the dataset
        }

        print("-" * 50)
        print("✅ Config:")
        for key, default_value in defaults.items():
            if key not in self.data:
                print(f'{key} not set; default value "{default_value}" applied.')
                self.data[key] = default_value
            else:
                print(f'{key} set to "{self.data[key]}".')
        print("-" * 50)

    def _validate_config(self):
        """Validates the configuration and adjusts invalid settings."""
        return
        
    def _generate_run_name(self):
        """
        Generates a unique run name based on the configuration file name and a timestamp.
        """
        # Extract the base name of the configuration file (without extension)
        config_name = os.path.splitext(os.path.basename(self.config_path))[0]

        # Generate a timestamp in the format YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # Generate a random name using the `names` library
        import names
        name = names.get_first_name()

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
            object: An instance of the specified scheduler class.
        """
        exploration_config = self.data.get("EXPLORATION_CONFIGS", {})
        scheduler_class_name = exploration_config.get("scheduler_class")
        scheduler_params = exploration_config.get("params", {})

        if not scheduler_class_name:
            raise ValueError("Scheduler class name is not specified in the config.")

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
                    if "out_features" in layer["params"] and layer["params"]["out_features"] == "obs_dim":
                        layer["params"]["out_features"] = 1

        # Instantiate the agent with the provided parameters
        return agent_class(obs_dim=obs_dim, act_dim=act_dim, **agent_params)
