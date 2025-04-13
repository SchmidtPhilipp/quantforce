import json
import os
import inspect
from datetime import datetime
from train.scheduler.epsilon_scheduler import LinearEpsilonScheduler
from train.scheduler.epsilon_scheduler import InverseSigmoidEpsilonScheduler
from train.scheduler.epsilon_scheduler import ExponentialEpsilonScheduler

class Config:
    def __init__(self, config_path):
        """
        Loads and validates the configuration from a JSON file.

        Parameters:
            config_path (str): Path to the configuration file.
        """
        self.config_path = config_path
        self.data = self._load_config()
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
            "model_config": None,
            "tau": 0.01, # Target network update rate
            "gamma": 0.99, # Discount factor
            "batch_size": 64, # Replay buffer size
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
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Combine the config name and timestamp to create the run name
        return f"{config_name}_{timestamp}"

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
            from train.scheduler.epsilon_scheduler import LinearEpsilonScheduler, ExponentialEpsilonScheduler
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
    