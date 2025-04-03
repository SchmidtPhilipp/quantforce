import json
import argparse

class Config:
    def __init__(self):
        """
        Loads and validates the configuration from a JSON file or CLI arguments.
        """
        # CLI arg: --config configs/dqn_msft.json
        parser = argparse.ArgumentParser()
        parser.add_argument("--config", type=str, help="Path to config file")
        args = parser.parse_args()

        if args.config:
            self.config_path = args.config
        else:
            self.config_path = "configs/maddpg_msft.json"
            print(f"Keine Konfigurationsdatei angegeben. Verwende Standardkonfiguration: {self.config_path}")

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
            "shared_obs": False,
            "shared_action": False,
            "trade_cost_percent": 0.01,
            "trade_cost_fixed": 1.0,
            "enable_tensorboard": False
        }

        print("#"*50)
        print("Configuration Report:")
        for key, default_value in defaults.items():
            if key not in self.data:
                print(f'"{key}" not set; default value "{default_value}" applied.')
                self.data[key] = default_value
            else:
                print(f'  "{key}" set to "{self.data[key]}".')
        print("#"*50)

    def _validate_config(self):
        """Validates the configuration and adjusts invalid settings."""
        if not self.data["shared_obs"] and self.data["n_agents"] != len(self.data["tickers"]):
            print(
                f"WARNING: Number of agents (n_agents={self.data['n_agents']}) does not match "
                f"the number of assets (tickers={len(self.data['tickers'])}). Adjusting n_agents to match tickers."
            )
            self.data["n_agents"] = len(self.data["tickers"])

        if not self.data["shared_action"] and self.data["n_agents"] != len(self.data["tickers"]):
            print(
                f"WARNING: Number of agents (n_agents={self.data['n_agents']}) does not match "
                f"the number of assets (tickers={len(self.data['tickers'])}). Adjusting n_agents to match tickers."
            )
            self.data["n_agents"] = len(self.data["tickers"])

        print("#"*50)


    def _generate_run_name(self):
        """Generates a unique run name based on the configuration."""
        tickers = self.data.get("tickers", [])
        return f"{self.data['agent']}_{'-'.join(tickers)}_{self.data['train_episodes']}ep"

    def get(self, key, default=None):
        """Gets a configuration value with an optional default."""
        return self.data.get(key, default)

    def __getitem__(self, key):
        """Allows dictionary-like access to the configuration."""
        return self.data[key]