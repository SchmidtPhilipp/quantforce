import torch
from stable_baselines3 import TD3
from qf.agents.sb3_agent import SB3Agent
import qf as qf


class TD3Agent(SB3Agent):
    def __init__(self, env, config=None):
        """
        Initializes the TD3 agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the TD3 agent.
        """
        super().__init__(env)

        # Default configuration
        default_config = {
            "policy": qf.DEFAULT_TD3_POLICY,  # Default policy architecture
            "learning_rate": qf.DEFAULT_TD3_LR,
            "buffer_size": qf.DEFAULT_TD3_BUFFER_MAX_SIZE,
            "batch_size": qf.DEFAULT_TD3_BATCH_SIZE,
            "tau": qf.DEFAULT_TD3_TAU,  # Target network update rate
            "gamma": qf.DEFAULT_TD3_GAMMA,
            "train_freq": qf.DEFAULT_TD3_TRAIN_FREQ,  # Frequency of training steps
            "gradient_steps": qf.DEFAULT_TD3_GRADIENT_STEPS,  # Number of gradient steps per training iteration
            "device": qf.DEFAULT_DEVICE  # Device to run the computations on
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        # Initialize TD3 model
        self.model = TD3(
            policy=self.config["policy"],
            env=env,
            learning_rate=self.config["learning_rate"],
            buffer_size=self.config["buffer_size"],
            batch_size=self.config["batch_size"],
            tau=self.config["tau"],
            gamma=self.config["gamma"],
            train_freq=self.config["train_freq"],
            gradient_steps=self.config["gradient_steps"],
            verbose=1,
            device=self.config["device"]
        )