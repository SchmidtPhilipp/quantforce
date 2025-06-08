from stable_baselines3 import DDPG
from qf.agents.sb3_agents.sb3_agent import SB3Agent
import qf

class DDPGAgent(SB3Agent):
    def __init__(self, env, config=None):
        """
        Initializes the DDPG agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the DDPG agent.
        """
        super().__init__(env)

        # Default configuration
        default_config = {
            "policy": qf.DEFAULT_DDPG_POLICY,
            "learning_rate": qf.DEFAULT_DDPG_LR,
            "buffer_size": qf.DEFAULT_DDPG_BUFFER_MAX_SIZE,
            "batch_size": qf.DEFAULT_DDPG_BATCH_SIZE,
            "tau": qf.DEFAULT_DDPG_TAU,
            "gamma": qf.DEFAULT_DDPG_GAMMA,
            "train_freq": qf.DEFAULT_DDPG_TRAIN_FREQ,
            "gradient_steps": qf.DEFAULT_DDPG_GRADIENT_STEPS,
            "device": qf.DEFAULT_DEVICE,
            "verbose": qf.DEFAULT_DDPG_VERBOSITY
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        # Initialize DDPG model
        self.model = DDPG(
            policy=self.config["policy"],
            env=self.env,
            learning_rate=self.config["learning_rate"],
            buffer_size=self.config["buffer_size"],
            batch_size=self.config["batch_size"],
            tau=self.config["tau"],
            gamma=self.config["gamma"],
            train_freq=self.config["train_freq"],
            gradient_steps=self.config["gradient_steps"],
            verbose=self.config["verbose"],
            device=self.config["device"]
        )

        from .td3_agent import train_TD3_with_TD_error_logging
        self.model.train = lambda gradient_steps, batch_size=64: train_TD3_with_TD_error_logging(self.model, gradient_steps, batch_size)

    @staticmethod
    def get_default_config():
        return qf.DEFAULT_DDPGAGENT_CONFIG
    
    @staticmethod
    def get_hyperparameter_space():
        return qf.DEFAULT_DDPGAGENT_HYPERPARAMETER_SPACE