from stable_baselines3 import SAC
import qf  as qf
from qf.agents.sb3_agents.sb3_agent import SB3Agent

class SACAgent(SB3Agent):
    def __init__(self, env, config=None):
        """
        Initializes the SAC agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the SAC agent.
        """
        super().__init__(env)

        # Default configuration
        default_config = {
            "policy": qf.DEFAULT_SAC_POLICY,  # Default policy architecture
            "learning_rate": qf.DEFAULT_SAC_LR,
            "buffer_size": qf.DEFAULT_SAC_BUFFER_MAX_SIZE,
            "batch_size": qf.DEFAULT_SAC_BATCH_SIZE,
            "tau": qf.DEFAULT_SAC_TAU,  # Target network update rate
            "gamma": qf.DEFAULT_SAC_GAMMA,
            "train_freq": qf.DEFAULT_SAC_TRAIN_FREQ,  # Frequency of training steps
            "gradient_steps": qf.DEFAULT_SAC_GRADIENT_STEPS,  # Number of gradient steps per training iteration
            "device": qf.DEFAULT_DEVICE,  # Device to run the computations on
            "ent_coef": qf.DEFAULT_SAC_ENT_COEF,  # Automatic entropy coefficient adjustment
            "verbose": qf.DEFAULT_SAC_VERBOSITY # Verbosity level for logging
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        # Initialize SAC model
        self.model = SAC(
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
            device=self.config["device"],
            ent_coef=self.config["ent_coef"]
        )


