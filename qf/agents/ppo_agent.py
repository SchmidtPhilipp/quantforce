from stable_baselines3 import PPO
from qf.agents.sb3_agent import SB3Agent
import qf

class PPOAgent(SB3Agent):
    def __init__(self, env, config=None):
        """
        Initializes the PPO agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the PPO agent.
        """
        super().__init__(env)

        # Default configuration
        default_config = {
            "policy": qf.DEFAULT_PPO_POLICY,
            "learning_rate": qf.DEFAULT_PPO_LR,
            "n_steps": qf.DEFAULT_PPO_N_STEPS,
            "batch_size": qf.DEFAULT_PPO_BATCH_SIZE,
            "gamma": qf.DEFAULT_PPO_GAMMA,
            "gae_lambda": qf.DEFAULT_PPO_GAE_LAMBDA,
            "clip_range": qf.DEFAULT_PPO_CLIP_RANGE,
            "device": qf.DEFAULT_DEVICE,
            "verbose": qf.DEFAULT_PPO_VERBOSITY
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        # Initialize PPO model
        self.model = PPO(
            policy=self.config["policy"],
            env=env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            batch_size=self.config["batch_size"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            clip_range=self.config["clip_range"],
            verbose=self.config["verbose"],
            device=self.config["device"]
        )