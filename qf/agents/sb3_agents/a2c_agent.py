from stable_baselines3 import A2C
from qf.agents.sb3_agents.sb3_agent import SB3Agent
import qf

class A2CAgent(SB3Agent):
    def __init__(self, env, config=None):
        """
        Initializes the A2C agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the A2C agent.
        """
        super().__init__(env)

        # Default configuration
        default_config = {
            "policy": qf.DEFAULT_A2C_POLICY,
            "learning_rate": qf.DEFAULT_A2C_LR,
            "n_steps": qf.DEFAULT_A2C_N_STEPS,
            "gamma": qf.DEFAULT_A2C_GAMMA,
            "gae_lambda": qf.DEFAULT_A2C_GAE_LAMBDA,
            "ent_coef": qf.DEFAULT_A2C_ENT_COEF,
            "vf_coef": qf.DEFAULT_A2C_VF_COEF,
            "max_grad_norm": qf.DEFAULT_A2C_MAX_GRAD_NORM,
            "device": qf.DEFAULT_DEVICE,
            "verbose": qf.DEFAULT_A2C_VERBOSITY
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        # Initialize A2C model
        self.model = A2C(
            policy=self.config["policy"],
            env=env,
            learning_rate=self.config["learning_rate"],
            n_steps=self.config["n_steps"],
            gamma=self.config["gamma"],
            gae_lambda=self.config["gae_lambda"],
            ent_coef=self.config["ent_coef"],
            vf_coef=self.config["vf_coef"],
            max_grad_norm=self.config["max_grad_norm"],
            verbose=self.config["verbose"],
            device=self.config["device"]
        )