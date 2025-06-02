import qf
from qf.agents.marllib_agents.marllib_agent import MARLLibAgent
from marllib import marl

class MAPPOAgent(MARLLibAgent):
    def __init__(self, env, config=None):
        """
        Initializes the MAPPOAgent with the given environment.
        Parameters:

            env: The environment in which the agents will operate.
        """
        super().__init__(env)

        # Default configuration
        default_config = {
            "learning_rate": qf.DEFAULT_MAPPO_LR,
            "gamma": qf.DEFAULT_MAPPO_GAMMA,
            "batch_size": qf.DEFAULT_MAPPO_BATCH_SIZE,
            "buffer_max_size": qf.DEFAULT_MAPPO_BUFFER_MAX_SIZE,
            "device": qf.DEFAULT_DEVICE,
            "epsilon_start": qf.DEFAULT_MAPPO_EPSILON_START
        }

        # Merge default config with provided config
        self.config = {**default_config, **(config or {})}

        # Initialize the model with the specified architecture
        self.model = self._build_model(self.config["model_architecture"])
        self.target_model = self._build_model(self.config["model_architecture"])

        mappo = marl.algos.mappo()


        