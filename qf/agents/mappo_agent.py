import qf
from qf.agents.marllib_agent import MARLLibAgent

class MAPPOAgent(MARLLibAgent):
    def __init__(self, env):
        """
        Initializes the MAPPOAgent with the given environment.
        Parameters:

            env: The environment in which the agents will operate.
        """
        super().__init__(env, model=qf.DEFAULT_MAPPO_MODEL)
        


        