

class RayLib_Env_Wrapper(MulitiAgentEnv):
    def __init__(self, env: TensorEnv):
        super().__init__()
        self.env = env
        self.observation_space = env.get_observation_space()
        self.action_space = env.get_action_space()




