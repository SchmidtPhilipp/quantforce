from qf.agents.agent import Agent
from qf.envs.sb3_wrapper import SB3Wrapper

class SB3Agent(Agent):
    # Class that inherits from Agent and gives a base implementation 
    # for all stable-baselines3 agents.
    def __init__(self, env):
        """
        Initializes the SB3 agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
            config (dict): Configuration dictionary for the SB3 agent.
        """ 
        if type(env) is not SB3Wrapper:
            env = SB3Wrapper(env)

        super().__init__(env)

    def train(self, total_timesteps=100000, use_tqdm=True):
        """
        Trains the SB3 agent for a specified number of timesteps and tracks the TD error.
        Parameters:
            total_timesteps (int): Total number of timesteps to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print summaries.
        """
        self.model.learn(total_timesteps=total_timesteps, progress_bar=True if use_tqdm else False, reset_num_timesteps=False)

    def act(self, state, deterministic=True):
        """
        Returns the action to take in the environment based on the current state.
        Parameters:
            state: The current state of the environment.
            epsilon (float): Epsilon value for exploration (not used in SB3 agents).
        Returns:
            action: The action to take in the environment.
        """
        action, _ = self.model.predict(state, deterministic=deterministic)
        return action
    
    def evaluate(self, eval_env=None, episodes=1, use_tqdm=True):
        
        if type(eval_env) is not SB3Wrapper:
            eval_env = SB3Wrapper(eval_env)

        return super().evaluate(eval_env=eval_env, episodes=episodes, use_tqdm=use_tqdm)
    
    def save(self, path):
        """
        Saves the SAC agent's model to a file.
        Parameters:
            path (str): Path to save the model.
        """

        self.model.save(path + "/sac_agent_model")

    @staticmethod
    def load(path, env, agent_class, device="cpu"):
        """
        Loads an SB3 agent's model from a file.
        Parameters:
            path (str): Path to load the model from.
            env: The environment to associate with the loaded model.
            agent_class (type): The class of the agent to instantiate (e.g., SACAgent, TD3Agent).
            device (str): Device to load the model on ("cpu" or "cuda").
        Returns:
            SB3Agent: A new instance of the specified agent class with the loaded model.
        """
        model = agent_class.model.load(path, env=env, device=device)
        agent = agent_class(env)
        agent.model = model
        return agent



