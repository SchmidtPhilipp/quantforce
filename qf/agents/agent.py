class Agent:
    def __init__(self, env):
        """
        Initializes the agent with the given environment.
        Parameters:

            env: The environment in which the agent will operate.
        """
        self.env = env
        self.env.set_environment_mode(self.set_env_mode()) 
        self.obs_dim = env.get_observation_space().shape[0]
        self.act_dim = env.get_action_space().shape[0]

    def set_env_mode(self):
        """
        Sets the mode of the environment for the agent.
        # Depending on this implementation the outputs of the environment change. 
        """
        raise NotImplementedError("This method should be implemented by subclasses.")

    
    def train(self, episodes=10, use_tqdm=True):
        """
        Trains the agent for a specified number of episodes.
        Parameters:
            episodes (int): Number of episodes to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def evaluate(self, eval_env, total_timesteps=1, use_tqdm=True):
        """
        Evaluates the agent for a specified number of episodes on the environment.
        Parameters:
            episodes (int): Number of episodes to evaluate the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def save(self, path):
        """
        Saves the agent's state to a file.
        Parameters:
            path (str): Path to save the agent's state.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def load(self, path):
        """
        Loads the agent's state from a file.
        Parameters:
            path (str): Path to load the agent's state from.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def visualize(self):
        """
        Visualizes the agent's performance or learned policy.
        This method can be overridden by subclasses to provide specific visualizations.
        """
        return Warning("This method should be implemented by subclasses.")