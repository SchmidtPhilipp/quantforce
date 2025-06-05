import numpy as np
from tqdm import tqdm


class Agent:
    def __init__(self, env):
        """
        Initializes the agent with the given environment.
        Parameters:

            env: The environment in which the agent will operate.
        """
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
    
    def train(self, episodes=10, use_tqdm=True):
        """
        Trains the agent for a specified number of episodes.
        Parameters:
            episodes (int): Number of episodes to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
    
    def evaluate(self, eval_env=None, episodes=1, use_tqdm=True, print_metrics=True):
        """
        Evaluates the agent for a specified number of episodes on the environment.
        Parameters:
            eval_env: The environment used for evaluation.
            episodes (int): Number of episodes to evaluate the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
        Returns:
            np.ndarray: Rewards matrix with shape (episodes, num_steps, n_agents).
        """
        if eval_env is None:
            print("Running evaluation on the training environment.")
            eval_env = self.eval_env

        rewards_dict = {}  # Dictionary to store rewards for each episode
        max_steps = 0  # Track the maximum number of steps across episodes

        progress = tqdm(range(episodes), desc=f"Evaluating {self.__class__.__name__}") if use_tqdm else range(episodes)

        for episode in progress:
            state, info = eval_env.reset()
            done = False
            episode_reward = []

            while not done:
                action = self.act(state)
                outputs = eval_env.step(action)

                # Unpack the outputs for compatibility
                next_state = outputs[0]
                reward = outputs[1]
                done = outputs[2]

                state = next_state

                # rewards should be of [n_agents,]
                if isinstance(reward, np.ndarray) and reward.ndim == 0:
                    reward = reward[np.newaxis]

                episode_reward.append(reward)

            rewards_dict[episode] = episode_reward  # Store rewards in the dictionary
            max_steps = max(max_steps, len(episode_reward))  # Update max_steps

            if use_tqdm:
                progress.set_postfix({"Episode Reward": sum(episode_reward)})

        # Convert dictionary to a padded matrix of shape (episodes, max_steps, n_agents)
        rewards_matrix = np.array([
            np.pad(rewards_dict[episode], (0, max_steps - len(rewards_dict[episode])), mode='constant', constant_values=0)
            for episode in range(episodes)
        ])
        if print_metrics:
            eval_env.print_metrics()

        avg_reward = np.mean(rewards_matrix) # remember in rl is the reward R_t not G_t
        std_reward = np.std(rewards_matrix) 
        print(f"Average reward over {episodes} episodes: {avg_reward}, Standard Deviation: {std_reward}")
        return rewards_matrix
    
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
    
    @staticmethod
    def get_hyperparameter_space():
        """
        Returns the hyperparameters of the agent.
        This method can be overridden by subclasses to provide specific hyperparameters.
        """
        return Warning("This method should be implemented by subclasses.")

    @staticmethod
    def get_default_config():
        """
        Returns the default configuration for the agent.
        This method can be overridden by subclasses to provide specific default configurations.
        """
        return Warning("This method should be implemented by subclasses.")
    
