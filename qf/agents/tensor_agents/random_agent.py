import numpy as np
from qf.agents.agent import Agent

class RandomAgent(Agent):
    def __init__(self, env, config=None):
        """
        Initializes the RandomAgent with the given environment.
        Parameters:
            env: The environment in which the agent will operate.
        """
        super().__init__(env)

    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Selects a random action from the action space of the environment.
        Parameters:
            state (np.ndarray): The current state of the environment.
        Returns:
            np.ndarray: A random action from the action space.
        """
        return self.env.action_space.sample()
    
    def train(self, total_timesteps=100000, use_tqdm=True):
        """
        Trains the RandomAgent for a specified number of timesteps.
        Parameters:
            total_timesteps (int): Total number of timesteps to train the agent.     
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print training summaries.
        """
        return  # RandomAgent does not require training
    
    def evaluate(self, eval_env, episodes=10, use_tqdm=True):
        """
        Evaluates the RandomAgent for a specified number of episodes on the environment.
        Parameters:
            eval_env: The environment used for evaluation.
            episodes (int): Number of episodes to evaluate the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
        Returns:
            float: The average reward over the evaluation episodes.
        """
        eval_env.set_environment_mode(self.set_env_mode())

        total_rewards = []
        for episode in range(episodes):
            state, _ = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action = self.act(state)
                next_state, reward, done, _, _ = eval_env.step(action)
                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg_reward}")
        return avg_reward
    

    def save(self, path):
        """
        Saves the RandomAgent's model to a file.
        Parameters:
            path (str): Path to save the model.
        """
        pass

    @staticmethod
    def load(path, env):
        """
        Loads the RandomAgent's model from a file.
        Parameters:
            path (str): Path to load the model from.
        """
        return RandomAgent(env=env)

