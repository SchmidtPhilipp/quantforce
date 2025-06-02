from qf.agents.agent import Agent
import numpy as np
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
        Trains the SAC agent for a specified number of episodes using the `learn` method.
        Parameters:
            episodes (int): Number of episodes to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
        """

        self.model.learn(total_timesteps=total_timesteps, progress_bar=True if use_tqdm else False)


    def evaluate(self, eval_env, episodes=1, use_tqdm=True):
        """
        Evaluates the SAC agent for a specified number of episodes on the environment.
        Parameters:
            eval_env: The environment used for evaluation.
            episodes (int): Number of episodes to evaluate the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
        Returns:
            float: The average reward over the evaluation episodes.
        """
        from tqdm import tqdm

        if type(eval_env) is not SB3Wrapper:
            eval_env = SB3Wrapper(eval_env)

        total_rewards = []
        progress = tqdm(range(episodes), desc=f"Evaluating {self.__class__.__name__}") if use_tqdm else range(episodes)

        for episode in progress:
            obs,_ = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                action, _ = self.model.predict(obs, deterministic=True)
                obs, rewards, done, truncateds, info = eval_env.step(action)
                episode_reward += rewards

            total_rewards.append(episode_reward)

            if use_tqdm:
                progress.set_postfix({"Episode Reward": episode_reward})

        eval_env.print_metrics()
        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg_reward}")
        return avg_reward
    
    def set_env_mode(self):
        return "sb3"
    
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