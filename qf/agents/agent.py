import importlib
import json
import os

import numpy as np
from tqdm import tqdm


class Agent:
    def __init__(self, env, config=None):
        """
        Initializes the agent with the given environment.
        Parameters:

            env: The environment in which the agent will operate.
        """
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.config = config
        self.device = env.device

    def train(
        self,
        episodes=10,
        use_tqdm=True,
        eval_env=None,
        n_eval_steps=None,
        save_best=True,
    ):
        """
        Trains the agent for a specified number of episodes.
        Parameters:
            episodes (int): Number of episodes to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
            eval_env: Optional environment for evaluation during training.
            n_eval_steps (int): Number of training steps between evaluations. If None, no evaluation is performed.
            save_best (bool): If True, saves the best performing agent based on evaluation.
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
            eval_env = self.env

        rewards_dict = {}  # Dictionary to store rewards for each episode
        max_steps = 0  # Track the maximum number of steps across episodes

        progress = (
            tqdm(range(episodes), desc=f"Evaluating {self.__class__.__name__}")
            if use_tqdm
            else range(episodes)
        )

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
        rewards_matrix = np.array(
            [
                np.pad(
                    rewards_dict[episode],
                    (0, max_steps - len(rewards_dict[episode])),
                    mode="constant",
                    constant_values=0,
                )
                for episode in range(episodes)
            ]
        )
        if print_metrics:
            eval_env.print_metrics()

        avg_reward = np.mean(rewards_matrix)  # remember in rl is the reward R_t not G_t
        std_reward = np.std(rewards_matrix)
        print(
            f"Average reward over {episodes} episodes: {avg_reward}, Standard Deviation: {std_reward}"
        )

        # End of evaluation save data
        eval_env.save_data()
        eval_env.log_metrics(logger=eval_env.get_logger(), run_type="EVAL")

        # Save agent configs
        if self.config is not None:
            # Save config as a json file
            with open(eval_env.get_save_dir() + "/agent_config.json", "w") as f:
                json.dump(self.config, f, indent=4)

        # Dump agent
        self.save(eval_env.get_save_dir())

        return rewards_matrix

    def save(self, path):
        """
        Saves the agent's state to a file.
        Parameters:
            path (str): Path to save the agent's state.
        """
        # Create directory if it doesn't exist
        os.makedirs(path, exist_ok=True)

        # Save agent config
        if self.config is not None:
            with open(os.path.join(path, "agent_config.json"), "w") as f:
                json.dump(self.config, f, indent=4)

        # Save environment config
        if hasattr(self.env, "config"):
            with open(os.path.join(path, "env_config.json"), "w") as f:
                json.dump(self.env.config, f, indent=4)

        # Save agent class name for loading
        with open(os.path.join(path, "agent_class.txt"), "w") as f:
            f.write(self.__class__.__name__)

        # Call subclass-specific save implementation
        self._save_impl(path)

    def _save_impl(self, path):
        """
        Implementation-specific save method to be overridden by subclasses.
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
        # Load agent config
        config_path = os.path.join(path, "agent_config.json")
        if os.path.exists(config_path):
            with open(config_path, "r") as f:
                self.config = json.load(f)

        # Call subclass-specific load implementation
        self._load_impl(path)

    def _load_impl(self, path):
        """
        Implementation-specific load method to be overridden by subclasses.
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

    @staticmethod
    def load_agent(path):
        """
        Loads an agent from a specified path.
        Parameters:
            path (str): Path to the saved agent.
        Returns:
            Agent: An instance of the loaded agent.
        """
        # Load environment config
        env_config_path = os.path.join(path, "env_config.json")
        if not os.path.exists(env_config_path):
            raise FileNotFoundError(
                f"Environment config not found at {env_config_path}"
            )

        with open(env_config_path, "r") as f:
            env_config = json.load(f)

        # Create environment
        env_module = importlib.import_module("qf.envs")
        env_class = getattr(env_module, env_config["class"])
        env = env_class(**env_config["params"])

        # Load agent class name
        agent_class_path = os.path.join(path, "agent_class.txt")
        if not os.path.exists(agent_class_path):
            raise FileNotFoundError(f"Agent class file not found at {agent_class_path}")

        with open(agent_class_path, "r") as f:
            agent_class_name = f.read().strip()

        # Import and instantiate agent
        agent_module = importlib.import_module("qf.agents")
        agent_class = getattr(agent_module, agent_class_name)
        agent = agent_class(env)

        # Load agent state
        agent.load(path)

        return agent

    def _evaluate_during_training(self, eval_env, n_episodes=5):
        """
        Evaluates the agent during training.
        Parameters:
            eval_env: The environment used for evaluation.
            n_episodes (int): Number of episodes to evaluate.
        Returns:
            float: Average reward over evaluation episodes.
        """
        rewards = self.evaluate(
            eval_env=eval_env, episodes=n_episodes, use_tqdm=False, print_metrics=False
        )
        return np.mean(rewards)
