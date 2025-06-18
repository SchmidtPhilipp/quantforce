import importlib
import json
import os

import numpy as np
from tqdm import tqdm

from qf import INFO_VERBOSITY
from qf.envs.multi_agent_portfolio_env import MultiAgentPortfolioEnv
from qf.utils.logging_config import get_logger

logger = get_logger(__name__)


class Agent:
    def __init__(self, env, verbosity=INFO_VERBOSITY):
        """
        Initializes the agent with the given environment.
        Parameters:

            env: The environment in which the agent will operate.
        """
        self.env = env
        self.obs_dim = env.observation_space.shape[0]
        self.act_dim = env.action_space.shape[0]
        self.device = env.device
        self.verbosity = verbosity

    def train(
        self,
        total_timesteps: int,
        eval_env=None,
        n_eval_steps=None,
        n_eval_episodes=5,
        save_best=True,
        print_eval_metrics=True,
        use_tqdm=True,
        save_checkpoints=True,
    ) -> None:
        """Train the agent with optional evaluation.

        Args:
            total_timesteps: Total number of timesteps to train for
            eval_env: Environment to use for evaluation
            n_eval_steps: Number of steps between evaluations
            n_eval_episodes: Number of episodes to evaluate on
            save_best: Whether to save the best model
            print_eval_metrics: Whether to print evaluation metrics
            use_tqdm: Whether to use tqdm progress bar
        """
        if eval_env is None or n_eval_steps is None:
            # Train without evaluation
            self._train(total_timesteps, use_tqdm)
            return

        # Calculate number of evaluation steps
        n_evaluations = total_timesteps // n_eval_steps
        best_mean_reward = float("-inf")

        # Training loop with evaluation
        for i in range(n_evaluations):
            # Train for n_eval_steps
            self._train(n_eval_steps, use_tqdm)

            # Evaluate
            mean_reward = self.evaluate(
                eval_env=eval_env,
                episodes=n_eval_episodes,
                print_metrics=print_eval_metrics,
            ).mean()

            if save_checkpoints:
                path = os.path.join(eval_env.get_save_dir(), f"checkpoint_{i}")
                self.save(path)
                self.env.save_data(path=path)

            # Save best model if needed
            if save_best and mean_reward > best_mean_reward:
                best_mean_reward = mean_reward
                best_model_path = os.path.join(eval_env.get_save_dir(), "best_model")
                self.save(best_model_path)
                self.env.save_data(path=best_model_path)
                if print_eval_metrics:
                    logger.info(
                        f"New best model saved with mean reward: {mean_reward:.2f}"
                    )

        # Train remaining steps
        remaining_steps = total_timesteps % n_eval_steps
        if remaining_steps > 0:
            self._train(remaining_steps, use_tqdm)

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
            if self.verbosity > INFO_VERBOSITY:
                logger.info("Running evaluation on the training environment.")
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
        if self.verbosity > INFO_VERBOSITY:
            logger.info(
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
    def load_agent(path, env=None, device="cpu"):
        """
        Loads an agent from a specified path. Use this method to load an agent that was saved with the save method.
        Parameters:
            path (str): Path to the saved agent.
            env: The environment to associate with the loaded agent. If None, will be loaded from config.
            device (str): Device to load the agent on ("cpu" or "cuda").
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

        # Create environment if not provided
        if env is None:
            env = MultiAgentPortfolioEnv(
                tensorboard_prefix="LOAD_ENV", config=env_config
            )

        # Load agent class name
        agent_class_path = os.path.join(path, "agent_class.txt")
        if not os.path.exists(agent_class_path):
            raise FileNotFoundError(f"Agent class file not found at {agent_class_path}")

        with open(agent_class_path, "r") as f:
            agent_class_name = f.read().strip()

        # Import and instantiate agent
        agent_module = importlib.import_module("qf")
        agent_class = getattr(agent_module, agent_class_name)
        agent = agent_class(env)

        # Load agent state
        agent.load(path)

        return agent
