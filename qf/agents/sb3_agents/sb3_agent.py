import importlib
import json
import os
from typing import Optional, Union

import gymnasium as gym
import numpy as np

# improt something that is not used
import torch as tf
from stable_baselines3.common.callbacks import BaseCallback
from tqdm.auto import tqdm

from qf.agents.agent import Agent
from qf.agents.config.base_agent_config import BaseAgentConfig
from qf.envs.multi_agent_portfolio_env import MultiAgentPortfolioEnv
from qf.envs.sb3_wrapper import SB3Wrapper


class CustomEvalCallback(BaseCallback):
    def __init__(
        self,
        eval_env,
        eval_freq,
        save_best,
        agent,
        n_eval_episodes=1,
        print_metrics=True,
    ):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_best = save_best
        self.agent = agent
        self.best_mean_reward = -float("inf")
        self.n_eval_episodes = n_eval_episodes
        self.print_metrics = print_metrics
        # Create directories for checkpoints and best agent
        self.checkpoints_dir = os.path.join(agent.env.get_save_dir(), "checkpoints")
        self.best_agent_dir = os.path.join(agent.env.get_save_dir(), "best_agent")
        os.makedirs(self.checkpoints_dir, exist_ok=True)
        os.makedirs(self.best_agent_dir, exist_ok=True)

        # Save environment config
        env_config_path = os.path.join(agent.env.get_save_dir(), "env_config.json")
        with open(env_config_path, "w") as f:
            json.dump(agent.env.config, f, indent=4)

    def _on_step(self):
        if self.n_calls % self.eval_freq == 0:
            # Evaluate using our own evaluate method
            reward = self.agent.evaluate(
                eval_env=self.eval_env,
                episodes=self.n_eval_episodes,
                print_metrics=self.print_metrics,
            )

            # Save checkpoint
            checkpoint_dir = os.path.join(
                self.checkpoints_dir, f"checkpoint_{self.n_calls}"
            )
            os.makedirs(checkpoint_dir, exist_ok=True)
            self.agent.save(checkpoint_dir)

            # Save best model if needed
            if self.save_best and np.mean(reward) > self.best_mean_reward:
                self.best_mean_reward = np.mean(reward)
                self.agent.save(self.best_agent_dir)

            return True
        return True


class SB3Agent(Agent):
    # Class that inherits from Agent and gives a base implementation
    # for all stable-baselines3 agents.
    def __init__(self, env, config: Optional[BaseAgentConfig] = None):
        """
        Initializes the SB3 agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
        """
        if type(env) is not SB3Wrapper:
            env = SB3Wrapper(env)

        super().__init__(env, config=config)
        self.class_name = self.__class__.__name__

    def _train(
        self,
        total_timesteps: int,
        use_tqdm: bool = True,
    ) -> None:
        """Base training method that executes the learn function of stable baselines.

        Args:
            total_timesteps: Total number of timesteps to train for
            use_tqdm: Whether to use tqdm progress bar
        """
        self.model.learn(
            total_timesteps=total_timesteps,
            progress_bar=use_tqdm,
            reset_num_timesteps=False,
        )

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

    def evaluate(self, eval_env=None, episodes=1, use_tqdm=True, print_metrics=True):
        """
        Evaluates the agent for a specified number of episodes.
        Parameters:
            eval_env: The environment used for evaluation.
            episodes (int): Number of episodes to evaluate the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking.
        Returns:
            float: Mean reward over all episodes.
        """
        if type(eval_env) is not SB3Wrapper:
            eval_env = SB3Wrapper(eval_env)

        # Don't save during evaluation
        mean_reward = super().evaluate(
            eval_env=eval_env,
            episodes=episodes,
            use_tqdm=use_tqdm,
            print_metrics=print_metrics,
        )
        return mean_reward

    def _save_impl(self, path):
        """
        Implementation-specific save method for SB3 agent.
        Parameters:
            path (str): Path to save the agent's state.
        """
        # Save model using zip-archive format without environment
        model_path = os.path.join(path, f"model_{self.class_name}.zip")

        # First, exclude everything to find the problematic parameter
        exclude = [
            "env",
            "logger",
            "tensorboard_log",
            "_logger",
            "_custom_logger",
            "action_noise",
            "lr_schedule",
            "train",
            "ep_info_buffer",
            "ep_success_buffer",
            "_last_obs",
            "_last_episode_starts",
            "_last_original_obs",
            "_episode_num",
            "_n_updates",
            "_num_timesteps_at_start",
            "_stats_window_size",
            "_total_timesteps",
            "_current_progress_remaining",
            "start_time",
            "batch_norm_stats",
            "batch_norm_stats_target",
            "replay_buffer_class",
            "replay_buffer_kwargs",
        ]

        # Save with all parameters excluded
        self.model.save(model_path, exclude=exclude)

    def _load_impl(self, path):
        """
        Implementation-specific load method for SB3 agent.
        Parameters:
            path (str): Path to load the agent's state from.
        """
        # Load model using zip-archive format
        model_path = os.path.join(path, f"model_{self.class_name}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load the model without environment and set the current environment
        self.model = self.model.load(model_path, env=None)
        self.model.set_env(self.env)

    @staticmethod
    def load_agent(path, env=None, device="cpu"):
        """
        Loads an SB3 agent's model from a file.
        Parameters:
            path (str): Path to load the model from.
            env: The environment to associate with the loaded model.
            device (str): Device to load the model on ("cpu" or "cuda").
        Returns:
            SB3Agent: A new instance of the specified agent class with the loaded model.
        """
        # Find the model file
        model_files = [
            f for f in os.listdir(path) if f.startswith("model_") and f.endswith(".zip")
        ]
        if not model_files:
            raise FileNotFoundError(f"No model file found in {path}")

        # Get agent class name from the first model file
        model_file = model_files[0]
        agent_class_name = model_file[6:-4]  # Remove "model_" prefix and ".zip" suffix

        # Convert class name to module name (e.g., SACAgent -> sac_agent)
        # First we make all the letters lowercase
        # Then we find agent in the name and place a "_" before it
        module_name = agent_class_name.lower().replace("agent", "_agent")

        # Import the agent class
        agent_module = importlib.import_module(f"qf.agents.sb3_agents.{module_name}")
        agent_class = getattr(agent_module, agent_class_name)

        # Load the environment form the env config file of the path
        if env is None:
            env_config_path = os.path.join(path, "env_config.json")
            with open(env_config_path, "r") as f:
                env_config = json.load(f)
            env = MultiAgentPortfolioEnv(
                tensorboard_prefix="LOADED_ENV", config=env_config
            )
        # Create agent instance
        agent = agent_class(env)

        # Load agent state
        agent.load(path)

        return agent
