import importlib
import json
import os

import cloudpickle
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
from tqdm import tqdm

from qf.agents.agent import Agent
from qf.envs.multi_agent_portfolio_env import MultiAgentPortfolioEnv
from qf.envs.sb3_wrapper import SB3Wrapper


class CustomEvalCallback(BaseCallback):
    def __init__(self, eval_env, eval_freq, save_best, agent):
        super().__init__()
        self.eval_env = eval_env
        self.eval_freq = eval_freq
        self.save_best = save_best
        self.agent = agent
        self.best_mean_reward = -float("inf")

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
            reward = self.agent.evaluate(self.eval_env, episodes=1)

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
    def __init__(self, env):
        """
        Initializes the SB3 agent with the given environment and configuration.
        Parameters:
            env: The environment in which the agent will operate.
        """
        if type(env) is not SB3Wrapper:
            env = SB3Wrapper(env)

        super().__init__(env)
        self.class_name = self.__class__.__name__

    def train(
        self,
        total_timesteps=100000,
        use_tqdm=True,
        eval_env=None,
        n_eval_steps=None,
        save_best=True,
    ):
        """
        Trains the SB3 agent for a specified number of timesteps and tracks the TD error.
        Parameters:
            total_timesteps (int): Total number of timesteps to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print summaries.
            eval_env: Optional environment for evaluation during training.
            n_eval_steps (int): Number of training steps between evaluations. If None, no evaluation is performed.
            save_best (bool): If True, saves the best performing agent based on evaluation.
        """
        # Create a callback for evaluation if needed
        if eval_env is not None and n_eval_steps is not None:

            if eval_env == self.env:
                print("Eval env is the same as the training env")
                # Instantiate a new eval env
                eval_env = SB3Wrapper(
                    MultiAgentPortfolioEnv(
                        tensorboard_prefix="EVAL_ENV", config=self.env.config
                    )
                )

            eval_callback = CustomEvalCallback(
                eval_env=SB3Wrapper(eval_env),
                eval_freq=n_eval_steps,
                save_best=save_best,
                agent=self,
            )

            self.model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True if use_tqdm else False,
                reset_num_timesteps=False,
                callback=eval_callback,
            )
        else:
            self.model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True if use_tqdm else False,
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

    def evaluate(self, eval_env=None, episodes=1, use_tqdm=True):
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
            use_tqdm=use_tqdm,  # , print_metrics=False
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
