import os

import cloudpickle

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
            from stable_baselines3.common.callbacks import EvalCallback

            # Create a temporary directory for evaluation checkpoints
            temp_save_dir = os.path.join(self.env.get_save_dir(), "temp_checkpoints")
            os.makedirs(temp_save_dir, exist_ok=True)

            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=os.path.join(temp_save_dir, "best_model"),
                log_path=os.path.join(temp_save_dir, "eval_logs"),
                eval_freq=n_eval_steps,
                deterministic=True,
                render=False,
            )

            self.model.learn(
                total_timesteps=total_timesteps,
                progress_bar=True if use_tqdm else False,
                reset_num_timesteps=False,
                callback=eval_callback,
            )

            # If we have a best model, copy it to the final location
            if save_best and os.path.exists(os.path.join(temp_save_dir, "best_model")):
                import shutil

                shutil.copytree(
                    os.path.join(temp_save_dir, "best_model"),
                    os.path.join(self.env.get_save_dir(), "best_model"),
                    dirs_exist_ok=True,
                )
                shutil.rmtree(temp_save_dir)
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

        if type(eval_env) is not SB3Wrapper:
            eval_env = SB3Wrapper(eval_env)

        return super().evaluate(eval_env=eval_env, episodes=episodes, use_tqdm=use_tqdm)

    def _save_impl(self, path):
        """
        Implementation-specific save method for SB3 agent.
        Parameters:
            path (str): Path to save the agent's state.
        """
        import os

        # Save model using zip-archive format
        model_path = os.path.join(path, f"model.zip")
        self.model.save(model_path)

    def _load_impl(self, path):
        """
        Implementation-specific load method for SB3 agent.
        Parameters:
            path (str): Path to load the agent's state from.
        """
        import os

        # Load model using zip-archive format
        model_path = os.path.join(path, f"model_{self.class_name}.zip")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found at {model_path}")

        # Load the model with the current environment
        self.model = self.model.load(model_path, env=self.env)

    @staticmethod
    def load_agent(path, env, agent_class, device="cpu"):
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
        # Create agent instance
        agent = agent_class(env)

        # Load agent state
        agent.load(path)

        return agent
