import numpy as np
import torch
import qf
from tqdm import tqdm
import matplotlib.pyplot as plt

class RobustnessTester:
    def __init__(self, agent_class, env_class, train_env_config, eval_env_config, agent_config, n_trials=10, n_evaluations=5):
        """
        Initializes the RobustnessTester.

        Parameters:
            agent_class (class): The agent class to test.
            env_class (class): The environment class used for training and evaluation.
            train_env_config (dict): Configuration for the training environment.
            eval_env_config (dict): Configuration for the evaluation environment.
            agent_config (dict): Configuration for the agent.
            n_trials (int): Number of trials for robustness testing.
            n_evaluations (int): Number of evaluations during training (e.g., after every n steps).
        """
        self.agent_class = agent_class
        self.env_class = env_class
        self.train_env_config = train_env_config
        self.eval_env_config = eval_env_config
        self.agent_config = agent_config
        self.n_trials = n_trials
        self.n_evaluations = n_evaluations

        self.balances = None  # Placeholder for the balances matrix

    def test_progress(self, total_timesteps=5000, episodes=10, use_tqdm=True):
        """
        Tests the agent's training progress by evaluating it after every `evaluation_interval` steps.

        Parameters:
            total_timesteps (int): Total number of timesteps for training.
            episodes (int): Number of episodes for evaluation.
            evaluation_interval (int, optional): Interval of timesteps between evaluations.
            use_tqdm (bool): Whether to use tqdm for progress tracking.

        Returns:
            dict: Contains the balances matrix and other statistics.
        """
        
        evaluation_interval = total_timesteps // self.n_evaluations
        episode_length = self.env_class(tensorboard_prefix=f"", config=self.train_env_config).get_timesteps()


        # Initialize the balances matrix
        balances = np.zeros((self.n_trials, self.n_evaluations, episodes, episode_length, self.train_env_config["n_agents"]))

        progress = tqdm(range(self.n_trials), desc="Testing Training Progress") if use_tqdm else range(self.n_trials)

        for trial in progress:
            # Set a fixed random seed for reproducibility
            seed = trial
            np.random.seed(seed)
            torch.manual_seed(seed)

            # Initialize environment and agent
            train_env = self.env_class(tensorboard_prefix=f"TRAIN_TRIAL_{trial}", config=self.train_env_config)
            agent = self.agent_class(train_env, config=self.agent_config)

            for eval_step in range(self.n_evaluations):
                # Train the agent for `evaluation_interval` steps
                agent.train(total_timesteps=evaluation_interval, use_tqdm=use_tqdm)

                # Evaluate the agent
                eval_env = self.env_class(tensorboard_prefix=f"EVAL_TRIAL_{trial}_STEP_{eval_step}", config=self.eval_env_config)
                rewards_matrix = agent.evaluate(eval_env, episodes=episodes, use_tqdm=use_tqdm)

                # Get the balances from the evaluation environment tracker
                if hasattr(eval_env, 'tracker') and "balance" in eval_env.tracker.tracked_values:
                    result = eval_env.tracker.get_value("balance")
                    # if the evaluation ended before the end of the episode, we need to pad the result

                    balances[trial, eval_step] = eval_env.tracker.get_value("balance")

        self.balances = balances

        return {
            "balances": balances,
            "shape": balances.shape
        }

    def plot_portfolio_balance(self):
        """
        Plots the mean portfolio balance and the 3-sigma regions over the environment timesteps.

        Returns:
            None: Displays the plot.
        """
        if self.balances is None:
            raise ValueError("No balances data available. Please run the robustness test first.")

        # Flatten balances across trials and evaluations
        balances_flat = self.balances.reshape(-1, self.balances.shape[-2], self.balances.shape[-1])

        # Calculate mean and standard deviation
        mean_balance = np.mean(balances_flat, axis=0)
        std_balance = np.std(balances_flat, axis=0)

        # Calculate 3-sigma regions
        upper_bound = mean_balance + 3 * std_balance
        lower_bound = mean_balance - 3 * std_balance

        # Plot the data
        timesteps = np.arange(mean_balance.shape[0])
        plt.figure(figsize=(10, 6))
        plt.plot(timesteps, mean_balance, label="Mean Balance", color="blue")
        plt.fill_between(timesteps, lower_bound, upper_bound, color="blue", alpha=0.2, label="3-Sigma Region")
        plt.title("Portfolio Balance Over Time")
        plt.xlabel("Timesteps")
        plt.ylabel("Portfolio Balance")
        plt.legend()
        plt.grid(True)
        plt.show()