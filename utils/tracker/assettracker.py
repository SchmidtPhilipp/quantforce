import numpy as np
import os

class AssetTracker:
    def __init__(self, n_agents, n_assets, timesteps, tickers, tensorboard_prefix="04_eval_assets", n_episodes=1):
        """
        Initializes the AssetTracker.

        Parameters:
            n_agents (int): Number of agents.
            n_assets (int): Number of assets.
            timesteps (int): Maximum number of timesteps per episode.
            tickers (list): List of asset tickers.
            tensorboard_prefix (str): Prefix for TensorBoard logging.
        """
        self.tensorboard_prefix = tensorboard_prefix
        self.n_agents = n_agents
        self.n_assets = n_assets
        self.timesteps = timesteps
        self.tickers = tickers

        # Initialize lists to store data dynamically
        self.actions = []  # Actions (weights)
        self.asset_holdings = []  # Asset holdings
        self.actor_balances = []  # Agent balances
        self.balances = []  # Total portfolio balances
        self.rewards = []  # Rewards

        # Track the current episode and timestep
        self.current_episode = 0
        self.current_timestep = 0

    def _ensure_episode_exists(self, episode_idx):
        """
        Ensures that the data structures are large enough to handle the given episode index.

        Parameters:
            episode_idx (int): The episode index to ensure exists.
        """
        while len(self.actions) <= episode_idx:
            self.actions.append(np.zeros((self.timesteps, self.n_agents, self.n_assets + 1)))
            self.asset_holdings.append(np.zeros((self.timesteps, self.n_agents, self.n_assets)))
            self.actor_balances.append(np.zeros((self.timesteps, self.n_agents)))
            self.balances.append(np.zeros(self.timesteps))
            self.rewards.append(np.zeros((self.timesteps, self.n_agents)))

    def end_episode(self):
        """
        Ends the current episode and resets the timestep.
        """
        self.current_episode += 1
        self.current_timestep = 0

    def record_step(self, actions, asset_holdings, actor_balance, balance, reward):
        """
        Records the actions, asset holdings, balances, and rewards for a single step.

        Parameters:
            actions (np.ndarray): Actions taken by agents (shape: [n_agents, n_assets + 1]).
            asset_holdings (np.ndarray): Asset holdings of agents (shape: [n_agents, n_assets]).
            actor_balance (np.ndarray): Balances of agents (shape: [n_agents]).
            balance (float): Total portfolio balance.
            reward (np.ndarray): Rewards for agents (shape: [n_agents]).
        """
        ep = self.current_episode
        ts = self.current_timestep

        # Ensure the current episode exists
        self._ensure_episode_exists(ep)

        # Store data in arrays
        self.actions[ep][ts] = actions
        self.asset_holdings[ep][ts] = asset_holdings
        self.actor_balances[ep][ts] = actor_balance
        self.balances[ep][ts] = balance
        self.rewards[ep][ts] = reward

        # Increment timestep
        self.current_timestep += 1

    def get_episode_data(self, episode_idx):
        """
        Retrieves all data for a specific episode.

        Parameters:
            episode_idx (int): Index of the episode.

        Returns:
            dict: A dictionary containing actions, asset holdings, balances, and rewards for the episode.
        """
        self._ensure_episode_exists(episode_idx)
        return {
            "actions": self.actions[episode_idx],
            "asset_holdings": self.asset_holdings[episode_idx],
            "actor_balances": self.actor_balances[episode_idx],
            "balances": self.balances[episode_idx],
            "rewards": self.rewards[episode_idx],
        }

    def get_episode_data(self, data_name, episode_idx=None):
        """
        Retrieves all data for a specific episode.

        Parameters:
            data_name (str): Name of the data to retrieve (actions, asset_holdings, actor_balances, balances, rewards).
            episode_idx (int): Index of the episode.

        Returns:
            np.ndarray: The requested data for the specified episode.
        """
        if episode_idx is None:
            episode_idx = self.current_episode-1

        self._ensure_episode_exists(episode_idx)

        if data_name == "actions":
            return self.actions[episode_idx]
        elif data_name == "asset_holdings":
            return self.asset_holdings[episode_idx]
        elif data_name == "actor_balances":
            return self.actor_balances[episode_idx]
        elif data_name == "balances":
            return self.balances[episode_idx]
        elif data_name == "rewards":
            return self.rewards[episode_idx]
        else:
            raise ValueError(f"Invalid data name: {data_name}")

    def calculate_statistics(self):
        """
        Calculates mean and standard deviation across episodes for evaluation.

        Returns:
            dict: A dictionary containing mean and std for balances, actions, and rewards.
        """
        actions = np.concatenate(self.actions, axis=0)
        asset_holdings = np.concatenate(self.asset_holdings, axis=0)
        actor_balances = np.concatenate(self.actor_balances, axis=0)
        balances = np.concatenate(self.balances, axis=0)
        rewards = np.concatenate(self.rewards, axis=0)

        return {
            "mean_balances": np.mean(balances, axis=0),
            "std_balances": np.std(balances, axis=0),
            "mean_actions": np.mean(actions, axis=0),
            "std_actions": np.std(actions, axis=0),
            "mean_rewards": np.mean(rewards, axis=0),
            "std_rewards": np.std(rewards, axis=0),
        }

    def reset(self):
        """
        Resets the tracker for a new set of episodes.
        """
        self.actions = []
        self.asset_holdings = []
        self.actor_balances = []
        self.balances = []
        self.rewards = []
        self.current_episode = 0
        self.current_timestep = 0

    def log_episode(self, logger):
        """
        Logs the total reward for each agent at the end of the episode.

        Parameters:
            logger (Logger): Logger instance for logging.
        """

        tickers = self.tickers
        for timestep in range(0, self.timesteps):
            # Log portfolio balance
            for agent_idx in range(self.n_agents):
                # Log balances
                logger.log_scalar(f"{self.tensorboard_prefix}_balance/agent_{agent_idx}/balance", self.actor_balances[self.current_episode][timestep][agent_idx])

                # Log actions (weights)
                for asset_idx, ticker in enumerate(tickers):
                    logger.log_scalar(f"{self.tensorboard_prefix}_weights/agent_{agent_idx}/{ticker}_weight", self.actions[self.current_episode][timestep][agent_idx, asset_idx])
                logger.log_scalar(f"{self.tensorboard_prefix}_weights/agent_{agent_idx}/cash_weight", self.actions[self.current_episode][timestep][agent_idx, -1])

                # Log asset holdings
                for asset_idx, ticker in enumerate(tickers):
                    logger.log_scalar(f"{self.tensorboard_prefix}_assets/agent_{agent_idx}/{ticker}_holding", self.asset_holdings[self.current_episode][timestep][agent_idx, asset_idx])

                # Log rewards
                if isinstance(self.rewards[self.current_episode][timestep], (list, np.ndarray)):  # Multi-agent reward
                    logger.log_scalar(f"{self.tensorboard_prefix}_reward/agent_{agent_idx}_reward", self.rewards[self.current_episode][timestep][agent_idx])
                else:  # Single-agent reward
                    logger.log_scalar(f"{self.tensorboard_prefix}_reward/agent_0_reward", self.rewards[self.current_episode][timestep])
                
            # Log total portfolio value
            logger.log_scalar(f"{self.tensorboard_prefix}_portfolio/portfolio_value", self.balances[self.current_episode][timestep])

            logger.next_step()

        ##########################################################

        total_reward = np.sum(self.rewards[self.current_episode], axis=0)
        if self.n_agents > 1:
            for agent_idx, agent_reward in enumerate(total_reward):
                    logger.log_scalar(f"{self.tensorboard_prefix}_total_reward_of_episode/agent_{agent_idx}_total_reward_of_episode", agent_reward, step=self.current_episode)
        else:  # Single-agent reward
            logger.log_scalar(f"{self.tensorboard_prefix}_total_reward_of_episode/agent_0_total_reward_of_episode", total_reward, step=self.current_episode)


    def log_statistics(self, logger):
        """
        Logs the statistics of the actions, asset holdings, and balances.

        Parameters:
            logger (Logger): Logger instance for logging.
        """
        for timestep in range(self.timesteps):
        # Log statistics for each agent
            for agent_idx in range(self.n_agents):
                logger.log_scalar(f"{self.tensorboard_prefix}_balance/agent_{agent_idx}/mean_balance", np.mean([ep[timestep, agent_idx] for ep in self.actor_balances], axis=0), step=timestep)
                logger.log_scalar(f"{self.tensorboard_prefix}_balance/agent_{agent_idx}/std_balance", np.std([ep[timestep, agent_idx] for ep in self.actor_balances], axis=0), step=timestep)
                logger.log_scalar(f"{self.tensorboard_prefix}_reward/agent_{agent_idx}/mean_reward", np.mean([ep[timestep, agent_idx] for ep in self.rewards], axis=0), step=timestep)
                logger.log_scalar(f"{self.tensorboard_prefix}_reward/agent_{agent_idx}/std_reward", np.std([ep[timestep, agent_idx] for ep in self.rewards], axis=0), step=timestep)

                for i, t in enumerate(self.tickers):
                    logger.log_scalar(f"{self.tensorboard_prefix}_assets_mean/agent_{agent_idx}/{t}_mean_weight", np.mean([ep[timestep, agent_idx, i] for ep in self.asset_holdings], axis=0), step=timestep)
                    logger.log_scalar(f"{self.tensorboard_prefix}_assets_std/agent_{agent_idx}/{t}_std_weight", np.std([ep[timestep, agent_idx, i] for ep in self.asset_holdings], axis=0), step=timestep)

            logger.log_scalar(f"{self.tensorboard_prefix}_portfolio/mean_portfolio_value", np.mean([ep[timestep] for ep in self.balances], axis=0), step=timestep)
            logger.log_scalar(f"{self.tensorboard_prefix}_portfolio/std_portfolio_value", np.std([ep[timestep] for ep in self.balances], axis=0), step=timestep)

    def print_episode_summary(self, run_type=None, episode=None):
        """
        Prints a summary of the actions, asset holdings, and balances over the episode.
        """
        if run_type is None:
            run_type = self.tensorboard_prefix
        
        if episode is None:
            episode = self.current_episode

        steps = self.current_timestep
        total_reward = np.sum(self.rewards[episode], axis=0)

        # Print episode summary
        if isinstance(total_reward, (list, np.ndarray)):  # Multi-agent rewards
            agent_rewards_str = " -> ".join([f"Agent {i}: {agent_reward:.4f}" for i, agent_reward in enumerate(total_reward)])
        else:  # Single-agent reward
            agent_rewards_str = f"Agent 0: {total_reward:.4f}"

        print(f"📈[{run_type}] Episode {episode+1:>3} | Steps: {steps} | Rewards: {agent_rewards_str}")
        print(f"Portfolio Value: {self.balances[episode][steps-1]:.2f}")
        print(f"Total Reward: {np.sum(total_reward):.4f}")
        print(f"Asset Holdings: {self.asset_holdings[episode][steps-1]}")

    def save(self, run_path):
        """
        Saves the asset tracker data to a file.

        Parameters:
            run_path (str): Path to save the data.
        """
        run_path = os.path.join(run_path, f"asset_data.npz")
        np.savez(run_path, actions=self.actions, asset_holdings=self.asset_holdings, actor_balances=self.actor_balances, balances=self.balances, rewards=self.rewards)
        print(f"Asset tracker data saved to {run_path}.")

    @staticmethod
    def load(npz_path):
        """
        Lädt die AssetTracker-Daten aus einer NPZ-Datei.
        
        :param npz_path: Pfad zur NPZ-Datei.
        :return: Ein AssetTracker-Objekt mit geladenen Daten.
        """
        data = np.load(npz_path)
        tracker = AssetTracker(
            n_agents=data['actions'].shape[2],
            n_assets=data['actions'].shape[3] - 1,
            timesteps=data['actions'].shape[1],
            tickers=[]  # Tickers müssen separat gesetzt werden
        )
        tracker.actions = data['actions']
        tracker.asset_holdings = data['asset_holdings']
        tracker.actor_balances = data['actor_balances']
        tracker.balances = data['balances']
        tracker.rewards = data['rewards']
        return tracker