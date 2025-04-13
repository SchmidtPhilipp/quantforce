import numpy as np

class AssetTracker:
    def __init__(self, n_episodes, n_agents, n_assets, timesteps, tickers, tensorboard_prefix="04_eval_assets"):
        """
        Initializes the AssetTracker.

        Parameters:
            n_episodes (int): Number of episodes to track.
            n_agents (int): Number of agents.
            n_assets (int): Number of assets.
            timesteps (int): Maximum number of timesteps per episode.
            tickers (list): List of asset tickers.
            tensorboard_prefix (str): Prefix for TensorBoard logging.
        """
        self.tensorboard_prefix = tensorboard_prefix
        self.n_episodes = n_episodes
        self.n_agents = n_agents
        self.n_assets = n_assets
        self.timesteps = timesteps
        self.tickers = tickers

        # Initialize arrays to store data
        self.actions = np.zeros((n_episodes, timesteps, n_agents, n_assets + 1))  # Actions (weights)
        self.asset_holdings = np.zeros((n_episodes, timesteps, n_agents, n_assets))  # Asset holdings
        self.actor_balances = np.zeros((n_episodes, timesteps, n_agents))  # Agent balances
        self.balances = np.zeros((n_episodes, timesteps))  # Total portfolio balances
        self.rewards = np.zeros((n_episodes, timesteps, n_agents))  # Rewards

        # Track the current episode and timestep
        self.current_episode = 0
        self.current_timestep = 0

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

        # Store data in arrays
        self.actions[ep, ts] = actions
        self.asset_holdings[ep, ts] = asset_holdings
        self.actor_balances[ep, ts] = actor_balance
        self.balances[ep, ts] = balance
        self.rewards[ep, ts] = reward

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
        mean_balances = np.mean(self.balances, axis=0)  # Mean across episodes
        std_balances = np.std(self.balances, axis=0)

        mean_actions = np.mean(self.actions, axis=0)
        std_actions = np.std(self.actions, axis=0)

        mean_rewards = np.mean(self.rewards, axis=0)
        std_rewards = np.std(self.rewards, axis=0)

        return {
            "mean_balances": mean_balances,
            "std_balances": std_balances,
            "mean_actions": mean_actions,
            "std_actions": std_actions,
            "mean_rewards": mean_rewards,
            "std_rewards": std_rewards,
        }

    def reset(self):
        """
        Resets the tracker for a new set of episodes.
        """
        self.actions.fill(0)
        self.asset_holdings.fill(0)
        self.actor_balances.fill(0)
        self.balances.fill(0)
        self.rewards.fill(0)
        self.current_episode = 0
        self.current_timestep = 0

    def log_episode(self, logger):
        """
        Logs the total reward for each agent at the end of the episode.

        Parameters:
            logger (Logger): Logger instance for logging.
        """

        tickers = self.tickers
        for timestep in range(self.current_timestep):
            # Log portfolio balance
            for agent_idx in range(self.n_agents):
                # Log balances
                logger.log_scalar(f"{self.tensorboard_prefix}_balance/agent_{agent_idx}/balance", self.actor_balances[self.current_episode, timestep][agent_idx])

                # Log actions (weights)
                for asset_idx, ticker in enumerate(tickers):
                    logger.log_scalar(f"{self.tensorboard_prefix}_weights/agent_{agent_idx}/{ticker}_weight", self.actions[self.current_episode, timestep][agent_idx, asset_idx])
                logger.log_scalar(f"{self.tensorboard_prefix}_weights/agent_{agent_idx}/cash_weight", self.actions[self.current_episode, timestep][agent_idx, -1])

                # Log asset holdings
                for asset_idx, ticker in enumerate(tickers):
                    logger.log_scalar(f"{self.tensorboard_prefix}_assets/agent_{agent_idx}/{ticker}_holding", self.asset_holdings[self.current_episode, timestep][agent_idx, asset_idx])

                # Log rewards
                if isinstance(self.rewards[self.current_episode, timestep], (list, np.ndarray)):  # Multi-agent reward
                    logger.log_scalar(f"{self.tensorboard_prefix}_reward/agent_{agent_idx}_reward", self.rewards[self.current_episode, timestep][agent_idx])
                else:  # Single-agent reward
                    logger.log_scalar(f"{self.tensorboard_prefix}_reward/agent_0_reward", self.rewards[self.current_episode, timestep])
                
            # Log total portfolio value
            logger.log_scalar(f"{self.tensorboard_prefix}_portfolio/portfolio_value", self.balances[self.current_episode, timestep])

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
        step = logger.step
        for timestep in range(self.timesteps):
        # Log statistics for each agent
            for agent_idx in range(self.n_agents):
                logger.log_scalar(f"{self.tensorboard_prefix}_balance/agent_{agent_idx}/mean_balance", np.mean(self.actor_balances[:,timestep, agent_idx], axis=0), step=timestep)
                logger.log_scalar(f"{self.tensorboard_prefix}_balance/agent_{agent_idx}/std_balance", np.std(self.actor_balances[:,timestep,agent_idx], axis=0), step=timestep)
                logger.log_scalar(f"{self.tensorboard_prefix}_reward/agent_{agent_idx}/mean_reward", np.mean(self.rewards[:,timestep,agent_idx], axis=0), step=timestep)
                logger.log_scalar(f"{self.tensorboard_prefix}_reward/agent_{agent_idx}/std_reward", np.std(self.rewards[:,timestep,agent_idx], axis=0), step=timestep)

            logger.log_scalar(f"{self.tensorboard_prefix}_portfolio/mean_portfolio_value", np.mean(self.balances[:,timestep], axis=0), step=timestep)
            logger.log_scalar(f"{self.tensorboard_prefix}_portfolio/std_portfolio_value", np.std(self.balances[:,timestep], axis=0), step=timestep)

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

        print(f"[{run_type}] Episode {episode+1:>3} | Steps: {steps} | Rewards: {agent_rewards_str}")
        print(f"Portfolio Value: {self.balances[episode, steps-1]:.2f}")
        print(f"Total Reward: {np.sum(total_reward):.4f}")
        print(f"Asset Holdings: {self.asset_holdings[episode, steps-1]}")

    def print_summary(self):
        """
        Prints a summary of the actions, asset holdings, and balances over the episode.
        """
        print("\n📊 Asset Tracker Summary:")
        for step in range(self.current_timestep):
            actions = self.actions[self.current_episode, step]
            holdings = self.asset_holdings[self.current_episode, step]
            actor_balance = self.actor_balances[self.current_episode, step]
            print(f"Step {step}:")
            for agent_idx in range(self.n_agents):
                print(f"  Agent {agent_idx}:")
                print(f"    Balance: {actor_balance[agent_idx]:.2f}")
                print(f"    Actions (Weights): {actions[agent_idx]}")
                print(f"    Asset Holdings: {holdings[agent_idx]}")
        print("-" * 50)


    def save(self, run_path):
        """
        Saves the asset tracker data to a file.

        Parameters:
            run_path (str): Path to save the data.
        """
        np.savez(run_path, actions=self.actions, asset_holdings=self.asset_holdings, actor_balances=self.actor_balances, balances=self.balances, rewards=self.rewards)
        print(f"Data saved to {run_path}")