import numpy as np
import os
import pickle  # For saving and loading data

class Tracker:
    def __init__(self, timesteps, tensorboard_prefix="tracker"):
        """
        Generalized tracker for storing and logging values.

        Parameters:
            timesteps (int): Maximum number of timesteps per episode.
            tensorboard_prefix (str): Prefix for TensorBoard logging.
        """
        self.timesteps = timesteps
        self.tensorboard_prefix = tensorboard_prefix

        # Dictionary to store registered values
        self.tracked_values = {}

        # Track the current episode and timestep
        self.current_episode = 0
        self.current_timestep = 0

    def register_value(self, name, shape, description="", dimensions=None):
        """
        Registers a value to be tracked.

        Parameters:
            name (str): Name of the value (e.g., "actions", "balances").
            shape (tuple): Shape of the value (excluding timesteps and episodes).
            description (str): Optional description of the value.
            dimensions (list): Description of each dimension (e.g., ["timesteps", "agents", "assets"]).
        """
        if name in self.tracked_values:
            raise ValueError(f"Value '{name}' is already registered.")

        self.tracked_values[name] = {
            "data": [],
            "shape": shape,
            "description": description,
            "dimensions": dimensions or [],
        }

    def _ensure_episode_exists(self, episode_idx):
        """
        Ensures that the data structures are large enough to handle the given episode index.

        Parameters:
            episode_idx (int): The episode index to ensure exists.
        """
        for value in self.tracked_values.values():
            while len(value["data"]) <= episode_idx:
                value["data"].append(np.zeros((self.timesteps, *value["shape"])))

    def record_step(self, **kwargs):
        """
        Records values for a single step.

        Parameters:
            kwargs: Key-value pairs of tracked values to record.
        """
        ep = self.current_episode
        ts = self.current_timestep

        # Ensure the current episode exists
        self._ensure_episode_exists(ep)

        for name, value in kwargs.items():
            if name not in self.tracked_values:
                raise ValueError(f"Value '{name}' is not registered.")
            expected_shape = self.tracked_values[name]["shape"]
            if value.shape != expected_shape:
                raise ValueError(f"Shape mismatch for '{name}'. Expected {expected_shape}, got {value.shape}.")
            self.tracked_values[name]["data"][ep][ts] = value

        # Increment timestep
        self.current_timestep += 1

    def end_episode(self):
        """
        Ends the current episode and resets the timestep.
        """
        self.current_episode += 1
        self.current_timestep = 0

    def get_episode_data(self, name, episode_idx=None):
        """
        Retrieves data for a specific value and episode.

        Parameters:
            name (str): Name of the tracked value.
            episode_idx (int): Index of the episode. Defaults to the current episode.

        Returns:
            np.ndarray: The requested data for the specified episode.
        """
        if name not in self.tracked_values:
            raise ValueError(f"Value '{name}' is not registered.")
        if episode_idx is None:
            episode_idx = self.current_episode - 1
        self._ensure_episode_exists(episode_idx)
        return self.tracked_values[name]["data"][episode_idx]

    def log(self, logger):
        """
        Logs all tracked values for the last episode using the provided logger.

        Parameters:
            logger (Logger): Logger instance for logging.
        """
        if self.current_episode == 0:
            print("⚠️ No episodes to log yet.")
            return

        # Get the last episode index
        last_episode_idx = self.current_episode - 1

        # Log based on the dimensions of the tracked value
        for ts in range(self.timesteps):
            
            # Iterate over all tracked values
            for name, value in self.tracked_values.items():
                dimensions = value["dimensions"]
                episode_data = value["data"][last_episode_idx]  # Data for the last episode
                
                if len(dimensions) == 2:  # 2D data (e.g., timesteps x agents)
                    for agent_idx in range(episode_data.shape[1]):
                        logger.log_scalar(
                            f"{self.tensorboard_prefix}_{name}/agent_{agent_idx}",
                            episode_data[ts, agent_idx],
                        )
                elif len(dimensions) == 3:  # 3D data (e.g., timesteps x agents x assets)
                    for agent_idx in range(episode_data.shape[1]):
                        for asset_idx in range(episode_data.shape[2]):
                            logger.log_scalar(
                                f"{self.tensorboard_prefix}_{name}/agent_{agent_idx}/asset_{asset_idx}",
                                episode_data[ts, agent_idx, asset_idx],
                            )
                elif len(dimensions) == 1:  # 1D data (e.g., timesteps)
                    logger.log_scalar(
                        f"{self.tensorboard_prefix}_{name}",
                        episode_data[ts],
                    )

                logger.next_step()  # Increment the logger step

    def print_summary(self, run_type=None, episode=None):
        """
        Prints a summary of the tracked values for the last episode.

        Parameters:
            run_type (str): Optional run type (e.g., "TRAIN" or "VAL").
            episode (int): Optional episode index to summarize. Defaults to the last episode.
        """
        if self.current_episode == 0:
            print("⚠️ No episodes to summarize yet.")
            return

        if run_type is None:
            run_type = self.tensorboard_prefix

        if episode is None:
            episode = self.current_episode - 1

        # Ensure the episode exists
        self._ensure_episode_exists(episode)

        # Retrieve data for the episode
        steps = self.current_timestep
        rewards = self.get_episode_data("rewards", episode_idx=episode)
        actor_balances = self.get_episode_data("actor_balance", episode_idx=episode)
        env_balance = self.get_episode_data("balance", episode_idx=episode)
        asset_holdings = self.get_episode_data("asset_holdings", episode_idx=episode)

        # Calculate total rewards
        total_reward = np.sum(rewards, axis=0)

        # Print episode summary
        print(f"📈[{run_type}] Episode {episode + 1:>3} | Steps: {steps}")
        if isinstance(total_reward, (list, np.ndarray)):  # Multi-agent rewards
            agent_rewards_str = " -> ".join([f"Agent {i}: {agent_reward:.4f}" for i, agent_reward in enumerate(total_reward)])
        else:  # Single-agent reward
            agent_rewards_str = f"Agent 0: {total_reward:.4f}"
        print(f"  Rewards: {agent_rewards_str}")
        print(f"  Portfolio Value: {env_balance[-1][0]:.4f}")
        print(f"  Total Reward: {np.sum(total_reward):.4f}")
        print(f"  Asset Holdings: {asset_holdings[steps - 1]}")
        print(f"  Agent Balances: {actor_balances[steps - 1]}")

    def reset(self):
        """
        Resets the tracker for a new set of episodes.
        """
        for value in self.tracked_values.values():
            value["data"] = []
        self.current_episode = 0
        self.current_timestep = 0

    def save(self, run_path):
        """
        Saves the tracker data to a file.

        Parameters:
            run_path (str): Path to save the data.
        """
        # Prepare the data to save
        save_data = {
            "timesteps": self.timesteps,
            "tensorboard_prefix": self.tensorboard_prefix,
            "current_episode": self.current_episode,
            "current_timestep": self.current_timestep,
        }

        # Add all tracked values to the save data
        for name, value in self.tracked_values.items():
            save_data[name] = np.array(value["data"])

        # Save as a .npz file
        save_path = os.path.join(run_path, "tracker_data.npz")
        np.savez(save_path, **save_data)
        print(f"✅ Tracker data saved to {save_path}.")

    @staticmethod
    def load(filepath):
        """
        Loads tracker data from a file.

        Parameters:
            filepath (str): Path to the file from which the tracker data will be loaded.

        Returns:
            Tracker: A Tracker instance with the loaded data.
        """
        data = np.load(filepath, allow_pickle=True)

        # Create a new Tracker instance
        tracker = Tracker(timesteps=data["timesteps"].item(), tensorboard_prefix=data["tensorboard_prefix"].item())
        tracker.current_episode = data["current_episode"].item()
        tracker.current_timestep = data["current_timestep"].item()

        # Load tracked values
        for name in data.files:
            if name not in ["timesteps", "tensorboard_prefix", "current_episode", "current_timestep"]:
                tracker.tracked_values[name] = {
                    "data": list(data[name]),
                    "shape": data[name][0].shape[1:],  # Infer shape from the first episode
                    "description": "",
                    "dimensions": [],
                }

        print(f"✅ Tracker data loaded from {filepath}.")
        return tracker

    def log_statistics(self, logger):
        """
        Logs the statistics of all tracked values (e.g., actions, balances, rewards, asset holdings).

        Parameters:
            logger (Logger): Logger instance for logging.
        """
        if self.current_episode == 0:
            print("⚠️ No episodes to log statistics for yet.")
            return

        # Iterate over timesteps
        for timestep in range(self.timesteps):
            # Iterate over all tracked values
            for name, value in self.tracked_values.items():
                dimensions = value["dimensions"]
                data = value["data"]

                # Collect data for all episodes at the current timestep
                timestep_data = np.array([ep[timestep] for ep in data if timestep < len(ep)])

                if len(dimensions) == 2:  # 2D data (e.g., timesteps x agents)
                    for agent_idx in range(timestep_data.shape[1]):
                        logger.log_scalar(
                            f"{self.tensorboard_prefix}_{name}/agent_{agent_idx}/mean",
                            np.mean(timestep_data[:, agent_idx]),
                            step=timestep
                        )
                        logger.log_scalar(
                            f"{self.tensorboard_prefix}_{name}/agent_{agent_idx}/std",
                            np.std(timestep_data[:, agent_idx]),
                            step=timestep
                        )

                elif len(dimensions) == 3:  # 3D data (e.g., timesteps x agents x assets)
                    for agent_idx in range(timestep_data.shape[1]):
                        for asset_idx in range(timestep_data.shape[2]):
                            logger.log_scalar(
                                f"{self.tensorboard_prefix}_{name}/agent_{agent_idx}/asset_{asset_idx}/mean",
                                np.mean(timestep_data[:, agent_idx, asset_idx]),
                                step=timestep
                            )
                            logger.log_scalar(
                                f"{self.tensorboard_prefix}_{name}/agent_{agent_idx}/asset_{asset_idx}/std",
                                np.std(timestep_data[:, agent_idx, asset_idx]),
                                step=timestep
                            )

                elif len(dimensions) == 1:  # 1D data (e.g., timesteps)
                    logger.log_scalar(
                        f"{self.tensorboard_prefix}_{name}/mean",
                        np.mean(timestep_data),
                        step=timestep
                    )
                    logger.log_scalar(
                        f"{self.tensorboard_prefix}_{name}/std",
                        np.std(timestep_data),
                        step=timestep
                    )