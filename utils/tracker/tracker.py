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
        Logs all tracked values using the provided logger.

        Parameters:
            logger (Logger): Logger instance for logging.
        """
        for name, value in self.tracked_values.items():
            dimensions = value["dimensions"]
            for ep_idx, episode_data in enumerate(value["data"]):
                for ts in range(self.timesteps):
                    if len(dimensions) == 2:  # 2D data (e.g., timesteps x episodes)
                        logger.log_scalar(
                            f"{self.tensorboard_prefix}/{name}/timestep_{ts}",
                            episode_data[ts],
                            step=ep_idx,
                        )
                    elif len(dimensions) == 3:  # 3D data (e.g., timesteps x agents x episodes)
                        for agent_idx in range(episode_data.shape[1]):
                            logger.log_scalar(
                                f"{self.tensorboard_prefix}/{name}/agent_{agent_idx}/timestep_{ts}",
                                episode_data[ts, agent_idx],
                                step=ep_idx,
                            )

    def print_summary(self):
        """
        Prints a summary of the tracked values, including their dimensions and aggregated statistics.
        """
        print("📊 Tracker Summary:")
        for name, value in self.tracked_values.items():
            dimensions = value["dimensions"]
            description = value["description"]
            print(f"  - {name}: {description}")
            print(f"    Dimensions: {dimensions}")
            print(f"    Episodes Tracked: {len(value['data'])}")
            if value["data"]:
                data = np.concatenate(value["data"], axis=0)
                print(f"    Mean: {np.mean(data):.4f}, Std: {np.std(data):.4f}")

    def reset(self):
        """
        Resets the tracker for a new set of episodes.
        """
        for value in self.tracked_values.values():
            value["data"] = []
        self.current_episode = 0
        self.current_timestep = 0

    def save(self, filepath):
        """
        Saves the tracker data to a file.

        Parameters:
            filepath (str): Path to the file where the tracker data will be saved.
        """
        data = {
            "timesteps": self.timesteps,
            "tensorboard_prefix": self.tensorboard_prefix,
            "tracked_values": self.tracked_values,
            "current_episode": self.current_episode,
            "current_timestep": self.current_timestep,
        }
        with open(filepath, "wb") as f:
            pickle.dump(data, f)
        print(f"✅ Tracker data saved to {filepath}")

    def load(self, filepath):
        """
        Loads tracker data from a file.

        Parameters:
            filepath (str): Path to the file from which the tracker data will be loaded.
        """
        with open(filepath, "rb") as f:
            data = pickle.load(f)

        self.timesteps = data["timesteps"]
        self.tensorboard_prefix = data["tensorboard_prefix"]
        self.tracked_values = data["tracked_values"]
        self.current_episode = data["current_episode"]
        self.current_timestep = data["current_timestep"]
        print(f"✅ Tracker data loaded from {filepath}")