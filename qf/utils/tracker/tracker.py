import json  # For loading data from JSON files
import os
import pickle  # For saving and loading data

import torch

from qf import (
    DEBUG_VERBOSITY,
    ERROR_VERBOSITY,
    INFO_VERBOSITY,
    VERBOSITY,
    WARNING_VERBOSITY,
)
from qf.utils.logging_config import get_logger

# Get logger for this module
logger = get_logger(__name__)


class Tracker:
    def __init__(
        self,
        timesteps,
        tensorboard_prefix="tracker",
        device="cpu",
        verbosity=VERBOSITY,
    ):
        """
        Generalized tracker for storing and logging values.

        Parameters:
            timesteps (int): Maximum number of timesteps per episode.
            tensorboard_prefix (str): Prefix for TensorBoard logging.
            device (str): Device to use for storing tensors ("cpu" or "cuda").
        """
        self.timesteps = timesteps
        self.tensorboard_prefix = tensorboard_prefix
        self.device = device
        self.verbosity = verbosity

        # Dictionary to store registered values
        self.tracked_values = {}

        # Track the current episode and timestep
        self.current_episode = 0
        self.current_timestep = 0

    def register_value(self, name, shape, description="", dimensions=None, labels=None):
        """
        Registers a value to be tracked.

        Parameters:
            name (str): Name of the value (e.g., "actions", "balances").
            shape (tuple): Shape of the value (excluding timesteps and episodes).
            description (str): Optional description of the value.
            dimensions (list): Description of each dimension (e.g., ["timesteps", "agents", "assets"]).
            labels (list): List of labels for each dimension (e.g., [["AAPL", "MSFT"], range(n_agents)]).
                           If None, defaults to `range()` for each dimension.
        """
        if name in self.tracked_values:
            raise ValueError(f"Value '{name}' is already registered.")

        # Default labels to range() if not provided
        if labels is None:
            labels = [range(dim) for dim in shape]

        if len(labels) != len(shape):
            raise ValueError(
                f"Number of labels ({len(labels)}) must match the number of dimensions in shape ({len(shape)})."
            )

        # Validate that each label matches the corresponding dimension size
        # for i, (label, dim_size) in enumerate(zip(labels, shape)):
        #    if len(label) != dim_size:
        #        raise ValueError(f"Label size for dimension {i} ({len(label)}) does not match shape size ({dim_size}).")

        self.tracked_values[name] = {
            "data": [],
            "shape": shape,
            "description": description,
            "dimensions": dimensions or [],
            "labels": labels,
        }

    def get_value(self, name):
        """
        Retrieves the data for a specific tracked value.

        Parameters:
            name (str): Name of the tracked value.

        Returns:
            list: Data for the specified tracked value.
        """
        if name not in self.tracked_values:
            raise ValueError(f"Value '{name}' is not registered.")
        return self.tracked_values[name]["data"]

    def get_labels(self, name):
        """
        Retrieves the labels for a specific tracked value.

        Parameters:
            name (str): Name of the tracked value.

        Returns:
            list: Labels for each dimension of the tracked value.
        """
        if name not in self.tracked_values:
            raise ValueError(f"Value '{name}' is not registered.")
        return self.tracked_values[name]["labels"]

    def _ensure_episode_exists(self, episode_idx):
        """
        Ensures that the data structures are large enough to handle the given episode index.

        Parameters:
            episode_idx (int): The episode index to ensure exists.
        """
        for value in self.tracked_values.values():
            while len(value["data"]) <= episode_idx:
                value["data"].append(
                    torch.zeros((self.timesteps, *value["shape"]), device=self.device)
                )

    def record_step(self, **kwargs):
        """
        Records values for a single step.

        Parameters:
            kwargs: Key-value pairs of tracked values to record.
        """
        ep = self.current_episode
        ts = self.current_timestep

        if self.verbosity > DEBUG_VERBOSITY:
            logger.debug(f"Episode: {ep}, Timestep: {ts}")

        # Ensure the current episode exists
        self._ensure_episode_exists(ep)

        for name, value in kwargs.items():
            if name not in self.tracked_values:

                if self.verbosity > WARNING_VERBOSITY:
                    logger.warning(f"⚠️ Value '{name}' is not registered. Skipping.")
                continue
            expected_shape = self.tracked_values[name]["shape"]
            if value.shape != expected_shape:
                raise ValueError(
                    f"Shape mismatch for '{name}'. Expected {expected_shape}, got {value.shape}."
                )
            try:
                self.tracked_values[name]["data"][ep][ts] = value.to(self.device)
            except Exception as e:
                # Give full error message
                raise ValueError(
                    f"Error recording step {ts} for episode {ep} for value {name}: {e}. \n"
                )

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
            torch.Tensor: The requested data for the specified episode.
        """
        if name not in self.tracked_values:
            return
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
            if self.verbosity > WARNING_VERBOSITY:
                logger.warning("⚠️ No episodes to log yet.")
            return

        # Get the last episode index
        last_episode_idx = self.current_episode - 1

        # Log based on the dimensions of the tracked value
        for ts in range(self.timesteps):
            # Iterate over all tracked values
            for name, value in self.tracked_values.items():
                dimensions = value["dimensions"]
                episode_data = value["data"][
                    last_episode_idx
                ]  # Data for the last episode
                labels = value["labels"]  # Retrieve labels for naming

                if len(dimensions) == 2:  # 2D data (e.g., timesteps x agents)

                    if type(labels[0]) is not list:
                        labels[0] = [labels[0]]

                    for agent_idx, agent_label in enumerate(
                        labels[0]
                    ):  # Use agent labels
                        logger.log_scalar(
                            f"{self.tensorboard_prefix}_{name}/Actor_{agent_label}",
                            episode_data[ts, agent_idx],
                        )

                elif (
                    len(dimensions) == 3
                ):  # 3D data (e.g., timesteps x agents x assets)
                    for agent_idx, agent_label in enumerate(
                        labels[0]
                    ):  # Use agent labels
                        for asset_idx, asset_label in enumerate(
                            labels[1]
                        ):  # Use asset labels
                            logger.log_scalar(
                                f"{self.tensorboard_prefix}_{name}/{agent_label}/{asset_label}",
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
            if self.verbosity > WARNING_VERBOSITY:
                logger.warning("⚠️ No episodes to summarize yet.")
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
        total_reward = torch.sum(rewards, axis=0)

        # Print episode summary
        logger.info(f"📈[{run_type}] Episode {episode + 1:>3} | Steps: {steps}")
        if isinstance(total_reward, (list, torch.Tensor)):  # Multi-agent rewards
            agent_rewards_str = " -> ".join(
                [
                    f"Agent {i}: {agent_reward:.4f}"
                    for i, agent_reward in enumerate(total_reward)
                ]
            )
        else:  # Single-agent reward
            agent_rewards_str = f"Agent 0: {total_reward:.4f}"

        if self.verbosity > INFO_VERBOSITY:
            logger.info(f"Rewards: {agent_rewards_str}")
            logger.info(f"Portfolio Value: {env_balance[-1][0]:.4f}")
            logger.info(f"Total Reward: {torch.sum(total_reward):.4f}")
            logger.info(f"Asset Holdings: {asset_holdings[steps - 1]}")
            logger.info(f"Agent Balances: {actor_balances[steps - 1]}")

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
            save_data[name] = {
                "data": [
                    v.tolist() for v in value["data"]
                ],  # Konvertiere Tensoren in Listen
                "shape": value["shape"],
                "description": value["description"],
                "dimensions": value["dimensions"],
                "labels": value["labels"],
            }

        # Save as a JSON file
        save_path = os.path.join(run_path, "tracker_data.json")
        with open(save_path, "w") as f:
            json.dump(save_data, f, indent=4)
        if self.verbosity > INFO_VERBOSITY:
            logger.info(f"✅ Tracker-Daten wurden in {save_path} gespeichert.")

    @staticmethod
    def load(filepath):
        """
        Lädt Tracker-Daten aus einer JSON-Datei.

        Parameters:
            filepath (str): Pfad zur Datei, aus der die Tracker-Daten geladen werden sollen.

        Returns:
            Tracker: Eine Tracker-Instanz mit den geladenen Daten.
        """
        with open(filepath, "r") as f:
            data = json.load(f)

        # Neue Tracker-Instanz erstellen
        tracker = Tracker(
            timesteps=data["timesteps"], tensorboard_prefix=data["tensorboard_prefix"]
        )
        tracker.current_episode = data["current_episode"]
        tracker.current_timestep = data["current_timestep"]

        # Verfolgte Werte laden
        for name, value in data.items():
            if name not in [
                "timesteps",
                "tensorboard_prefix",
                "current_episode",
                "current_timestep",
            ]:
                tracker.tracked_values[name] = {
                    "data": [
                        torch.tensor(v) for v in value["data"]
                    ],  # Konvertiere Listen zurück in Tensoren
                    "shape": value["shape"],
                    "description": value["description"],
                    "dimensions": value["dimensions"],
                    "labels": value["labels"],
                }

        if tracker.verbosity > INFO_VERBOSITY:
            logger.info(f"✅ Tracker-Daten wurden aus {filepath} geladen.")
        return tracker

    def log_statistics(self, logger, values_to_log=None):
        """
        Logs the statistics (mean and standard deviation) of specific tracked values.

        Parameters:
            logger (Logger): Logger instance for logging.
            values_to_log (list, optional): List of specific tracked values to log.
                                             If None, logs all registered values.
        """
        if self.current_episode == 0:
            if self.verbosity > WARNING_VERBOSITY:
                logger.warning("⚠️ No episodes to log statistics for yet.")
            return

        # If no specific values are provided, log all tracked values
        if values_to_log is None:
            values_to_log = list(self.tracked_values.keys())

        # Iterate over timesteps
        for timestep in range(self.timesteps):
            # Iterate over the specified tracked values
            for name in values_to_log:
                if name not in self.tracked_values:
                    if self.verbosity > WARNING_VERBOSITY:
                        logger.warning(f"⚠️ Value '{name}' is not registered. Skipping.")
                    continue

                value = self.tracked_values[name]
                dimensions = value["dimensions"]
                data = value["data"]
                labels = value["labels"]  # Retrieve labels for naming

                # Collect data for all episodes at the current timestep
                timestep_data = torch.stack(
                    [ep[timestep] for ep in data if timestep < len(ep)]
                )

                if len(dimensions) == 2:  # 2D data (e.g., timesteps x agents)
                    for agent_idx, agent_label in enumerate(
                        labels[0]
                    ):  # Use agent labels
                        logger.log_scalar(
                            f"{self.tensorboard_prefix}_{name}_mean/{agent_label}",
                            torch.mean(timestep_data[:, agent_idx]),
                            step=timestep,
                        )
                        logger.log_scalar(
                            f"{self.tensorboard_prefix}_{name}_std/{agent_label}",
                            torch.std(timestep_data[:, agent_idx]),
                            step=timestep,
                        )

                elif (
                    len(dimensions) == 3
                ):  # 3D data (e.g., timesteps x agents x assets)
                    for agent_idx, agent_label in enumerate(
                        labels[0]
                    ):  # Use agent labels
                        for asset_idx, asset_label in enumerate(
                            labels[1]
                        ):  # Use asset labels
                            logger.log_scalar(
                                f"{self.tensorboard_prefix}_{name}_mean/Actor_{agent_label}/{asset_label}",
                                torch.mean(timestep_data[:, agent_idx, asset_idx]),
                                step=timestep,
                            )
                            logger.log_scalar(
                                f"{self.tensorboard_prefix}_{name}_std/Actor_{agent_label}/{asset_label}",
                                torch.std(timestep_data[:, agent_idx, asset_idx]),
                                step=timestep,
                            )

                elif len(dimensions) == 1:  # 1D data (e.g., timesteps)
                    logger.log_scalar(
                        f"{self.tensorboard_prefix}_{name}_mean",
                        torch.mean(timestep_data),
                        step=timestep,
                    )
                    logger.log_scalar(
                        f"{self.tensorboard_prefix}_{name}_std",
                        torch.std(timestep_data),
                        step=timestep,
                    )

    def get_df(self, name):
        """
        Converts the tracked values for a specific name to a pandas DataFrame.

        Parameters:
            name (str): Name of the tracked value.

        Returns:
            pd.DataFrame: DataFrame containing the tracked values for the specified name.
        """
        import numpy as np
        import pandas as pd

        if name not in self.tracked_values:
            raise ValueError(f"Value '{name}' is not registered.")

        value = self.tracked_values[name]

        # Extract the data for the current tracking element
        data = np.array(value["data"])  # Shape: (episodes, timesteps, ...)
        dimensions = value["dimensions"]
        labels = value["labels"]

        # Handle "date" separately
        # Get date from value["data"]
        dates = self.tracked_values.get("date", None)
        dates = dates["data"][0]
        dates = torch.tensor(dates, dtype=torch.int32).detach().clone()
        dates = dates[:, 0]
        dates = [pd.Timestamp.fromtimestamp(date.item()) for date in dates]
        dates = pd.to_datetime(dates)  # Convert to pandas datetime

        # Reshape the data into a 2D form for the DataFrame
        reshaped_data = data.reshape(
            -1, *value["shape"]
        )  # Shape: (episodes * timesteps, ...)
        reshaped_data = reshaped_data.reshape(
            reshaped_data.shape[0], -1
        )  # Flatten for DataFrame

        # Create MultiIndex for the columns based on dimensions and labels
        if len(dimensions) > 1:
            column_index = pd.MultiIndex.from_product(labels, names=dimensions[1:])
            # Create DataFrame for the current tracking element
            df = pd.DataFrame(reshaped_data, columns=column_index)

        else:
            column_index = pd.Index(labels[0], name=name)
            # Create DataFrame for the current tracking element
            df = pd.DataFrame(reshaped_data, columns=column_index)

        # Add timestamps as the index if available
        if dates is not None:
            df.index = dates
            df.index.name = "timestamp"

        return df

    @staticmethod
    def get_tracker_files(folders):
        """
        Sammelt alle Tracker-Dateien aus den angegebenen Ordnern.

        :param folders: Liste von Ordnern, in denen nach Tracker-Dateien gesucht wird.
        :return: Liste von Pfaden zu den gefundenen Tracker-Dateien.
        """
        import os

        tracker_files = []
        for folder in folders:
            for root, _, files in os.walk(folder):
                for file in files:
                    if file.endswith(".json") and "tracker" in file.lower():
                        tracker_files.append(os.path.join(root, file))
        return tracker_files

    @staticmethod
    def get_trackers(folders):
        """
        Sammelt alle Tracker-Instanzen aus den angegebenen Ordnern.

        :param folders: Liste von Ordnern, in denen nach Tracker-Dateien gesucht wird.
        :param names: Optional, Liste von Namen für die Tracker.
        :return: Liste von Tracker-Instanzen.
        """
        import os

        import pandas as pd

        # Get tracker files from the folders
        tracker_files = Tracker.get_tracker_files(folders)

        trackers = []
        for tracker_file in tracker_files:
            trackers.append(Tracker.load(tracker_file))

        return trackers

    @staticmethod
    def get_df_from_trackers(folders, df_request, col_names=None):
        """
        Sammelt Daten aus den Trackern in den angegebenen Ordnern und gibt sie als DataFrame zurück.
        :param folders: Liste von Ordnern, in denen nach Tracker-Dateien gesucht wird.
        :param names: Optional, Liste von Namen für die Tracker.
        :return: Dictionary mit DataFrames für "rewards", "balance" und "actions".
        """
        trackers = Tracker.get_trackers(folders)

        # Load data from the trackers
        df = []

        for tracker in trackers:
            df.append(tracker.get_df(df_request)[df_request])

        import pandas as pd

        # Combine balance dataframes
        df = pd.concat(df, axis=1)

        if col_names is not None:
            # Rename the columns in the balance dataframe
            df.columns = col_names

        return df
