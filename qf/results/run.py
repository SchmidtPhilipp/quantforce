import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, TYPE_CHECKING
import torch
import os
import re
from pathlib import Path
from qf.results.episode import Episode
from qf.results.tensorview import TensorView
from qf.results.plotframe import PlotFrame
import copy

if TYPE_CHECKING:
    from qf.results.run import Run


@dataclass
class Run:
    """
    Holds the results of a complete run, which consists of multiple episodes.
    """

    episodes: List[Episode] = field(default_factory=list)
    run_name: str = "default_run"
    tickers: List[str] = field(default_factory=list)

    def add_episode(self, episode: Episode):
        self.episodes.append(episode)

    def save(self, path: str):
        torch.save(self, path + f"/run.pt")

    def rename(self, name: str):
        self.run_name = name
        return self

    def copy(self):
        return copy.deepcopy(self)

    @staticmethod
    def load(path: str):
        return torch.load(path + f"/run.pt", weights_only=False)

    def _get_property(
        self, attr: str, pad_value: float = float("nan")
    ) -> Optional[TensorView]:
        """
        Returns a padded tensor of shape [episodes, steps, ...] wrapped in TensorView.

        Args:
            attr: Result field name (e.g., "balance", "rewards", etc.).
            pad_value: Fill value used for padding.

        Returns:
            TensorView object wrapping [episodes, max_steps, ...]
        """
        episode_tensors = [
            torch.stack(
                [
                    getattr(result, attr)
                    for result in episode.steps
                    if getattr(result, attr) is not None
                ]
            )
            for episode in self.episodes
            if any(getattr(result, attr) is not None for result in episode.steps)
        ]

        if not episode_tensors:
            return None

        max_len = max(tensor.shape[0] for tensor in episode_tensors)
        padded = []

        for tensor in episode_tensors:
            pad_len = max_len - tensor.shape[0]
            if pad_len > 0:
                pad_shape = (pad_len,) + tensor.shape[1:]
                pad_tensor = torch.full(
                    pad_shape, pad_value, dtype=tensor.dtype, device=tensor.device
                )
                tensor = torch.cat([tensor, pad_tensor], dim=0)
            padded.append(tensor)

        return TensorView(torch.stack(padded))  # shape: [episodes, max_len, ...]

    def to_frame(self, attr: str, use_dates: bool = True) -> PlotFrame:
        tensor_view = getattr(self, attr)
        if tensor_view is None:
            raise ValueError(f"No data found for attribute '{attr}'")

        tensor = tensor_view.tensor()
        shape = tensor.shape  # e.g. [episodes, steps, ...]
        ndim = len(shape)

        if ndim < 2:
            raise ValueError(
                f"Expected at least 2D tensor [episodes, steps, ...], got shape {shape}"
            )

        episodes, steps = shape[0], shape[1]
        rest_shape = shape[2:]  # anything beyond [episodes, steps]

        # === Index bestimmen (Datum oder Schritt) ===
        if use_dates and attr != "date" and self.date is not None:
            date_tensor = self.date.tensor().squeeze(-1)
            if date_tensor.shape != (episodes, steps):
                raise ValueError(
                    f"Date shape {date_tensor.shape} does not match tensor shape {shape}"
                )

            date_index = pd.to_datetime(date_tensor[0].cpu().numpy(), unit="s")
            index = date_index
            index_name = "date"
        else:
            index = np.arange(steps)
            index_name = "step"

        # === Daten in [steps, total_columns] umformen ===
        data = tensor.cpu().numpy()  # shape: [episodes, steps, *rest]
        data = data.reshape((episodes, steps, -1))  # flatten trailing dims
        data = data.transpose(1, 2, 0)  # [steps, flat_cols, episodes]
        flat_shape = rest_shape if rest_shape else (1,)
        data = data.reshape(steps, -1)

        # === Create Column-MultiIndex with complete hierarchy: runs -> episode -> property -> agent -> assets ===
        col_tuples = []

        # Always use the complete hierarchy
        level_names = ["run", "episode", "property", "agent", "assets"]

        # Determine the structure based on the attribute
        has_agents = attr in [
            "rewards",
            "actions",
            "actor_balance",
            "cash",
            "asset_holdings",
            "balance",
        ]
        has_assets = attr in ["asset_holdings", "actions"]

        # Generate column tuples with complete hierarchy
        for epi in range(episodes):
            episode_name = f"episode_{epi}"

            if has_agents and has_assets:
                # [episodes, steps, n_agents, n_assets] or [episodes, steps, n_agents, n_assets+1]
                n_agents = flat_shape[0] if flat_shape else 1
                n_assets = flat_shape[1] if len(flat_shape) > 1 else 1

                for agent_idx in range(n_agents):
                    agent_name = f"Agent{agent_idx + 1}"
                    for asset_idx in range(n_assets):
                        col_key = (
                            self.run_name,
                            episode_name,
                            attr,
                            agent_name,
                            asset_idx,
                        )
                        col_tuples.append(col_key)

            elif has_agents:
                # [episodes, steps, n_agents]
                n_agents = flat_shape[0] if flat_shape else 1

                for agent_idx in range(n_agents):
                    agent_name = f"Agent{agent_idx + 1}"
                    col_key = (self.run_name, episode_name, attr, agent_name, "-")
                    col_tuples.append(col_key)

            else:
                # [episodes, steps] - no agents or assets
                col_key = (self.run_name, episode_name, attr, "-", "-")
                col_tuples.append(col_key)

        # Map asset indices to ticker names for asset_holdings and actions
        if (
            attr == "asset_holdings"
            and hasattr(self, "tickers")
            and self.tickers is not None
        ):
            n_assets = len(self.tickers)
            new_col_tuples = []
            for col in col_tuples:
                # col: (run, episode, property, agent, asset_idx)
                *base, asset_idx = col
                if asset_idx == "-":
                    ticker = "-"
                else:
                    ticker = (
                        self.tickers[asset_idx]
                        if asset_idx < n_assets
                        else f"asset_{asset_idx}"
                    )
                new_col_tuples.append((*base, ticker))
            col_tuples = new_col_tuples
            # Keep "assets" as the level name for consistency

        elif (
            attr == "actions" and hasattr(self, "tickers") and self.tickers is not None
        ):
            n_assets = len(self.tickers)
            n_actions = n_assets + 1  # +1 for cash
            new_col_tuples = []
            for col in col_tuples:
                # col: (run, episode, property, agent, asset_idx)
                *base, asset_idx = col
                if asset_idx == "-":
                    label = "-"
                elif asset_idx < n_assets:
                    label = self.tickers[asset_idx]
                elif asset_idx == n_assets:
                    label = "cash"
                else:
                    label = f"action_{asset_idx}"
                new_col_tuples.append((*base, label))
            col_tuples = new_col_tuples
            # Keep "assets" as the level name for consistency

        columns = pd.MultiIndex.from_tuples(col_tuples, names=level_names)
        df = pd.DataFrame(data, index=index, columns=columns)
        df.index.name = index_name

        return PlotFrame(df)

    def reset(self):
        self.episodes = []

    # === Convenience properties ===
    @property
    def balance(self) -> Optional[TensorView]:
        """
        Returns a tensor of shape [episodes, steps] containing the total portfolio value at each step.
        """
        return self._get_property("balance")

    @property
    def rewards(self) -> Optional[TensorView]:
        """
        Returns a tensor of shape [episodes, steps, n_agents] containing the rewards of each agent at each step.
        """
        return self._get_property("rewards")

    @property
    def actions(self) -> Optional[TensorView]:
        """
        Returns a tensor of shape [episodes, steps, n_agents, n_actions] containing the actions of each agent at each step.
        """
        return self._get_property("actions")

    @property
    def asset_holdings(self) -> Optional[TensorView]:
        """
        Returns a tensor of shape [episodes, steps, n_agents, n_assets] containing the asset holdings of each agent at each step.
        """
        return self._get_property("asset_holdings")

    @property
    def actor_balance(self) -> Optional[TensorView]:
        """
        Returns a tensor of shape [episodes, steps, n_agents] containing the portfolio value of each agent at each step.
        """
        return self._get_property("actor_balance")

    @property
    def cash(self) -> Optional[TensorView]:
        """
        Returns a tensor of shape [episodes, steps, n_agents] containing the cash holdings of each agent at each step.
        """
        return self._get_property("cash")

    @property
    def date(self) -> Optional[TensorView]:
        """
        Returns a tensor of shape [episodes, steps] containing the date at each step.
        """
        return self._get_property("date")

    # === Convenience properties ===
    @property
    def balance_df(self) -> Optional[PlotFrame]:
        """
        Returns a tensor of shape [episodes, steps] containing the total portfolio value at each step.
        """
        return self.to_frame("balance")

    @property
    def rewards_df(self) -> Optional[PlotFrame]:
        """
        Returns a tensor of shape [episodes, steps, n_agents] containing the rewards of each agent at each step.
        """
        return self.to_frame("rewards")

    @property
    def actions_df(self) -> Optional[PlotFrame]:
        """
        Returns a tensor of shape [episodes, steps, n_agents, n_actions] containing the actions of each agent at each step.
        """
        return self.to_frame("actions")

    @property
    def asset_holdings_df(self) -> Optional[PlotFrame]:
        """
        Returns a tensor of shape [episodes, steps, n_agents, n_assets] containing the asset holdings of each agent at each step.
        """
        return self.to_frame("asset_holdings")

    @property
    def actor_balance_df(self) -> Optional[PlotFrame]:
        """
        Returns a tensor of shape [episodes, steps, n_agents] containing the portfolio value of each agent at each step.
        """
        return self.to_frame("actor_balance")

    @property
    def cash_df(self) -> Optional[PlotFrame]:
        """
        Returns a tensor of shape [episodes, steps, n_agents] containing the cash holdings of each agent at each step.
        """
        return self.to_frame("cash")

    @property
    def date_df(self) -> Optional[PlotFrame]:
        """
        Returns a tensor of shape [episodes, steps] containing the date at each step.
        """
        return self.to_frame("date")

    def get_frame(self) -> PlotFrame:
        """
        Returns a PlotFrame containing all properties of the run combined.

        Returns:
            PlotFrame with all run data combined into a single DataFrame
        """
        all_frames = []

        # Get all available property frames
        properties = [
            "balance",
            "rewards",
            "actions",
            "asset_holdings",
            "actor_balance",
            "cash",
        ]

        for prop in properties:
            try:
                frame = getattr(self, f"{prop}_df")
                if frame is not None:
                    all_frames.append(frame._df)  # Get the underlying DataFrame
            except Exception as e:
                print(f"Warning: Could not get {prop}_df: {e}")

        if not all_frames:
            raise ValueError("No valid property frames found in run")

        # Concatenate all DataFrames along the columns
        # This preserves the MultiIndex structure
        concatenated = pd.concat(all_frames, axis=1)

        return PlotFrame(concatenated)

    @staticmethod
    def combine_runs(runs: List["Run"]) -> "PlotFrame":
        """
        Combine multiple runs into a single PlotFrame.

        Args:
            runs: List of Run objects to combine

        Returns:
            Combined PlotFrame containing all run data

        Example:
            >>> runs = [run1, run2, run3]
            >>> combined = Run.combine_runs(runs)
        """
        if not runs:
            raise ValueError("No runs provided")

        combined_frame = runs[0].get_frame()
        for run in runs[1:]:
            combined_frame = combined_frame + run.get_frame()

        # Sort the columns by the first level (run level) to ensure consistent plotting order
        if hasattr(combined_frame, "_df") and combined_frame._df is not None:
            if combined_frame._df.columns.nlevels >= 1:
                # Sort by the first level (run level) and handle non-unique MultiIndex
                try:
                    # Try to sort by level 0, but don't reindex to avoid non-unique issues
                    sorted_columns = sorted(combined_frame._df.columns)
                    # Use reindex to reorder columns
                    combined_frame._df = combined_frame._df.reindex(
                        columns=sorted_columns
                    )
                except ValueError:
                    # If that fails, just use the original order
                    pass

        return combined_frame

    @staticmethod
    def discover_runs(runs_dir: str, pattern: str = None) -> Dict[str, List[str]]:
        """
        Automatically discover run directories and group them by agent type.

        Args:
            runs_dir: Path to the runs directory
            pattern: Optional regex pattern to filter runs

        Returns:
            Dictionary mapping agent types to lists of run paths
        """
        runs_dir = Path(runs_dir)
        agent_runs = {}

        # Check if directory exists
        if not runs_dir.exists():
            return agent_runs

        for run_dir in runs_dir.iterdir():
            if not run_dir.is_dir():
                continue

            dir_name = run_dir.name

            # Skip if pattern doesn't match
            if pattern and not re.search(pattern, dir_name):
                continue

            # Extract agent type from directory name
            # Example: "2025-08-18-16-39-10_Agent_ClassicOnePeriodMarkowitzAgent_seed_1_EVAL"
            match = re.search(r"Agent_(\w+)_seed_(\d+)_(\w+)", dir_name)
            if match:
                agent_type = match.group(1)
                seed = match.group(2)
                phase = match.group(3)  # TRAIN, EVAL, etc.

                if agent_type not in agent_runs:
                    agent_runs[agent_type] = []

                agent_runs[agent_type].append(str(run_dir))

        return agent_runs

    @staticmethod
    def load_runs_by_agent(
        runs_dir: str,
        agent_types: List[str] = None,
        phases: List[str] = None,
        exclude_phases: List[str] = None,
        pattern: str = None,
        remove_seed_from_name: bool = True,
    ) -> Dict[str, List["Run"]]:
        """
        Load runs grouped by agent type with automatic naming.

        Args:
            runs_dir: Path to the runs directory
            agent_types: List of agent types to load (if None, loads all)
            phases: List of phases to include (e.g., ['EVAL'], ['TRAIN', 'EVAL'])
            exclude_phases: List of phases to exclude (e.g., ['TRAIN_EVAL'])
            pattern: Optional regex pattern to filter runs
            remove_seed_from_name: Whether to remove seed information from run names

        Returns:
            Dictionary mapping agent types to lists of loaded runs
        """
        discovered_runs = Run.discover_runs(runs_dir, pattern)
        loaded_runs = {}

        for agent_type, run_paths in discovered_runs.items():
            if agent_types and agent_type not in agent_types:
                continue

            agent_runs = []

            for run_path in run_paths:
                # Check if this run matches the desired phases
                if phases:
                    phase_match = any(phase in run_path for phase in phases)
                    if not phase_match:
                        continue

                # Check if this run should be excluded
                if exclude_phases:
                    exclude_match = any(
                        exclude_phase in run_path for exclude_phase in exclude_phases
                    )
                    if exclude_match:
                        continue

                try:
                    run = Run.load(run_path)

                    # Create naming based on configuration
                    # Map agent types to display names
                    display_names = {
                        "ClassicOnePeriodMarkowitzAgent": "Tangency",
                        "OneOverNPortfolioAgent": "1/N",
                        "HJBPortfolioAgent": "HJB",
                        "HJBPortfolioAgentWithCosts": "HJB with Costs",
                        "SACAgent": "SAC",
                        "PPOAgent": "PPO",
                        "DDPGAgent": "DDPG",
                        "A2CAgent": "A2C",
                        "TD3Agent": "TD3",
                        "MADDPGAgent": "MADDPG",
                        "DQNAgent": "DQN",
                        "SPQLAgent": "SPQL",
                        "RandomAgent": "Random",
                    }

                    if remove_seed_from_name:
                        # Use short display name without seed
                        short_name = display_names.get(agent_type, agent_type)
                    else:
                        # Keep original name with seed information
                        # Extract seed from directory name for naming
                        seed_match = re.search(r"_seed_(\d+)_", run_path)
                        if seed_match:
                            seed = seed_match.group(1)
                            short_name = f"{display_names.get(agent_type, agent_type)}_seed_{seed}"
                        else:
                            short_name = display_names.get(agent_type, agent_type)

                    run.rename(short_name)
                    agent_runs.append(run)

                except Exception as e:
                    print(f"Failed to load run {run_path}: {e}")
                    continue

            if agent_runs:
                loaded_runs[agent_type] = agent_runs

        return loaded_runs
