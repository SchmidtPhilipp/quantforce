from typing import Any, Dict, List, Optional

import pandas as pd

from qf.utils.plot.plot_config import PlotConfig
from qf.utils.plot.plot_manager import PlotManager


class PlotFrame:
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def __getitem__(self, key):
        result = self._df[key]
        return PlotFrame(result) if isinstance(result, pd.DataFrame) else result

    def __getattr__(self, name):
        attr = getattr(self._df, name)
        if callable(attr):

            def wrapper(*args, **kwargs):
                result = attr(*args, **kwargs)
                return PlotFrame(result) if isinstance(result, pd.DataFrame) else result

            return wrapper
        else:
            return attr

    def __repr__(self):
        return repr(self._df)

    def __add__(self, other):
        if not isinstance(other, PlotFrame):
            return NotImplemented
        # Concatenate along columns (axis=1)
        concatenated = pd.concat([self._df, other._df], axis=1)
        return PlotFrame(concatenated)

    def __len__(self):
        return self._df.__len__()

    # ============================================================================
    # Helper Methods for Data Processing
    # ============================================================================

    def _normalize_levels(self, levels: int | str | list[int | str]) -> List[int]:
        """Normalize levels to list of integers."""
        if not isinstance(self._df.columns, pd.MultiIndex):
            raise ValueError("MultiIndex columns required")

        if isinstance(levels, (int, str)):
            levels = [levels]

        level_names = self._df.columns.names
        level_indices = []

        for lvl in levels:
            if isinstance(lvl, int):
                level_indices.append(lvl)
            elif isinstance(lvl, str):
                if lvl not in level_names:
                    raise ValueError(
                        f"Level name '{lvl}' not found in column MultiIndex names {level_names}"
                    )
                level_indices.append(level_names.index(lvl))
            else:
                raise TypeError(f"Level must be int or str, got {type(lvl)}")

        return level_indices

    def _get_remaining_levels(self, levels_to_remove: List[int]) -> List[int]:
        """Get remaining levels after removing specified levels."""
        all_levels = list(range(len(self._df.columns.names)))
        return [i for i in all_levels if i not in levels_to_remove]

    def _get_filtered_values(
        self, level_name: str, exclude_values: List[str] = None
    ) -> List[str]:
        """Get unique values from a level, excluding specified values."""
        if exclude_values is None:
            exclude_values = ["-"]
        return [
            v
            for v in self.columns.get_level_values(level_name).unique()
            if v not in exclude_values
        ]

    def _extract_property_data(
        self,
        property_name: str,
        reductions: List[str] = None,
        agent: str = None,
        asset: str = None,
    ) -> pd.DataFrame:
        """Extract data for a specific property with optional filtering."""
        if reductions is None:
            reductions = ["episode"]

        data = self.mean(levels=reductions)

        # Extract property data
        data = data.xs(property_name, level="property", axis=1)

        # Apply agent filter if specified
        if agent is not None:
            data = data.xs(agent, level="agent", axis=1)

        # Apply asset filter if specified
        if asset is not None:
            data = data.xs(asset, level="assets", axis=1)
        elif property_name not in ["balance", "rewards"]:
            # Drop assets level for properties that don't need it
            data = data.droplevel(["assets"], axis=1)

        # Drop agent and assets levels for balance
        if property_name == "balance":
            data = data.droplevel(["agent", "assets"], axis=1)

        # Sort columns to maintain consistent order after xs operations
        try:
            if data.columns.nlevels >= 1:
                # Convert to list, sort, and reindex
                sorted_columns = sorted(data.columns)
                data = data.reindex(columns=sorted_columns)
        except (ValueError, TypeError):
            # If sorting fails, keep original order
            pass

        return data

    def _get_plot_config(self, plot_config: Optional[PlotConfig] = None) -> PlotConfig:
        """Get plot configuration with default fallback."""
        return plot_config or PlotConfig()

    def _get_plot_labels(self, property_name: str) -> Dict[str, str]:
        """Get standardized plot labels for a property."""
        labels = {
            "balance": {"ylabel": "Portfolio Value (USD)", "filename": "balance"},
            "cash": {"ylabel": "Cash (USD)", "filename": "cash"},
            "actions": {"ylabel": "Actions", "filename": "actions"},
            "asset_holdings": {
                "ylabel": "Share Count (1)",
                "filename": "asset_holdings",
            },
            "rewards": {"ylabel": "Reward", "filename": "rewards"},
            "cumulative_rewards": {
                "ylabel": "Cumulative Reward",
                "filename": "cumulative_rewards",
            },
        }
        return labels.get(
            property_name,
            {
                "ylabel": f'{property_name.replace("_", " ").title()}',
                "filename": property_name,
            },
        )

    # ============================================================================
    # Statistical Methods
    # ============================================================================

    def mean(self, levels: int | str | list[int | str]):
        """
        Computes the mean over one or more MultiIndex column levels, using either names or indices.

        Parameters:
            levels (int, str, or list): Column level(s) to reduce by averaging.

        Returns:
            PlotFrame: A new PlotFrame with mean over the specified level(s).
        """
        level_indices = self._normalize_levels(levels)
        remaining_levels = self._get_remaining_levels(level_indices)
        df_mean = self._df.T.groupby(level=remaining_levels).mean().T

        # Sort columns to maintain consistent order after groupby operation
        try:
            if df_mean.columns.nlevels >= 1:
                # Convert to list, sort, and reindex
                sorted_columns = sorted(df_mean.columns)
                df_mean = df_mean.reindex(columns=sorted_columns)
        except (ValueError, TypeError):
            # If sorting fails, keep original order
            pass

        return PlotFrame(df_mean)

    def std(self, levels: int | str | list[int | str]):
        """
        Computes the standard deviation over one or more MultiIndex column levels, using either names or indices.

        Parameters:
            levels (int, str, or list): Column level(s) to reduce by computing std.

        Returns:
            PlotFrame: A new PlotFrame with std over the specified level(s).
        """
        level_indices = self._normalize_levels(levels)
        remaining_levels = self._get_remaining_levels(level_indices)
        df_std = self._df.T.groupby(level=remaining_levels).std().T

        # Sort columns to maintain consistent order after groupby operation
        try:
            if df_std.columns.nlevels >= 1:
                # Convert to list, sort, and reindex
                sorted_columns = sorted(df_std.columns)
                df_std = df_std.reindex(columns=sorted_columns)
        except (ValueError, TypeError):
            # If sorting fails, keep original order
            pass

        return PlotFrame(df_std)

    # ============================================================================
    # Plotting Methods
    # ============================================================================

    def plot_balance(self, plot_config: Optional[PlotConfig] = None, **kwargs):
        """Plot balance using PlotManager."""
        data = self._extract_property_data("balance")
        labels = self._get_plot_labels("balance")

        plot_manager = PlotManager(self._get_plot_config(plot_config))
        plot_manager.plot_lines(
            data,
            filename=labels["filename"],
            ylabel=labels["ylabel"],
            **kwargs,
        )

    def plot_cash(self, plot_config: Optional[PlotConfig] = None, **kwargs):
        """Plot cash for each agent using PlotManager."""
        agents = self._get_filtered_values("agent")
        labels = self._get_plot_labels("cash")

        plot_manager = PlotManager(self._get_plot_config(plot_config))

        for agent in agents:
            data = self._extract_property_data("cash", agent=agent)
            plot_manager.plot_lines(
                data,
                filename=f"{labels['filename']}_{agent}",
                ylabel=labels["ylabel"],
                **kwargs,
            )

    def plot_actions(self, plot_config: Optional[PlotConfig] = None, **kwargs):
        """Plot actions for each agent and asset using PlotManager."""
        agents = self._get_filtered_values("agent")
        assets = self._get_filtered_values("assets")
        labels = self._get_plot_labels("actions")

        plot_manager = PlotManager(self._get_plot_config(plot_config))

        for agent in agents:
            for asset in assets:
                data = self._extract_property_data("actions", agent=agent, asset=asset)
                plot_manager.plot_lines(
                    data,
                    filename=f"{labels['filename']}_{agent}_{asset}",
                    ylabel=labels["ylabel"],
                    **kwargs,
                )

    def plot_asset_holdings(self, plot_config: Optional[PlotConfig] = None, **kwargs):
        """Plot asset holdings for each agent and asset using PlotManager."""
        agents = self._get_filtered_values("agent")
        assets = self._get_filtered_values("assets", exclude_values=["-", "cash"])
        labels = self._get_plot_labels("asset_holdings")

        plot_manager = PlotManager(self._get_plot_config(plot_config))

        for agent in agents:
            for asset in assets:
                data = self._extract_property_data(
                    "asset_holdings", agent=agent, asset=asset
                )
                plot_manager.plot_lines(
                    data,
                    filename=f"{labels['filename']}_{agent}_{asset}",
                    ylabel=labels["ylabel"],
                    **kwargs,
                )

    def plot_rewards(self, plot_config: Optional[PlotConfig] = None, **kwargs):
        """Plot rewards for each agent using PlotManager."""
        agents = self._get_filtered_values("agent")
        labels = self._get_plot_labels("rewards")

        plot_manager = PlotManager(self._get_plot_config(plot_config))

        for agent in agents:
            data = self._extract_property_data("rewards", agent=agent)
            plot_manager.plot_lines(
                data,
                filename=f"{labels['filename']}_{agent}",
                ylabel=labels["ylabel"],
                **kwargs,
            )

    def plot_cumulative_rewards(
        self, plot_config: Optional[PlotConfig] = None, **kwargs
    ):
        """Plot cumulative rewards for each agent using PlotManager."""
        agents = self._get_filtered_values("agent")
        labels = self._get_plot_labels("cumulative_rewards")

        plot_manager = PlotManager(self._get_plot_config(plot_config))

        for agent in agents:
            data = self._extract_property_data("rewards", agent=agent)
            # Calculate cumulative rewards
            cumulative_data = data.cumsum()
            plot_manager.plot_lines(
                cumulative_data,
                filename=f"{labels['filename']}_{agent}",
                ylabel=labels["ylabel"],
                **kwargs,
            )

    def plot_property(
        self,
        property_name: str,
        reductions: List[str] = None,
        agents: List[str] = None,
        assets: List[str] = None,
        plot_config: Optional[PlotConfig] = None,
    ):
        """
        Generalized plotting function for any property with flexible reductions.

        Args:
            property_name (str): The property to plot (e.g., 'balance', 'cash', 'actions', 'asset_holdings', 'rewards')
            reductions (list): List of levels to reduce by averaging (e.g., ['episode', 'run'])
            agents (list): List of agents to plot. If None, plots all agents except "-"
            assets (list): List of assets to plot. If None, plots all assets except "-" and "cash"
            plot_config (Optional[PlotConfig]): Optional plotting configuration
        """
        plot_manager = PlotManager(self._get_plot_config(plot_config))

        # Get available agents and assets
        available_agents = self._get_filtered_values("agent")
        available_assets = self._get_filtered_values("assets")

        # Use provided lists or defaults
        if agents is None:
            agents = available_agents
        if assets is None:
            # For properties that don't need assets, use empty list
            if property_name in ["balance", "rewards"]:
                assets = []
            else:
                assets = [a for a in available_assets if a != "cash"]

        # Prepare all plot data in a single DataFrame
        all_plot_data = pd.DataFrame()

        # Handle different property types
        if property_name == "balance":
            # Balance is global, no agent/asset iteration needed
            all_plot_data = self._extract_property_data(property_name, reductions)

        elif property_name in ["cash", "rewards"]:
            # Cash and rewards need agent iteration
            for agent in agents:
                if agent not in available_agents:
                    continue
                plot_data = self._extract_property_data(
                    property_name, reductions, agent=agent
                )
                all_plot_data[f"{agent}"] = plot_data

        elif property_name in ["actions", "asset_holdings"]:
            # Actions and asset holdings need agent and asset iteration
            for agent in agents:
                if agent not in available_agents:
                    continue
                for asset in assets:
                    if asset not in available_assets:
                        continue
                    # Skip cash for asset_holdings
                    if property_name == "asset_holdings" and asset == "cash":
                        continue

                    plot_data = self._extract_property_data(
                        property_name, reductions, agent=agent, asset=asset
                    )
                    all_plot_data[f"{agent}_{asset}"] = plot_data
        else:
            raise ValueError(
                f"Unsupported property: {property_name}. Supported properties: ['balance', 'cash', 'actions', 'asset_holdings', 'rewards']"
            )

        # Single plot call with all data
        if not all_plot_data.empty:
            # Sort columns to ensure consistent plotting order
            try:
                # Convert to list, sort, and reindex
                sorted_columns = sorted(all_plot_data.columns)
                all_plot_data = all_plot_data.reindex(columns=sorted_columns)
            except (ValueError, TypeError):
                # If sorting fails, keep original order
                pass

            labels = self._get_plot_labels(property_name)

            # Determine appropriate labels and title
            if property_name == "balance":
                ylabel = f"{property_name.title()} (USD)"
                filename = property_name
            elif property_name in ["cash", "rewards"]:
                ylabel = f"{property_name.title()}{' (USD)' if property_name == 'cash' else ''}"
                filename = f"{property_name}_all_agents"
            else:  # actions, asset_holdings
                ylabel = f"{property_name.replace('_', ' ').title()}"
                filename = f"{property_name}_all_combinations"

            plot_manager.plot_lines(
                all_plot_data,
                filename=filename,
                ylabel=ylabel,
            )

    # ============================================================================
    # Utility Methods
    # ============================================================================

    def rename_run_name(self, name: str):
        """Rename the run name of the PlotFrame."""
        self._df.columns = self._df.columns.set_levels(
            [name] * len(self._df.columns.levels[0]), level=0
        )
        return self

    # ============================================================================
    # Confidence Plotting Methods
    # ============================================================================

    @staticmethod
    def _prepare_confidence_data(
        frames: List["PlotFrame"],
        mean_of_level: str,
        property_name: str,
        agent: str = None,
        asset: str = None,
    ) -> tuple:
        """Prepare mean and std data for confidence plotting."""
        mean_frame = pd.DataFrame()
        std_frame = pd.DataFrame()

        for i, frame in enumerate(frames):
            # Extract data based on property type
            if property_name == "balance":
                mean_series = (
                    frame.mean(levels=mean_of_level)
                    .xs(property_name, level="property", axis=1)
                    .droplevel(["agent", "assets"], axis=1)
                )
                std_series = (
                    frame.std(levels=mean_of_level)
                    .xs(property_name, level="property", axis=1)
                    .droplevel(["agent", "assets"], axis=1)
                )
            elif property_name in ["cash", "rewards"]:
                mean_series = (
                    frame.mean(levels=mean_of_level)
                    .xs(property_name, level="property", axis=1)
                    .xs(agent, level="agent", axis=1)
                    .droplevel(["assets"], axis=1)
                )
                std_series = (
                    frame.std(levels=mean_of_level)
                    .xs(property_name, level="property", axis=1)
                    .xs(agent, level="agent", axis=1)
                    .droplevel(["assets"], axis=1)
                )
            else:  # actions, asset_holdings
                mean_series = (
                    frame.mean(levels=mean_of_level)
                    .xs(property_name, level="property", axis=1)
                    .xs(agent, level="agent", axis=1)
                    .xs(asset, level="assets", axis=1)
                )
                std_series = (
                    frame.std(levels=mean_of_level)
                    .xs(property_name, level="property", axis=1)
                    .xs(agent, level="agent", axis=1)
                    .xs(asset, level="assets", axis=1)
                )

            # Add to plot data
            run_name = frame.columns.get_level_values("run")[i]
            mean_frame.index = mean_series.index
            mean_frame[run_name] = mean_series.values
            std_frame.index = std_series.index
            std_frame[run_name] = std_series.values

        # Sort columns to ensure consistent plotting order
        try:
            # Convert to list, sort, and reindex
            sorted_columns = sorted(mean_frame.columns)
            mean_frame = mean_frame.reindex(columns=sorted_columns)
            std_frame = std_frame.reindex(columns=sorted_columns)
        except (ValueError, TypeError):
            # If sorting fails, keep original order
            pass

        return mean_frame, std_frame

    @staticmethod
    def _plot_confidence_intervals(
        mean_frame: pd.DataFrame,
        std_frame: pd.DataFrame,
        sigma: int,
        ylabel: str,
        filename: str,
        plot_config: Optional[PlotConfig] = None,
    ):
        """Plot confidence intervals using PlotManager."""
        plot_manager = PlotManager(plot_config or PlotConfig())
        plot_manager.plot_confidence_intervals(
            mean_frame=mean_frame,
            std_frame=std_frame,
            sigma=sigma,
            ylabel=ylabel,
            xlabel="Date",
            filename=filename,
        )

    @staticmethod
    def plot_confidence_balance(
        frames: List["PlotFrame"],
        mean_of_level: str,
        sigma: int = 1,
        plot_config: Optional[PlotConfig] = None,
        filename: str = "balance_confidence",
    ):
        """
        Plots the mean and ±sigma standard deviation regions for balance.

        Args:
            frames (list["PlotFrame"]): List of PlotFrames from different seeded runs
            mean_of_level (str): The level name over which the mean and std are calculated
            sigma (int): Number of standard deviations for the confidence interval
            plot_config (Optional[PlotConfig]): Optional plotting configuration
        """
        mean_frame, std_frame = PlotFrame._prepare_confidence_data(
            frames, mean_of_level, "balance"
        )
        PlotFrame._plot_confidence_intervals(
            mean_frame, std_frame, sigma, "Balance (USD)", filename, plot_config
        )

    @staticmethod
    def plot_confidence_cash(
        frames: List["PlotFrame"],
        mean_of_level: str,
        sigma: int = 1,
        plot_config: Optional[PlotConfig] = None,
    ):
        """
        Plots the mean and ±sigma standard deviation regions for cash for each agent.

        Args:
            frames (list["PlotFrame"]): List of PlotFrames from different seeded runs
            mean_of_level (str): The level name over which the mean and std are calculated
            sigma (int): Number of standard deviations for the confidence interval
            plot_config (Optional[PlotConfig]): Optional plotting configuration
        """
        agents = frames[0].columns.get_level_values("agent").unique()

        for agent in agents:
            if agent == "-":
                continue

            mean_frame, std_frame = PlotFrame._prepare_confidence_data(
                frames, mean_of_level, "cash", agent=agent
            )
            PlotFrame._plot_confidence_intervals(
                mean_frame,
                std_frame,
                sigma,
                "Cash (USD)",
                f"cash_confidence_{agent}",
                plot_config,
            )

    @staticmethod
    def plot_confidence_actions(
        frames: List["PlotFrame"],
        mean_of_level: str,
        sigma: int = 1,
        plot_config: Optional[PlotConfig] = None,
    ):
        """
        Plots the mean and ±sigma standard deviation regions for actions for each agent and asset.

        Args:
            frames (list["PlotFrame"]): List of PlotFrames from different seeded runs
            mean_of_level (str): The level name over which the mean and std are calculated
            sigma (int): Number of standard deviations for the confidence interval
            plot_config (Optional[PlotConfig]): Optional plotting configuration
        """
        agents = [
            a for a in frames[0].columns.get_level_values("agent").unique() if a != "-"
        ]
        assets = [
            a for a in frames[0].columns.get_level_values("assets").unique() if a != "-"
        ]

        for agent in agents:
            for asset in assets:
                mean_frame, std_frame = PlotFrame._prepare_confidence_data(
                    frames, mean_of_level, "actions", agent=agent, asset=asset
                )
                PlotFrame._plot_confidence_intervals(
                    mean_frame,
                    std_frame,
                    sigma,
                    "Actions",
                    f"actions_confidence_{agent}_{asset}",
                    plot_config,
                )

    @staticmethod
    def plot_confidence_asset_holdings(
        frames: List["PlotFrame"],
        mean_of_level: str,
        sigma: int = 3,
        plot_config: Optional[PlotConfig] = None,
    ):
        """
        Plots the mean and ±sigma standard deviation regions for asset holdings for each agent and asset.

        Args:
            frames (list["PlotFrame"]): List of PlotFrames from different seeded runs
            mean_of_level (str): The level name over which the mean and std are calculated
            sigma (int): Number of standard deviations for the confidence interval
            plot_config (Optional[PlotConfig]): Optional plotting configuration
        """
        agents = [
            a for a in frames[0].columns.get_level_values("agent").unique() if a != "-"
        ]
        assets = [
            a
            for a in frames[0].columns.get_level_values("assets").unique()
            if (a != "-") and (a != "cash")
        ]

        for agent in agents:
            for asset in assets:
                mean_frame, std_frame = PlotFrame._prepare_confidence_data(
                    frames, mean_of_level, "asset_holdings", agent=agent, asset=asset
                )
                PlotFrame._plot_confidence_intervals(
                    mean_frame,
                    std_frame,
                    sigma,
                    "Asset Holdings",
                    f"asset_holdings_confidence_{agent}_{asset}",
                    plot_config,
                )

    @staticmethod
    def plot_confidence_rewards(
        frames: List["PlotFrame"],
        mean_of_level: str,
        sigma: int = 1,
        plot_config: Optional[PlotConfig] = None,
    ):
        """
        Plots the mean and ±sigma standard deviation regions for rewards for each agent.

        Args:
            frames (list["PlotFrame"]): List of PlotFrames from different seeded runs
            mean_of_level (str): The level name over which the mean and std are calculated
            sigma (int): Number of standard deviations for the confidence interval
            plot_config (Optional[PlotConfig]): Optional plotting configuration
        """
        agents = [
            a for a in frames[0].columns.get_level_values("agent").unique() if a != "-"
        ]

        for agent in agents:
            mean_frame, std_frame = PlotFrame._prepare_confidence_data(
                frames, mean_of_level, "rewards", agent=agent
            )
            PlotFrame._plot_confidence_intervals(
                mean_frame,
                std_frame,
                sigma,
                "Reward",
                f"rewards_confidence_{agent}",
                plot_config,
            )

    @staticmethod
    def plot_confidence_cumulative_rewards(
        frames: List["PlotFrame"],
        mean_of_level: str,
        sigma: int = 1,
        plot_config: Optional[PlotConfig] = None,
    ):
        """
        Plots the mean and ±sigma standard deviation regions for cumulative rewards for each agent.
        Calculates cumulative rewards from the normal rewards data.

        Args:
            frames (list["PlotFrame"]): List of PlotFrames from different seeded runs
            mean_of_level (str): The level name over which the mean and std are calculated
            sigma (int): Number of standard deviations for the confidence interval
            plot_config (Optional[PlotConfig]): Optional plotting configuration
        """
        agents = [
            a for a in frames[0].columns.get_level_values("agent").unique() if a != "-"
        ]

        for agent in agents:
            # First get the mean and std data like plot_confidence_rewards does
            mean_frame, std_frame = PlotFrame._prepare_confidence_data(
                frames, mean_of_level, "rewards", agent=agent
            )

            # Then apply cumsum to convert to cumulative rewards
            mean_frame = mean_frame.cumsum()
            std_frame = std_frame.cumsum()

            PlotFrame._plot_confidence_intervals(
                mean_frame,
                std_frame,
                sigma,
                "Cumulative Reward",
                f"cumulative_rewards_confidence_{agent}",
                plot_config,
            )

    # Keep the original function name as an alias for backward compatibility
    @staticmethod
    def plot_confidence(
        frames: List["PlotFrame"],
        mean_of_level: str,
        attribute: str,
        sigma: int = 1,
        plot_config: Optional[PlotConfig] = None,
    ):
        """
        Plots the mean and ±sigma standard deviation regions for multi-index DataFrames.

        Args:
            frames (list[pd.DataFrame]): List of MultiIndex DataFrames.
            mean_of_level (str): The level name over which the mean and std are calculated.
            attribute (str): The attribute/property to plot (e.g., 'balance').
            sigma (int): Number of standard deviations for the confidence interval.
            plot_config (Optional[PlotConfig]): Optional plotting configuration.
        """
        if attribute == "balance":
            PlotFrame.plot_confidence_balance(frames, mean_of_level, sigma, plot_config)
        elif attribute == "cash":
            PlotFrame.plot_confidence_cash(frames, mean_of_level, sigma, plot_config)
        elif attribute == "actions":
            PlotFrame.plot_confidence_actions(frames, mean_of_level, sigma, plot_config)
        elif attribute == "asset_holdings":
            PlotFrame.plot_confidence_asset_holdings(
                frames, mean_of_level, sigma, plot_config
            )
        elif attribute == "rewards":
            PlotFrame.plot_confidence_rewards(frames, mean_of_level, sigma, plot_config)
        elif attribute == "cumulative_rewards":
            PlotFrame.plot_confidence_cumulative_rewards(
                frames, mean_of_level, sigma, plot_config
            )
        else:
            raise ValueError(
                f"Unsupported attribute: {attribute}. Supported attributes: ['balance', 'cash', 'actions', 'asset_holdings', 'rewards', 'cumulative_rewards']"
            )

    # ============================================================================
    # LaTeX Table Generation Methods
    # ============================================================================

    @staticmethod
    def dataframe_to_latex(
        df: pd.DataFrame,
        title: str = None,
        caption: str = None,
        float_format: str = "%.3f",
        escape_special_chars: bool = True,
    ) -> str:
        """
        Convert a pandas DataFrame to a LaTeX table.

        Args:
            df (pd.DataFrame): DataFrame to convert
            title (str, optional): Table title for caption (ignored - kept for compatibility)
            caption (str, optional): Custom caption text (ignored - kept for compatibility)
            float_format (str): Format string for float values
            escape_special_chars (bool): Whether to escape LaTeX special characters

        Returns:
            str: LaTeX tabular string (without table environment)
        """
        if df.empty:
            return "\\begin{tabular}{lr}\n\\hline\n\\textbf{No Data} & \\\\\n\\hline\n\\end{tabular}"

        # Convert DataFrame to LaTeX using pandas - only tabular content
        latex_content = df.to_latex(
            float_format=float_format,
            escape=escape_special_chars,
            index=True,
            bold_rows=True,
            caption=None,  # No caption
        )

        # Remove table environment if present and return only tabular content
        if latex_content.startswith("\\begin{table}"):
            # Extract only the tabular part
            start_idx = latex_content.find("\\begin{tabular}")
            end_idx = latex_content.find("\\end{tabular}") + len("\\end{tabular}")
            latex_content = latex_content[start_idx:end_idx]

        return latex_content

    @staticmethod
    def dict_to_latex_table(
        data: Dict[str, Any], title: str = None, float_format: str = "%.3f"
    ) -> str:
        """
        Convert a dictionary to a LaTeX table.

        Args:
            data (Dict[str, Any]): Dictionary to convert
            title (str, optional): Table title (ignored - kept for compatibility)
            float_format (str): Format string for float values

        Returns:
            str: LaTeX tabular string (without table environment)
        """
        if not data:
            return "\\begin{tabular}{lr}\n\\hline\n\\textbf{No Data} & \\\\\n\\hline\n\\end{tabular}"

        # Convert dict to DataFrame
        df = pd.DataFrame(list(data.items()), columns=["Metric", "Value"])

        # Format numeric values
        def format_value(val):
            if isinstance(val, (int, float)):
                return float_format % val
            return str(val)

        df["Value"] = df["Value"].apply(format_value)

        return PlotFrame.dataframe_to_latex(df, float_format=float_format)

    @staticmethod
    def dict_of_dicts_to_latex_table(
        data: Dict[str, Dict[str, Any]],
        title: str = None,
        float_format: str = "%.3f",
        tranposed: bool = False,
    ) -> str:
        """
        Convert a dictionary of dictionaries to a LaTeX comparison table.

        Args:
            data (Dict[str, Dict[str, Any]]): Dictionary mapping keys to metric dictionaries
            title (str, optional): Table title (ignored - kept for compatibility)
            float_format (str): Format string for float values

        Returns:
            str: LaTeX tabular string (without table environment)
        """
        if not data:
            return "\\begin{tabular}{lr}\n\\hline\n\\textbf{No Data} & \\\\\n\\hline\n\\end{tabular}"

        # Convert to DataFrame
        if tranposed:
            df = pd.DataFrame(data).T  # Transpose so keys become index
        else:
            df = pd.DataFrame(data)

        # Format numeric values
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: float_format % x if isinstance(x, (int, float)) else str(x)
            )

        return PlotFrame.dataframe_to_latex(df, float_format=float_format)

    # ============================================================================
    # Metrics Table Methods (Simplified)
    # ============================================================================

    def metrics_table(
        self, periods_per_year: int = 252, risk_free_rate: float = 0.0
    ) -> str:
        """
        Generate a LaTeX table with metrics calculated from the balance data.

        Args:
            periods_per_year (int): Number of periods per year for annualization
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation

        Returns:
            str: LaTeX table string
        """
        import torch

        from qf.utils.metrics import Metrics

        # Extract balance data
        try:
            balance_data = (
                self.mean(levels="episode")
                .xs("balance", level="property", axis=1)
                .droplevel(["agent", "assets"], axis=1)
            )
        except (KeyError, ValueError):
            raise ValueError("No balance data found in PlotFrame")

        # Convert to torch tensor for metrics calculation
        balances = torch.tensor(balance_data.values.flatten(), dtype=torch.float32)

        # Calculate metrics
        metrics = Metrics(
            periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
        )
        metrics.append(balances)

        # Get formatted metrics
        formatted_metrics = metrics.formated(std_dev=False)

        # Generate LaTeX table using the general method
        return PlotFrame.dict_to_latex_table(formatted_metrics)

    @staticmethod
    def metrics_table_comparison(
        frame: "PlotFrame",
        periods_per_year: int = 252,
        risk_free_rate: float = 0.0,
        tranposed: bool = False,
    ) -> str:
        """
        Generate a LaTeX table comparing metrics across runs within a single PlotFrame.

        Args:
            frame (PlotFrame): Single PlotFrame containing multiple runs
            periods_per_year (int): Number of periods per year for annualization
            risk_free_rate (float): Risk-free rate for Sharpe ratio calculation
            tranposed (bool): Whether to transpose the resulting table

        Returns:
            str: LaTeX tabular string (without table environment)
        """
        import torch

        from qf.utils.metrics import Metrics

        if frame is None:
            raise ValueError("No frame provided")

        # Extract run names from the frame
        run_names = frame.columns.get_level_values("run").unique()
        if len(run_names) == 0:
            raise ValueError("No runs found in the frame")

        # Calculate metrics for each run
        all_metrics = {}
        for run_name in run_names:
            try:
                # Extract balance data for this specific run
                balance_data = (
                    frame.xs(run_name, level="run", axis=1)
                    .mean(levels="episode")
                    .xs("balance", level="property", axis=1)
                )

                # Try to drop levels if they exist, otherwise keep the data as is
                try:
                    balance_data = balance_data.droplevel(["agent", "assets"], axis=1)
                except ValueError:
                    # If we can't drop both levels, try dropping them one by one
                    try:
                        balance_data = balance_data.droplevel("agent", axis=1)
                    except ValueError:
                        pass
                    try:
                        balance_data = balance_data.droplevel("assets", axis=1)
                    except ValueError:
                        pass

                # Convert to torch tensor for metrics calculation
                balances = torch.tensor(
                    balance_data.values.flatten(), dtype=torch.float32
                )

                # Calculate metrics
                metrics = Metrics(
                    periods_per_year=periods_per_year, risk_free_rate=risk_free_rate
                )
                metrics.append(balances)

                # Get formatted metrics
                formatted_metrics = metrics.formated(std_dev=False)
                all_metrics[run_name] = formatted_metrics

            except (KeyError, ValueError) as e:
                print(f"Warning: Could not calculate metrics for {run_name}: {e}")
                all_metrics[run_name] = {}

        # Generate comparison LaTeX table using the general method
        latex_table = PlotFrame.dict_of_dicts_to_latex_table(
            all_metrics, tranposed=tranposed
        )

        # Save to file
        with open("metrics_table.tex", "w") as f:
            f.write(latex_table)

        return latex_table

    # ============================================================================
    # Legacy Methods (for backward compatibility)
    # ============================================================================

    @staticmethod
    def _generate_latex_table(metrics: Dict[str, Any]) -> str:
        """
        Legacy method for backward compatibility.
        Use dict_to_latex_table() instead.
        """
        return PlotFrame.dict_to_latex_table(metrics)

    @staticmethod
    def _generate_comparison_latex_table(
        metrics_dict: Dict[str, Dict[str, Any]],
    ) -> str:
        """
        Legacy method for backward compatibility.
        Use dict_of_dicts_to_latex_table() instead.
        """
        return PlotFrame.dict_of_dicts_to_latex_table(metrics_dict)

    # ============================================================================
    # Utility Methods for Run and Frame Management
    # ============================================================================

    @staticmethod
    def combine_frames(frames: List["PlotFrame"]) -> "PlotFrame":
        """
        Combine multiple PlotFrames into a single PlotFrame.

        Args:
            frames: List of PlotFrame objects to combine

        Returns:
            Combined PlotFrame containing all frame data

        Example:
            >>> frames = [frame1, frame2, frame3]
            >>> combined = PlotFrame.combine_frames(frames)
        """
        if not frames:
            raise ValueError("No frames provided")

        combined_frame = frames[0]
        for frame in frames[1:]:
            combined_frame = combined_frame + frame

        return combined_frame
