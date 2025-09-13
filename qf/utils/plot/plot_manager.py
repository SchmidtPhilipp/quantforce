import os
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from qf.utils.plot.plot_config import MatplotlibConfig, PlotConfig


class PlotManager:
    def __init__(self, config: Optional[PlotConfig] = None):
        if config is None:
            config = PlotConfig()
        self.config = config

        # Check if we're using PGF or Jupyter configuration
        if self.config.matplotlib.pgf_enabled:
            self.setup_pgf()
        else:
            self.setup_jupyter()

        os.makedirs(self.config.save.save_dir, exist_ok=True)

    def setup_jupyter(self):
        """Set up matplotlib for Jupyter notebook display."""
        # Use default backend for Jupyter
        # matplotlib.use("default")  # Use Agg backend for better compatibility

        # Get rcParams from Jupyter configuration
        rc_params = self.config.matplotlib.get_jupyter_rc_params()

        # Update rcParams with Jupyter-optimized settings
        plt.rcParams.update(rc_params)

        # Enable interactive mode for Jupyter
        plt.ion()

    def setup_default(self):
        """Set up default matplotlib configuration."""
        # Use default backend
        matplotlib.use("Agg")

        # Get rcParams from configuration
        rc_params = self.config.matplotlib.get_rc_params()

        # Update rcParams with settings
        plt.rcParams.update(rc_params)

    def setup_pgf(self):
        """Set up PGF backend for matplotlib to export figures in LaTeX-compatible format."""
        matplotlib.use("pgf")

        # Get rcParams from configuration
        rc_params = self.config.matplotlib.get_rc_params()
        pgf_rc_params = self.config.matplotlib.get_pgf_rc_params()

        # Update rcParams with all settings
        plt.rcParams.update(rc_params)
        plt.rcParams.update(pgf_rc_params)

        # Set PGF preamble
        plt.rcParams["pgf.preamble"] = "\n".join(
            [
                r"\usepackage{amsmath}",
                r"\usepackage{amssymb}",
                r"\usepackage{mathpazo}",
            ]
        )
        plt.rcParams["text.latex.preamble"] = r"\newcommand{\mathdefault}[1][]{}"

    def reset_pgf(self):
        """Reset PGF settings and switch to display backend."""
        plt.rcdefaults()
        plt.close("all")  # Close all figures
        matplotlib.use("Agg")  # Switch to non-interactive backend

    def reset_config(self):
        """Reset to the appropriate configuration based on the current setup."""
        plt.rcdefaults()
        if self.config.matplotlib.pgf_enabled:
            self.setup_pgf()
        else:
            self.setup_jupyter()

    def create_figure(self):
        """Create a figure with the configured settings."""
        fig, ax = plt.subplots(figsize=self.config.matplotlib.figsize)
        ax.set_xlabel(self.config.line_plot.xlabel)
        ax.set_ylabel(self.config.line_plot.ylabel)
        ax.grid(self.config.matplotlib.grid)
        return fig, ax

    def save_figure(self, filename: str):
        """Speichert die aktuelle Figure in mehreren Formaten und schließt sie danach."""
        # Hole die aktuelle Figure
        fig = plt.gcf()
        for fmt in self.config.save.save_formats:
            # Für PGF: rcParams sicher setzen
            if fmt == "pgf":
                plt.rcParams.update(self.config.matplotlib.get_pgf_rc_params())
                plt.rcParams["pgf.preamble"] = "\n".join(
                    [
                        r"\usepackage{amsmath}",
                        r"\usepackage{amssymb}",
                        r"\usepackage{mathpazo}",
                    ]
                )
                plt.rcParams["text.latex.preamble"] = (
                    r"\newcommand{\mathdefault}[1][]{}"
                )
            path = os.path.join(self.config.save.save_dir, f"{filename}.{fmt}")
            fig.savefig(
                path,
                # bbox_inches=self.config.save.bbox_inches,
                dpi=self.config.matplotlib.dpi,
            )
        plt.close(fig)
        print(
            f"✅ Saved to {self.config.save.save_dir}/{filename}.{{{', '.join(self.config.save.save_formats)}}}"
        )

    def finish(self):
        """Finish plotting and clean up."""
        if self.config.matplotlib.pgf_enabled:
            # For PGF plots, save and reset backend
            plt.close("all")  # Close all figures
            self.reset_pgf()
        else:
            # For Jupyter/display plots, show the plot
            plt.show()

    def __del__(self):
        """Destructor to clean up when the object is destroyed."""
        try:
            self.finish()
        except:
            pass  # Ignore errors during cleanup

    def get_colors(self, n):
        """Get a color palette for n elements."""
        colormap = plt.get_cmap(self.config.matplotlib.colormap)
        colormap_type = self.config.matplotlib.get_colormap_type()

        if colormap_type == "qualitative":
            # For qualitative colormaps, use discrete color indices
            # Most qualitative colormaps have 10 colors, so we cycle through them
            colors = []
            for i in range(n):
                colors.append(colormap(i % colormap.N))
            return colors
        else:
            # For continuous colormaps, use linspace
            return colormap(np.linspace(0, 1, n))

    def plot_confidence_intervals(
        self,
        mean_frame: pd.DataFrame,
        std_frame: pd.DataFrame,
        sigma: int = None,
        **kwargs,
    ):
        """Plot confidence intervals for a list of PlotFrames."""

        # Use config sigma if not provided
        if sigma is None:
            sigma = self.config.confidence_interval.sigma

        fig, ax = plt.subplots(figsize=self.config.matplotlib.figsize)
        colors = self.get_colors(len(mean_frame.columns))

        filename = kwargs.get("filename", "confidence_intervals")
        ylabel = kwargs.get("ylabel", "Balance")
        xlabel = kwargs.get("xlabel", "Date")

        for i, column in enumerate(mean_frame.columns):
            # Compute mean and std over the specified level
            mean_series = mean_frame[column]
            std_series = std_frame[column]

            # Plot mean line
            ax.plot(
                mean_series.index,
                mean_series.values,
                label=f"{column} $(\\pm {sigma} \\sigma)$",
                color=colors[i],
                linewidth=self.config.matplotlib.linewidth,
            )

            # Plot ±sigma region
            ax.fill_between(
                mean_series.index,
                (mean_series.values.flatten() - sigma * std_series.values.flatten()),
                (mean_series.values.flatten() + sigma * std_series.values.flatten()),
                alpha=self.config.confidence_interval.fill_alpha,
                color=colors[i],
            )

        ax.set_ylabel(ylabel)
        ax.set_xlabel(xlabel)

        # Legend positioning
        if self.config.confidence_interval.legend_loc is not None:
            if self.config.confidence_interval.legend_loc == "outside right":
                ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
                plt.tight_layout(rect=[0, 0, 0.85, 1])
            elif self.config.confidence_interval.legend_loc == "outside top":
                ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
                plt.tight_layout(rect=[0, 0, 1, 0.9])
            else:
                ax.legend()
        else:
            ax.legend().remove()

        # limit x axis to the min and max of the mean_frame
        ax.set_xlim(mean_frame.index.min(), mean_frame.index.max())
        ax.grid(self.config.matplotlib.grid)
        # plt.tight_layout() # Keep this to have Plot size and Legend fit the desired size.
        self.save_figure(filename)

        # Return the figure for Jupyter display
        return fig

    def plot_lines(
        self,
        df: pd.DataFrame,
        x_axis: Optional[pd.Series] = None,
        xlabel: str = None,
        ylabel: str = None,
        title: str = "",
        filename: str = "plot_output",
        max_xticks: int = None,
        linewidth: float = None,
        y_limits: tuple[float, float] | None = None,
        linestyles: bool = None,
        smoothing: int = None,
        legend_loc: str = None,
    ):
        """Internal method for plotting lines with various configurations."""

        # Use config defaults if not specified
        if max_xticks is None:
            max_xticks = self.config.line_plot.max_xticks
        if linewidth is None:
            linewidth = self.config.matplotlib.linewidth
        if linestyles is None:
            linestyles = self.config.line_plot.linestyles
        if legend_loc is None:
            legend_loc = self.config.line_plot.legend_loc
        if y_limits is None:
            y_limits = self.config.line_plot.y_limits
        if smoothing is None:
            smoothing = self.config.line_plot.smoothing

        if isinstance(df, pd.Series):
            df = df.to_frame()

        n_lines = len(df.columns)

        # Handle x-axis
        if x_axis is None:
            x = df.index
            if not pd.api.types.is_datetime64_any_dtype(x):
                try:
                    x = pd.to_datetime(x)
                    df = df.set_index(x)
                except Exception as e:
                    raise ValueError(
                        "Index is not datetime and cannot be converted."
                    ) from e
        else:
            x = x_axis

        # Get colors from configuration
        colors = self.get_colors(n_lines)

        # Linestyles
        if linestyles:
            linestyle_list = ["-", "--", "-."] * (n_lines // 4 + 1)
        else:
            linestyle_list = ["-"] * n_lines

        fig, ax = plt.subplots(figsize=self.config.matplotlib.figsize)

        for i, column in enumerate(df.columns):
            label = str(column).replace("_", " ")  # .upper()

            ax.plot(
                x,
                df[column],
                label=label,
                color=colors[i],
                linewidth=linewidth,
                linestyle=linestyle_list[i % len(linestyle_list)],
                alpha=1 if smoothing is None else 0.5,
            )

            if smoothing is not None:
                df_smoothed = df[column].rolling(window=smoothing).mean()
                ax.plot(
                    x,
                    df_smoothed,
                    label="",
                    color=colors[i],
                    linewidth=linewidth,
                    linestyle=linestyle_list[i % len(linestyle_list)],
                    alpha=1.0,
                )

        # Handle time-based x-axis
        if x_axis is None:
            ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=max_xticks))
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            fig.autofmt_xdate()

        if xlabel is not None:
            ax.set_xlabel(xlabel)
        if ylabel is not None:
            ax.set_ylabel(ylabel)
        if title is not None:
            ax.set_title(title)

        # Y-Limits handling
        if y_limits is not None:
            if y_limits[0] is not None:
                ax.set_ylim(bottom=y_limits[0])
            if y_limits[1] is not None:
                ax.set_ylim(top=y_limits[1])
        else:
            y_ticks = ax.get_yticks()
            y_min_data, y_max_data = ax.get_ylim()
            y_lower_ticks = y_ticks[y_ticks <= y_min_data]
            y_upper_ticks = y_ticks[y_ticks >= y_max_data]
            if len(y_lower_ticks) > 0 and len(y_upper_ticks) > 0:
                ax.set_ylim(y_lower_ticks[0], y_upper_ticks[-1])

        # X-Limits
        ax.set_xlim(x.min(), x.max())
        # ax.set_xticks(x[::max(1, len(x) // max_xticks)])

        # Legend positioning
        if legend_loc == "outside right":
            ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        elif legend_loc == "outside top":
            ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.02), frameon=False)
            plt.tight_layout(rect=[0, 0, 1, 0.9])

        plt.grid(True)
        # plt.tight_layout()  #
        # Save figure
        self.save_figure(filename)

        # Return the figure for Jupyter display
        return fig

    def plot_dual_axis(
        self,
        df: pd.DataFrame,
        ylabel_left: str = "Y1",
        ylabel_right: str = "Y2",
        title: str = "",
        filename: str = "dual_axis_plot",
        max_xticks: int = None,
        max_entries: int = None,
        linewidth: float = None,
        num_yticks: int = None,
        round_base: int = None,
        verbosity: int = None,
        y_limits_left: tuple[float, float] = None,
        y_limits_right: tuple[float, float] = None,
    ):
        """Plot two lines from a DataFrame using left and right Y-axes."""

        # Use config defaults if not specified
        if max_xticks is None:
            max_xticks = self.config.dual_axis.max_xticks
        if max_entries is None:
            max_entries = self.config.dual_axis.max_entries
        if linewidth is None:
            linewidth = self.config.matplotlib.linewidth
        if num_yticks is None:
            num_yticks = self.config.dual_axis.num_yticks
        if round_base is None:
            round_base = self.config.dual_axis.round_base
        if verbosity is None:
            verbosity = self.config.dual_axis.verbosity
        if y_limits_left is None:
            y_limits_left = self.config.dual_axis.y_limits_left
        if y_limits_right is None:
            y_limits_right = self.config.dual_axis.y_limits_right

        if len(df.columns) != 2:
            raise ValueError(
                "The DataFrame must contain exactly two columns for dual-axis plotting."
            )

        # Limit entries if specified
        if max_entries is not None and len(df) > max_entries:
            step = max(1, len(df) // max_entries)
            df = df.iloc[::step]

        x = df.index
        left_label, right_label = df.columns

        fig, ax_left = plt.subplots(figsize=self.config.matplotlib.figsize)

        # Plot left axis
        ax_left.plot(
            x,
            df[left_label],
            label=left_label.replace("_", " ").upper(),
            color="black",
            linewidth=linewidth,
        )
        ax_left.set_ylabel(ylabel_left)
        ax_left.tick_params(axis="y", labelcolor="black")

        # Plot right axis
        ax_right = ax_left.twinx()
        ax_right.plot(
            x,
            df[right_label],
            label=right_label.replace("_", " ").upper(),
            color="gray",
            linewidth=linewidth,
        )
        ax_right.set_ylabel(ylabel_right, color="gray")
        ax_right.tick_params(axis="y", labelcolor="gray")

        # Handle time-based x-axis
        is_time = pd.api.types.is_datetime64_any_dtype(df.index)
        if is_time:
            ax_left.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=max_xticks))
            ax_left.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
            fig.autofmt_xdate()
        else:
            if len(x) > max_xticks:
                step = max(1, len(x) // max_xticks)
                ax_left.set_xticks(x[::step])

        ax_left.set_xlabel(self.config.line_plot.xlabel)
        ax_left.set_title(title)

        # Set Y-limits and ticks
        if y_limits_left is None:
            # Auto-calculate limits
            left_min, left_max = df[left_label].min(), df[left_label].max()
            right_min, right_max = df[right_label].min(), df[right_label].max()

            # Round limits
            left_min, left_max = np.floor(left_min), np.ceil(
                self._round_up_to_nearest(left_max, base=round_base)
            )
            right_min, right_max = np.floor(right_min), np.ceil(
                self._round_up_to_nearest(right_max, base=round_base)
            )

            # Set ticks and limits
            left_ticks = np.linspace(0, left_max, num_yticks)
            right_ticks = np.linspace(0, right_max, num_yticks)

            ax_left.set_yticks(left_ticks)
            ax_right.set_yticks(right_ticks)
            ax_left.set_ylim(left_min, left_ticks[-1])
            ax_right.set_ylim(right_min, right_ticks[-1])
            ax_left.set_ylim(bottom=0)
            ax_right.set_ylim(bottom=0)
        else:
            # Use provided limits
            ax_left.set_ylim(y_limits_left)
            ax_right.set_ylim(y_limits_right)

        ax_left.set_xlim(x.min(), x.max())
        plt.tight_layout()
        plt.grid(True)
        plt.grid(which="both", linestyle="--", linewidth=0.5)

        self.save_figure(filename)
        if verbosity > 0:
            print(f"✅ Dual-axis plot saved")

        # Return the figure for Jupyter display
        return fig

    def _round_up_to_nearest(self, value, base=10):
        """Round a value up to the nearest base value."""
        return base * np.ceil(value / base)

    def plot_hist_grid(
        self,
        data: pd.DataFrame,
        n_cols: int = None,
        bins: int = None,
        x_name: str = "Return",
        y_name: str = "Probability Density",
        figsize: Tuple[float, float] = None,
        log_y_scale: bool = None,
        ylim: Optional[Tuple[float, float]] = None,
        hist_color: str = None,
        kde_color: str = None,
        alpha: float = None,
    ):
        """Plot a grid of histogram and KDE plots for MultiIndex DataFrame."""

        # Use config defaults if not specified
        if n_cols is None:
            n_cols = self.config.grid_plot.n_cols
        if bins is None:
            bins = self.config.grid_plot.bins
        if log_y_scale is None:
            log_y_scale = self.config.grid_plot.log_y_scale
        if ylim is None:
            ylim = self.config.grid_plot.ylim
        if hist_color is None:
            hist_color = self.config.grid_plot.hist_color or self.get_colors(1)[0]
        if kde_color is None:
            kde_color = self.config.grid_plot.kde_color or self.get_colors(2)[1]
        if alpha is None:
            alpha = self.config.matplotlib.alpha
        if figsize is None:
            figsize = self.config.matplotlib.figsize

        tickers = sorted(set(data.columns))
        n_rows = int(np.ceil(len(tickers) / n_cols))

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        )
        axes = np.atleast_1d(axes).flatten()

        for i, ticker in enumerate(tickers):
            ax = axes[i]
            data[ticker].plot.hist(
                bins=bins, density=True, ax=ax, color=hist_color, alpha=alpha
            )
            data[ticker].plot.kde(ax=ax, color=kde_color)

            ax.set_title(ticker)
            ax.set_ylabel(y_name)
            ax.set_xlabel(x_name)
            ax.grid(True)
            ax.set_xlim(-0.1, 0.1)

            if log_y_scale:
                ax.set_yscale("log")
            if ylim is not None:
                ax.set_ylim(ylim)

            ax.legend(["Histogram", "KDE"], loc="upper right")

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        self.save_figure("hist_grid")

        # Return the figure for Jupyter display
        return fig

    def plot_grid(
        self,
        data: pd.DataFrame,
        n_cols: int = None,
        y_name: str = "Price",
        x_name: str = "Date",
        figsize: Tuple[float, float] = None,
        ylim: Optional[Tuple[float, float]] = None,
        line_color: str = None,
    ):
        """Plot a grid of time series plots for MultiIndex DataFrame."""

        # Use config defaults if not specified
        if n_cols is None:
            n_cols = self.config.grid_plot.n_cols
        if figsize is None:
            figsize = self.config.matplotlib.figsize
        if line_color is None:
            line_color = self.get_colors(1)[0]  # Use first color from config colormap

        # try to extract the tickers form the first level of the columns
        try:
            tickers = sorted(set(col for col in data.columns.levels[0]))
        except:
            tickers = sorted(set(col for col in data.columns))

        n_rows = int(np.ceil(len(tickers) / n_cols))

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        )
        # Ensure axes is always 2D for consistent indexing
        if n_rows == 1 and n_cols == 1:
            axes = np.array([[axes]])
        elif n_rows == 1:
            axes = np.array([axes])
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        else:
            axes = np.atleast_2d(axes)

        # Always use the first color from the colormap for all plots
        first_color = self.get_colors(1)[0]

        for i, ticker in enumerate(tickers):
            row, col = divmod(i, n_cols)
            ax = axes[row, col]

            data[ticker].plot(
                ax=ax,
                title=ticker,
                ylabel=y_name,
                xlabel=x_name,
                grid=True,
                color=first_color,
                linewidth=self.config.matplotlib.linewidth,
            )
            ax.legend().remove()

            if ylim is not None:
                ax.set_ylim(ylim)

        # Hide unused axes
        total_plots = n_rows * n_cols
        for j in range(len(tickers), total_plots):
            row, col = divmod(j, n_cols)
            axes[row, col].axis("off")

        plt.tight_layout()
        self.save_figure("time_series_grid")

        # Return the figure for Jupyter display
        return fig

    def plot_lines_grayscale(
        self,
        df: pd.DataFrame,
        x_axis: Optional[pd.Series] = None,
        xlabel: str = "Date",
        ylabel: str = "Y",
        title: str = "",
        filename: str = "plot_output",
        max_xticks: int = 12,
        y_limits: tuple[float, float] | None = None,
        figsize: tuple[float, float] = None,
        linewidth: float = None,
    ) -> None:
        """
        Plot multiple lines from a DataFrame in grayscale with maximally spaced intensities.
        Detects time-based x-axis and formats accordingly.
        Optionally applies fixed y-limits, otherwise snaps to next grid tick.
        Replaces underscores in legend labels and converts them to uppercase.
        """
        # Use config defaults if not specified
        if figsize is None:
            figsize = self.config.matplotlib.figsize
        if linewidth is None:
            linewidth = self.config.matplotlib.linewidth

        n_lines: int = len(df.columns)

        if x_axis is None:
            x = df.index
            # Ensure the index is datetime if it looks like dates
            if not pd.api.types.is_datetime64_any_dtype(x):
                try:
                    x = pd.to_datetime(x)
                    df = df.set_index(x)
                except Exception as e:
                    raise ValueError(
                        "Index is not datetime and cannot be converted."
                    ) from e
        else:
            x = x_axis

        grays = np.linspace(0.25, 1.0, n_lines)
        colors = [str(1 - g) for g in grays]

        fig, ax = plt.subplots(figsize=figsize)

        for i, column in enumerate(df.columns):
            label = str(column).replace("_", " ").upper()
            ax.plot(
                df.index, df[column], label=label, color=colors[i], linewidth=linewidth
            )

        # Handle time-based x-axis
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=max_xticks))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()

        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)

        fig.canvas.draw()  # ensure ticks are computed

        # Y-Limits: manually set or snap to nearest grid ticks
        if y_limits is not None:
            ax.set_ylim(y_limits)
        else:
            y_ticks = ax.get_yticks()
            y_min_data, y_max_data = ax.get_ylim()
            y_lower_ticks = y_ticks[y_ticks <= y_min_data]
            y_upper_ticks = y_ticks[y_ticks >= y_max_data]
            if len(y_lower_ticks) > 0 and len(y_upper_ticks) > 0:
                ax.set_ylim(y_lower_ticks[0], y_upper_ticks[-1])

        # X-Limits: set to min and max of x
        ax.set_xlim(x.min(), x.max())
        ax.set_xticks(x[:: max(1, len(x) // max_xticks)])  # Set x-ticks to max_xticks

        ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
        plt.tight_layout(rect=[0, 0, 0.85, 1])
        ax.grid(True)

        self.save_figure(filename)

        # Return the figure for Jupyter display
        return fig

    def plot_risk_matrix(
        self,
        expected_returns: pd.DataFrame,
        expected_covariance: pd.DataFrame,
        colorscheme: str = None,
        figsize: tuple[float, float] = None,
        fontsize: int = 8,
        filename: str = "risk_matrix",
        title: str = "Risk Matrix",
    ):
        """
        Plot a risk matrix showing covariance and expected returns.
        """
        # Use config defaults if not specified
        if colorscheme is None:
            colorscheme = self.config.matplotlib.colormap
        if figsize is None:
            figsize = self.config.matplotlib.figsize

        tickers = expected_returns.index.tolist()

        # if tickers are a list of tuples we have a multi-index DataFrame
        if isinstance(tickers[0], tuple):
            # then we want a list of the first elements of the tuples
            tickers = [ticker[0] for ticker in tickers]

        # and make them unique
        tickers = list(dict.fromkeys(tickers))

        # Extend the matrix with mean returns
        extended_matrix = expected_covariance.copy()
        extended_matrix["E[R]"] = expected_returns  # Add a column for mean returns

        # Visualize the extended matrix with matplotlib
        fig, ax = plt.subplots(figsize=figsize)  # Larger plot window for additional row

        # Axis ticks and labels (using LaTeX for labels)
        latex_tickers = [r"{" + ticker + r"}" for ticker in tickers]
        ax.set_xticks(range(len(tickers) + 1))  # Additional column for mean returns
        ax.set_yticks(range(len(tickers)))  # Additional row for mean returns
        ax.set_yticklabels(latex_tickers, fontsize=8)

        # Write numbers into the matrix (rounded to one decimal place)
        for i in range(len(tickers)):
            for j in range(len(tickers)):
                value = expected_covariance.iloc[i, j]
                color = (
                    "white" if value > 0 else "black"
                )  # White for values > 0, black for values <= 0
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=fontsize,
                )

        # Write mean returns into the last column
        for j in range(len(tickers)):
            mean_value = expected_returns.iloc[j]
            value = mean_value
            color = (
                "white" if value > 0 else "black"
            )  # White for values > 0, black for values <= 0
            ax.text(
                len(tickers),
                j,
                f"{mean_value:.2%}",
                ha="center",
                va="center",
                color=color,
                fontsize=fontsize,
            )

        cax = ax.matshow(
            extended_matrix, cmap=colorscheme
        )  # Grayscale representation with limited color scale
        fig.colorbar(cax, ax=ax, fraction=0.04, pad=0.03)  # Add a color bar
        if title:
            ax.set_title(title, fontsize=fontsize + 2, pad=20)

        # Remove the title and save the plot
        plt.tight_layout()
        self.save_figure(filename)

        # Return the figure for Jupyter display
        return fig

    def plot_hist_grid_compare(
        self,
        dataframes: List[pd.DataFrame],
        data_names: List[str],
        plot_kde: bool = True,
        plot_hist: bool = False,
        x_name: str = "Return",
        y_name: str = "Probability Density",
        n_cols: int = None,
        bins: int = 50,
        figsize: Tuple[float, float] = None,
        log_y_scale: bool = False,
        ylim: Optional[Tuple[float, float]] = None,
        colors: Optional[List[str]] = None,
        markers: Optional[List[str]] = None,
        hist_alpha: float = 0.3,
        legend: bool = True,
        legend_loc: str = "upper right",
        filename: str = "hist_grid_compare",
    ) -> None:
        """
        Plots a grid of histogram and KDE plots for each unique first-level column label in multiple MultiIndex DataFrames.

        Args:
            dataframes (List[pd.DataFrame]): List of MultiIndex DataFrames with two-level column index (e.g., (ticker, attribute)).
            data_names (List[str]): List of names for each DataFrame to be used in the legend.
            n_cols (int): Number of columns in the plot grid.
            bins (int): Number of bins for the histogram.
            figsize (Tuple[float, float]): Size of each subplot (width, height).
            log_y_scale (bool): Whether to use a logarithmic scale for the y-axis.
            ylim (Optional[Tuple[float, float]]): Y-axis limits. If None, defaults to matplotlib's auto-scaling.
            colors (Optional[List[str]]): Colors for each DataFrame. If None, uses default colors.
            hist_alpha (float): Alpha value for histogram bars (0-1).
        """
        if len(dataframes) != len(data_names):
            raise ValueError("Number of dataframes must match number of data_names")

        # Use config defaults if not specified
        if n_cols is None:
            n_cols = self.config.grid_plot.n_cols
        if figsize is None:
            figsize = self.config.matplotlib.figsize
        if colors is None:
            colors = self.get_colors(len(dataframes))

        if len(colors) != len(dataframes):
            raise ValueError("Number of colors must match number of dataframes")

        if markers is None:
            markers = ["-"] * len(dataframes)
        if len(markers) != len(dataframes):
            raise ValueError("Number of markers must match number of dataframes")

        # Extract unique tickers from the first level of all MultiIndex DataFrames
        tickers = sorted(
            set.intersection(*[set(col[0] for col in df.columns) for df in dataframes])
        )
        n_rows = int(np.ceil(len(tickers) / n_cols))

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        )
        axes = np.atleast_1d(axes).flatten()
        labels = []
        for i, ticker in enumerate(tickers):
            ax = axes[i]

            # Plot histogram and KDE for all attributes under this ticker for each DataFrame
            for df, name, color, marker in zip(dataframes, data_names, colors, markers):
                if ticker in df.columns.get_level_values(0):
                    if plot_hist:
                        df[ticker].plot.hist(
                            bins=bins,
                            density=True,
                            ax=ax,
                            color=color,
                            alpha=hist_alpha,
                        )
                        labels.append(f"{ticker} ({name})")
                    if plot_kde:
                        df[ticker].plot.kde(ax=ax, color=color, linestyle=marker)
                        labels.append(f"{ticker} KDE ({name})")
                    if not plot_hist and not plot_kde:
                        raise ValueError(
                            "plot_hist and plot_kde cannot be False at the same time"
                        )

            ax.set_title(ticker)
            ax.set_ylabel(y_name)
            ax.set_xlabel(x_name)
            ax.grid(True)
            ax.set_xlim(-0.1, 0.1)

            if log_y_scale:
                ax.set_yscale("log")
            if ylim is not None:
                ax.set_ylim(ylim)

            # Create legend entries for all DataFrames
            if legend:
                ax.legend(labels, loc=legend_loc)

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        self.save_figure(filename)

        # Return the figure for Jupyter display
        return fig

    def plot_cdf_grid_compare(
        self,
        dataframes: List[pd.DataFrame],
        data_names: List[str],
        x_name: str = "Return",
        y_name: str = "Cumulative Probability",
        n_cols: int = None,
        figsize: Tuple[float, float] = None,
        log_y_scale: bool = False,
        ylim: Optional[Tuple[float, float]] = None,
        colors: Optional[List[str]] = None,
        line_alpha: float = 0.7,
        x_lim: Optional[Tuple[float, float]] = None,
        filename: str = "cdf_grid_compare",
    ) -> None:
        """
        Plots a grid of CDF plots for each unique first-level column label in multiple MultiIndex DataFrames.

        Args:
            dataframes (List[pd.DataFrame]): List of MultiIndex DataFrames with two-level column index (e.g., (ticker, attribute)).
            data_names (List[str]): List of names for each DataFrame to be used in the legend.
            n_cols (int): Number of columns in the plot grid.
            figsize (Tuple[float, float]): Size of each subplot (width, height).
            log_y_scale (bool): Whether to use a logarithmic scale for the y-axis.
            ylim (Optional[Tuple[float, float]]): Y-axis limits. If None, defaults to matplotlib's auto-scaling.
            colors (Optional[List[str]]): Colors for each DataFrame. If None, uses default colors.
            line_alpha (float): Alpha value for the CDF lines (0-1).
        """
        if len(dataframes) != len(data_names):
            raise ValueError("Number of dataframes must match number of data_names")

        # Use config defaults if not specified
        if n_cols is None:
            n_cols = self.config.grid_plot.n_cols
        if figsize is None:
            figsize = self.config.matplotlib.figsize
        if colors is None:
            colors = self.get_colors(len(dataframes))

        if len(colors) != len(dataframes):
            raise ValueError("Number of colors must match number of dataframes")

        # Extract unique tickers from the first level of all MultiIndex DataFrames
        tickers = sorted(
            set.intersection(*[set(col[0] for col in df.columns) for df in dataframes])
        )
        n_rows = int(np.ceil(len(tickers) / n_cols))

        fig, axes = plt.subplots(
            nrows=n_rows,
            ncols=n_cols,
            figsize=(figsize[0] * n_cols, figsize[1] * n_rows),
        )
        axes = np.atleast_1d(axes).flatten()

        for i, ticker in enumerate(tickers):
            ax = axes[i]

            # Plot CDF for all attributes under this ticker for each DataFrame
            for df, name, color in zip(dataframes, data_names, colors):
                if ticker in df.columns.get_level_values(0):
                    # Calculate CDF
                    data = df[ticker].values.flatten()
                    sorted_data = np.sort(data)
                    cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

                    # Plot CDF
                    ax.plot(
                        sorted_data,
                        cdf,
                        color=color,
                        alpha=line_alpha,
                        label=f"{ticker} ({name})",
                    )

            ax.set_title(ticker)
            ax.set_ylabel(y_name)
            ax.set_xlabel(x_name)
            ax.grid(True)
            if x_lim is not None:
                ax.set_xlim(x_lim)
            ax.set_ylim(0, 1)

            if log_y_scale:
                ax.set_yscale("log")
            if ylim is not None:
                ax.set_ylim(ylim)

            # Create legend entries for all DataFrames
            legend_entries = [f"{ticker} ({name})" for name in data_names]
            ax.legend(legend_entries, loc="lower right")

        # Hide unused subplots
        for j in range(i + 1, len(axes)):
            axes[j].axis("off")

        plt.tight_layout()
        self.save_figure(filename)

        # Return the figure for Jupyter display
        return fig
