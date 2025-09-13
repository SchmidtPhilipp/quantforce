"""
Plotting utilities for QuantForce.

This module provides both traditional matplotlib plotting functions and
LaTeX-optimized plotting capabilities.
"""

import os
from typing import List, Optional, Tuple

import matplotlib
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import qf


def setup_pgf() -> None:
    """
    Set up PGF backend for matplotlib to export figures in LaTeX-compatible format.
    """
    # Use PGF backend
    matplotlib.use("pgf")

    # PGF export for LaTeX
    plt.rcParams.update(
        {
            "pgf.texsystem": "pdflatex",
            "figure.figsize": [6.6, 2.3],
            "font.family": "helvet",
            "text.usetex": True,
            "pgf.rcfonts": False,
            "axes.formatter.use_mathtext": True,
            "pgf.preamble": "\n".join(
                [
                    r"\usepackage{amsmath}",
                    r"\usepackage{amssymb}",
                    r"\usepackage{mathpazo}",  # Mathpazo als Schriftart
                ]
            ),
            "font.size": 12,  # Allgemeine Schriftgröße
            "axes.titlesize": 14,  # Titel der Achsen
            "axes.labelsize": 12,  # Beschriftung der Achsen
            "xtick.labelsize": 9,  # Schriftgröße der X-Achsenticks
            "ytick.labelsize": 9,  # Schriftgröße der Y-Achsenticks
            "legend.fontsize": 10,  # Schriftgröße der Legend"
            "axes.grid": True,  # Gitterlinien aktivieren
            "grid.alpha": 0.5,  # Transparenz der Gitterlinien
            "grid.linestyle": "--",  # Linienstil der Gitterlinien
            # "grid.linewidth": 0.5,  # Linienstärke der Gitterlinien
        }
    )
    plt.rcParams["text.latex.preamble"] = (
        r"\newcommand{\mathdefault}[1][]{}"  # Setze den Befehl für mathdefault
    )


def reset_pgf() -> None:
    plt.rcdefaults()  # Standardwerte für rcParams
    matplotlib.use("macosx")  # Setzt das Backend auf das Standard-Backend zurück


def plot_lines_grayscale(
    df: pd.DataFrame,
    x_axis: Optional[pd.Series] = None,
    xlabel: str = "Date",
    ylabel: str = "Y",
    title: str = "",
    filename: str = "plot_output",
    save_dir: str = "tmp/plots",
    max_xticks: int = 12,
    y_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (10, 2.5),
    linewidth: float = 2.0,
) -> None:
    """
    Plot multiple lines from a DataFrame in grayscale with maximally spaced intensities.
    Detects time-based x-axis and formats accordingly.
    Optionally applies fixed y-limits, otherwise snaps to next grid tick.
    Replaces underscores in legend labels and converts them to uppercase.
    Exports as PGF and PNG.
    """
    setup_pgf()  # Set up PGF backend for LaTeX compatibility
    os.makedirs(save_dir, exist_ok=True)

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

    grays = np.linspace(0.25, 1.0, n_lines)
    colors = [str(1 - g) for g in grays]

    fig, ax = plt.subplots(figsize=figsize)

    for i, column in enumerate(df.columns):
        label = str(column).replace("_", " ").upper()
        ax.plot(df.index, df[column], label=label, color=colors[i], linewidth=linewidth)

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
    plt.grid(True)

    pgf_path = os.path.join(save_dir, f"{filename}.pgf")
    png_path = os.path.join(save_dir, f"{filename}.png")
    fig.savefig(pgf_path, format="pgf")
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    reset_pgf()  # Reset PGF settings to default
    print(f"✅ Plot saved to {pgf_path} and {png_path}")


def plot_lines(
    df: pd.DataFrame,
    x_axis: Optional[pd.Series] = None,
    xlabel: str = "Date",
    ylabel: str = "Y",
    title: str = "",
    filename: str = "plot_output",
    save_dir: str = "tmp/plots",
    max_xticks: int = 12,
    x_limits: tuple[float, float] | None = None,
    y_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = qf.DEFAULT_FIGSIZE_PAPER,
    linewidth: float = 2.0,
    colorscheme: str = "jet",
    linestyles: bool = True,
    return_fig: bool = False,
    pgf_setup: bool = True,
    smoothing: int = None,
):
    """
    Plot multiple lines from a DataFrame in grayscale with maximally spaced intensities.
    Detects time-based x-axis and formats accordingly.
    Optionally applies fixed y-limits, otherwise snaps to next grid tick.
    Replaces underscores in legend labels and converts them to uppercase.
    Exports as PGF and PNG.

    Parameters:
        df: DataFrame containing the data to plot
        x_axis: Optional Series to use as x-axis, if None the index of the DataFrame is used
        xlabel: Label for the x-axis
        ylabel: Label for the y-axis
        title: Title of the plot
        filename: Name of the file to save the plot to
        save_dir: Directory to save the plot to
        max_xticks: Maximum number of x-ticks
        y_limits: Tuple of (minimum, maximum) y-limits
        figsize: Tuple of (width, height) in inches
        linewidth: Width of the lines
        colorscheme: Color scheme to use
        linestyles: Whether to use different linestyles for each line
        return_fig: Whether to return the figure object
        pgf_setup: Whether to set up the PGF backend
        smoothing: Window size for smoothing the data, the smoothed data is plotted as the same color but with a alpha of 1 and the original data is plotted with a alpha of 0.5. If smoothing is None, no smoothing is applied.
    """
    if pgf_setup:
        setup_pgf()  # Set up PGF backend for LaTeX compatibility
    os.makedirs(save_dir, exist_ok=True)

    n_lines = len(df.columns)
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

    if colorscheme == "gray":
        grays = np.linspace(0.25, 1.0, n_lines)
        colors = [str(1 - g) for g in grays]

    elif colorscheme == "hsv":
        # Use HSV colors with varying saturation and brightness
        hsv_colors = plt.cm.hsv(np.linspace(0, 1, n_lines))
        colors = [matplotlib.colors.rgb2hex(hsv_color) for hsv_color in hsv_colors]
    elif colorscheme == "viridis":
        # Use Viridis colors
        viridis_colors = plt.cm.viridis(np.linspace(0, 1, n_lines))
        colors = [
            matplotlib.colors.rgb2hex(viridis_color) for viridis_color in viridis_colors
        ]
    elif colorscheme == "plasma":
        # Use Plasma colors
        plasma_colors = plt.cm.plasma(np.linspace(0, 1, n_lines))
        colors = [
            matplotlib.colors.rgb2hex(plasma_color) for plasma_color in plasma_colors
        ]
    elif colorscheme == "cividis":
        # Use Cividis colors
        cividis_colors = plt.cm.cividis(np.linspace(0, 1, n_lines))
        colors = [
            matplotlib.colors.rgb2hex(cividis_color) for cividis_color in cividis_colors
        ]
    elif colorscheme == "jet":
        # Use Jet colors
        jet_colors = plt.cm.jet(np.linspace(0, 1, n_lines))
        colors = [matplotlib.colors.rgb2hex(jet_color) for jet_color in jet_colors]
    elif colorscheme == "gist_rainbow":
        # Use Gist Rainbow colors
        gist_rainbow_colors = plt.cm.gist_rainbow(np.linspace(0, 1, n_lines))
        colors = [
            matplotlib.colors.rgb2hex(gist_rainbow_color)
            for gist_rainbow_color in gist_rainbow_colors
        ]

    else:
        raise ValueError(
            f"Unsupported colorscheme: {colorscheme}. Supported schemes are: 'gray', 'hsv', 'viridis', 'plasma', 'cividis', 'jet', 'gist_rainbow'."
        )

    # If linestyles is True then we cylce thorugh different linestyles
    if linestyles:
        linestyle_list = ["-", "--", "-."] * (n_lines // 4 + 1)
    else:
        linestyle_list = ["-"] * n_lines

    fig, ax = plt.subplots(figsize=figsize)

    for i, column in enumerate(df.columns):
        label = str(column).replace("_", " ").upper()

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
            label = f""
            ax.plot(
                x,
                df_smoothed,
                label=label,
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

    if x_limits is not None:
        ax.set_xlim(x_limits)
    else:
        ax.set_xlim(x.min(), x.max())
        ax.set_xticks(x[:: max(1, len(x) // max_xticks)])  # Set x-ticks to max_xticks

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True)

    pgf_path = os.path.join(save_dir, f"{filename}.pgf")
    png_path = os.path.join(save_dir, f"{filename}.png")
    fig.savefig(pgf_path, format="pgf")
    fig.savefig(png_path, dpi=300)
    if return_fig:
        return fig
    else:
        plt.close(fig)
        reset_pgf()  # Reset PGF settings to default
        print(f"✅ Plot saved to {pgf_path} and {png_path}")


def plot_dual_axis(
    df: pd.DataFrame,
    xlabel: str = "Date",
    ylabel_left: str = "Y1",
    ylabel_right: str = "Y2",
    title: str = "",
    filename: str = "dual_axis_plot",
    save_dir: str = "tmp/plots",
    max_xticks: int = 12,
    max_entries: int = None,
    linewidth: float = 2.0,
    num_yticks: int = 5,  # Anzahl der Y-Ticks
    round_base: int = 100,  # Basis für das Runden der Achsenlimits
    verbosity: int = 0,
    y_limits_left: tuple[float, float] = None,
    y_limits_right: tuple[float, float] = None,
):
    """
    Plot two lines from a DataFrame using left and right Y-axes.
    Ensures the same number of Y-ticks on both sides.
    Positions legends outside the plot.
    """
    setup_pgf()  # Set up PGF backend for LaTeX compatibility
    if len(df.columns) != 2:
        raise ValueError(
            "The DataFrame must contain exactly two columns for dual-axis plotting."
        )

    os.makedirs(save_dir, exist_ok=True)

    # Limit the number of entries if max_entries is specified
    if max_entries is not None and len(df) > max_entries:
        step = max(1, len(df) // max_entries)
        df = df.iloc[::step]

    x = df.index
    left_label, right_label = df.columns

    fig, ax_left = plt.subplots()

    # Plot the first line on the left Y-axis
    ax_left.plot(
        x,
        df[left_label],
        label=left_label.replace("_", " ").upper(),
        color="black",
        linewidth=linewidth,
    )
    ax_left.set_ylabel(ylabel_left)
    ax_left.tick_params(axis="y", labelcolor="black")

    # Create the right Y-axis
    ax_right = ax_left.twinx()
    ax_right.plot(
        x,
        df[right_label],
        label=right_label.replace("_", " ").upper(),
        color="gray",
        linewidth=linewidth,
    )
    ax_right.set_ylabel(ylabel_right)
    ax_right.tick_params(axis="y", labelcolor="gray")
    # Set the right Y-axis label and color
    ax_right.set_ylabel(
        ylabel_right, color="gray"
    )  # Text in der gleichen Farbe wie die Linie
    ax_right.tick_params(
        axis="y", labelcolor="gray"
    )  # Achsenticks in der gleichen Farbe

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

    ax_left.set_xlabel(xlabel)
    ax_left.set_title(title)

    # Set the same number of Y-ticks on both sides
    if y_limits_left is None:
        left_min, left_max = df[left_label].min(), df[left_label].max()
        right_min, right_max = df[right_label].min(), df[right_label].max()
        # Round the limits to the nearest "round" value
        left_min = np.floor(left_min)
        left_max = round_up_to_nearest(left_max, base=round_base)
        left_max = np.ceil(left_max)
        right_min = np.floor(right_min)
        right_max = round_up_to_nearest(right_max, base=round_base)
        right_max = np.ceil(right_max)
        # Calculate the tick positions
        left_ticks = np.linspace(0, left_max, num_yticks)
        right_ticks = np.linspace(0, right_max, num_yticks)

        ax_left.set_yticks(left_ticks)
        ax_right.set_yticks(right_ticks)
        # Ensure the largest tick is at the upper edge
        ax_left.set_ylim(left_min, left_ticks[-1])
        ax_right.set_ylim(right_min, right_ticks[-1])

        # Set the tick labels
        ax_left.set_ylim(bottom=0)
        ax_right.set_ylim(bottom=0)
    else:
        left_min, left_max = y_limits_left
        right_min, right_max = y_limits_right

        ax_left.set_ylim(left_min, left_max)
        ax_right.set_ylim(right_min, right_max)

    ax_left.set_xlim(x.min(), x.max())

    plt.tight_layout()
    plt.grid(True)
    plt.grid(which="both", linestyle="--", linewidth=0.5)

    # Save the plot
    pgf_path = os.path.join(save_dir, f"{filename}.pgf")
    png_path = os.path.join(save_dir, f"{filename}.png")
    fig.savefig(pgf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    reset_pgf()  # Reset PGF settings to default
    if verbosity > 0:
        # Print the paths of the saved files
        print(f"✅ Dual-axis plot saved to {pgf_path} and {png_path}")


def round_up_to_nearest(value, base=10):
    """
    Rundet einen Wert auf die nächste "runde" Zahl auf.

    Parameters:
        value (float): Der Wert, der aufgerundet werden soll.
        base (int): Die Basis, auf die aufgerundet werden soll (z. B. 10, 100, 1000).

    Returns:
        float: Der aufgerundete Wert.
    """
    return base * np.ceil(value / base)


def plot_risk_matrix(
    expected_returns: pd.DataFrame,
    expected_covariance: pd.DataFrame,
    colorscheme: str = qf.DEFAULT_COLORSCHEME,
    figsize: tuple[float, float] = (10, 8),
    fontsize: int = 8,
    save_path: Optional[str] = None,
    title: str = "Risk Matrix",
):

    tickers = (
        expected_returns.index.tolist()
    )  # Get the tickers from the index of expected returns

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
    latex_tickers = [r"$\\textbf{" + ticker + r"}$" for ticker in tickers]
    latex_tickers = [r"{" + ticker + r"}" for ticker in tickers]
    ax.set_xticks(range(len(tickers) + 1))  # Additional column for mean returns
    ax.set_yticks(range(len(tickers)))  # Additional row for mean returns
    # ax.set_xticklabels(latex_tickers + [r"$\\mathbf{E[R]}$"], rotation=45, ha="right", fontsize=fontsize)
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
    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()


def plot_hist_grid(
    data: pd.DataFrame,
    n_cols: int = 3,
    bins: int = 50,
    x_name: str = "Return",
    y_name: str = "Probability Density",
    figsize: Tuple[float, float] = (6, 2),
    log_y_scale: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    hist_color: str = qf.DEFAULT_SINGLE_LINE_COLORS[0],
    kde_color: str = qf.DEFAULT_SINGLE_LINE_COLORS[1],
) -> None:
    """
    Plots a grid of histogram and KDE plots for each unique first-level column label in a MultiIndex DataFrame.

    Args:
        data (pd.DataFrame): MultiIndex DataFrame with two-level column index (e.g., (ticker, attribute)).
        n_cols (int): Number of columns in the plot grid.
        bins (int): Number of bins for the histogram.
        figsize (Tuple[float, float]): Size of each subplot (width, height).
        log_y_scale (bool): Whether to use a logarithmic scale for the y-axis.
        ylim (Optional[Tuple[float, float]]): Y-axis limits. If None, defaults to matplotlib's auto-scaling.
        hist_color (str): Color for histogram bars.
        kde_color (str): Color for KDE curve.
    """
    # Extract unique tickers from the first level of the MultiIndex
    tickers = sorted(set(col[0] for col in data.columns))
    n_rows = int(np.ceil(len(tickers) / n_cols))

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows)
    )
    axes = np.atleast_1d(axes).flatten()

    for i, ticker in enumerate(tickers):
        ax = axes[i]

        # Plot histogram and KDE for all attributes under this ticker
        data[ticker].plot.hist(
            bins=bins, density=True, ax=ax, color=hist_color, alpha=0.7
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
    plt.show()


def plot_hist_grid_compare(
    dataframes: List[pd.DataFrame],
    data_names: List[str],
    plot_kde: bool = True,
    plot_hist: bool = False,
    x_name: str = "Return",
    y_name: str = "Probability Density",
    n_cols: int = 3,
    bins: int = 50,
    figsize: Tuple[float, float] = (6, 2),
    log_y_scale: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    colors: Optional[List[str]] = None,
    markers: Optional[List[str]] = None,
    hist_alpha: float = 0.3,
    legend: bool = True,
    legend_loc: str = "upper right",
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

    if colors is None:
        colors = qf.DEFAULT_SINGLE_LINE_COLORS[: len(dataframes)]

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
        nrows=n_rows, ncols=n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows)
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
    plt.show()


def plot_grid(
    data: pd.DataFrame,
    n_cols: int = 3,
    y_name: str = "Price",
    x_name: str = "Date",
    figsize: Tuple[float, float] = (6, 2),
    ylim: Optional[Tuple[float, float]] = None,
    line_color: str = qf.DEFAULT_SINGLE_LINE_COLORS[0],
) -> None:
    """
    Plots a grid of time series plots for each unique first-level column label in a MultiIndex DataFrame.

    Args:
        data (pd.DataFrame): MultiIndex DataFrame with column format (ticker, attribute).
        n_cols (int): Number of columns in the grid layout.
        figsize (Tuple[float, float]): Size of each subplot (width, height).
        ylim (Optional[Tuple[float, float]]): Optional y-axis limits.
        line_color (str): Color of the time series line.
    """
    # Extract unique tickers from the first level of the MultiIndex
    tickers = sorted(set(col[0] for col in data.columns))
    n_rows = int(np.ceil(len(tickers) / n_cols))

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows)
    )
    axes = np.atleast_2d(axes)

    for i, ticker in enumerate(tickers):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]

        # Plot the time series for the ticker
        data[ticker].plot(
            ax=ax,
            title=ticker,
            ylabel=y_name,
            xlabel=x_name,
            grid=True,
            color=line_color,
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
    plt.show()


def plot_cdf_grid_compare(
    dataframes: List[pd.DataFrame],
    data_names: List[str],
    x_name: str = "Return",
    y_name: str = "Cumulative Probability",
    n_cols: int = 3,
    figsize: Tuple[float, float] = (6, 2),
    log_y_scale: bool = False,
    ylim: Optional[Tuple[float, float]] = None,
    colors: Optional[List[str]] = None,
    line_alpha: float = 0.7,
    x_lim: Optional[Tuple[float, float]] = None,
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

    if colors is None:
        colors = qf.DEFAULT_SINGLE_LINE_COLORS[: len(dataframes)]

    if len(colors) != len(dataframes):
        raise ValueError("Number of colors must match number of dataframes")

    # Extract unique tickers from the first level of all MultiIndex DataFrames
    tickers = sorted(
        set.intersection(*[set(col[0] for col in df.columns) for df in dataframes])
    )
    n_rows = int(np.ceil(len(tickers) / n_cols))

    fig, axes = plt.subplots(
        nrows=n_rows, ncols=n_cols, figsize=(figsize[0] * n_cols, figsize[1] * n_rows)
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
    plt.show()
