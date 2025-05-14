import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates
import matplotlib


def setup_pgf():
    """
    Set up PGF backend for matplotlib to export figures in LaTeX-compatible format.
    """
    # Use PGF backend
    matplotlib.use('pgf')

    # PGF export for LaTeX
    plt.rcParams.update({
        "pgf.texsystem": "pdflatex",
        "figure.figsize": [6.6, 2.3],
        "font.family": "helvet",
        "text.usetex": True,
        "pgf.rcfonts": False,
        "axes.formatter.use_mathtext": True,  
        "pgf.preamble": "\n".join([
            r"\usepackage{amsmath}",
            r"\usepackage{amssymb}",
            r"\usepackage{mathpazo}"  # Mathpazo als Schriftart
        ]),
        "font.size": 12,  # Allgemeine Schriftgröße
        "axes.titlesize": 14,  # Titel der Achsen
        "axes.labelsize": 12,  # Beschriftung der Achsen
        "xtick.labelsize": 9,  # Schriftgröße der X-Achsenticks
        "ytick.labelsize": 9,  # Schriftgröße der Y-Achsenticks
        "legend.fontsize": 10,  # Schriftgröße der Legend"
        "axes.grid": True,  # Gitterlinien aktivieren
        "grid.alpha": 0.5,  # Transparenz der Gitterlinien
        "grid.linestyle": "--",  # Linienstil der Gitterlinien
        #"grid.linewidth": 0.5,  # Linienstärke der Gitterlinien
    })
    plt.rcParams['text.latex.preamble'] = r'\newcommand{\mathdefault}[1][]{}' # Setze den Befehl für mathdefault

def reset_pgf():
    plt.rcdefaults()  # Standardwerte für rcParams
    matplotlib.use('macosx')  # Setzt das Backend auf das Standard-Backend zurück

def plot_lines_grayscale(
    df: pd.DataFrame,
    xlabel: str = "Date",
    ylabel: str = "Y",
    title: str = "",
    filename: str = "plot_output",
    save_dir: str = "tmp/plots",
    max_xticks: int = 12,
    y_limits: tuple[float, float] | None = None,
    figsize: tuple[float, float] = (10, 2.5),
    linewidth: float = 2.0
):
    """
    Plot multiple lines from a DataFrame in grayscale with maximally spaced intensities.
    Detects time-based x-axis and formats accordingly.
    Optionally applies fixed y-limits, otherwise snaps to next grid tick.
    Replaces underscores in legend labels and converts them to uppercase.
    Exports as PGF and PNG.
    """
    setup_pgf()  # Set up PGF backend for LaTeX compatibility
    os.makedirs(save_dir, exist_ok=True)

    n_lines = len(df.columns)
    x = df.index

    # Ensure the index is datetime if it looks like dates
    if not pd.api.types.is_datetime64_any_dtype(x):
        try:
            x = pd.to_datetime(x)
            df = df.set_index(x)
        except Exception as e:
            raise ValueError("Index is not datetime and cannot be converted.") from e

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
    ax.set_xticks(x[::max(1, len(x) // max_xticks)])  # Set x-ticks to max_xticks

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
    y_limits_right: tuple[float, float] = None
):
    """
    Plot two lines from a DataFrame using left and right Y-axes.
    Ensures the same number of Y-ticks on both sides.
    Positions legends outside the plot.
    """
    setup_pgf()  # Set up PGF backend for LaTeX compatibility
    if len(df.columns) != 2:
        raise ValueError("The DataFrame must contain exactly two columns for dual-axis plotting.")

    os.makedirs(save_dir, exist_ok=True)

    # Limit the number of entries if max_entries is specified
    if max_entries is not None and len(df) > max_entries:
        step = max(1, len(df) // max_entries)
        df = df.iloc[::step]

    x = df.index
    left_label, right_label = df.columns

    fig, ax_left = plt.subplots()

    # Plot the first line on the left Y-axis
    ax_left.plot(x, df[left_label], label=left_label.replace("_", " ").upper(), color="black", linewidth=linewidth)
    ax_left.set_ylabel(ylabel_left)
    ax_left.tick_params(axis="y", labelcolor="black")

    # Create the right Y-axis
    ax_right = ax_left.twinx()
    ax_right.plot(x, df[right_label], label=right_label.replace("_", " ").upper(), color="gray", linewidth=linewidth)
    ax_right.set_ylabel(ylabel_right)
    ax_right.tick_params(axis="y", labelcolor="gray")
    # Set the right Y-axis label and color
    ax_right.set_ylabel(ylabel_right, color="gray")  # Text in der gleichen Farbe wie die Linie
    ax_right.tick_params(axis="y", labelcolor="gray")  # Achsenticks in der gleichen Farbe

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
    plt.grid(which='both', linestyle='--', linewidth=0.5)

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
