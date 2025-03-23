import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as mdates


# PGF export for LaTeX
plt.rcParams.update({
    "pgf.texsystem": "pdflatex",
    "font.family": "serif",
    "text.usetex": True,
    "pgf.rcfonts": False
})


def plot_lines_grayscale(
    df: pd.DataFrame,
    xlabel: str = "Date",
    ylabel: str = "Y",
    title: str = "",
    filename: str = "plot_output",
    save_dir: str = "tmp/plots",
    max_xticks: int = 12,
    y_limits: tuple[float, float] | None = None
):
    """
    Plot multiple lines from a DataFrame in grayscale with maximally spaced intensities.
    Detects time-based x-axis and formats accordingly.
    Optionally applies fixed y-limits, otherwise snaps to next grid tick.
    Replaces underscores in legend labels and converts them to uppercase.
    Exports as PGF and PNG.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_lines = len(df.columns)
    x = df.index

    grays = np.linspace(0.25, 1.0, n_lines)
    colors = [str(1 - g) for g in grays]

    fig, ax = plt.subplots(figsize=(20, 5))

    for i, column in enumerate(df.columns):
        label = str(column).replace("_", " ").upper()
        ax.plot(x, df[column], label=label, color=colors[i], linewidth=2)

    is_time = pd.api.types.is_datetime64_any_dtype(df.index)
    if is_time:
        ax.xaxis.set_major_locator(mdates.AutoDateLocator(maxticks=max_xticks))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b %Y"))
        fig.autofmt_xdate()
    else:
        if len(x) > max_xticks:
            step = max(1, len(x) // max_xticks)
            ax.set_xticks(x[::step])

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

    # X-Limits snapping (only for numeric X)
    if not is_time:
        x_ticks = ax.get_xticks()
        x_min_data, x_max_data = ax.get_xlim()
        x_lower_ticks = x_ticks[x_ticks <= x_min_data]
        x_upper_ticks = x_ticks[x_ticks >= x_max_data]
        if len(x_lower_ticks) > 0 and len(x_upper_ticks) > 0:
            ax.set_xlim(x_lower_ticks[0], x_upper_ticks[-1])

    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    plt.tight_layout(rect=[0, 0, 0.85, 1])
    plt.grid(True)

    pgf_path = os.path.join(save_dir, f"{filename}.pgf")
    png_path = os.path.join(save_dir, f"{filename}.png")
    fig.savefig(pgf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)

    #print(f"✅ Plot saved to {pgf_path} and {png_path}")
