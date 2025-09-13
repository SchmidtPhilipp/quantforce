import matplotlib.pyplot as plt
from qf.utils.plot.plot_config import (
    PlotConfig,
    ConfidenceIntervalConfig,
    LinePlotConfig,
    DualAxisConfig,
    GridPlotConfig,
    SaveConfig,
    MatplotlibConfig,
)
from qf.utils.plot.plot_manager import PlotManager


# Convenience function for Jupyter notebooks
def setup_jupyter_plotting():
    """Set up matplotlib with Jupyter-optimized configuration."""
    import matplotlib.pyplot as plt

    # Create full Jupyter-optimized configuration (includes save settings)
    jupyter_config = PlotConfig.jupyter()

    # Create plot manager with Jupyter config
    plot_manager = PlotManager(jupyter_config)

    # Apply the configuration
    plot_manager.setup_jupyter()

    return plot_manager


def get_jupyter_config() -> MatplotlibConfig:
    """Get a Jupyter-optimized configuration."""
    return PlotConfig.jupyter_config()


# Convenience function for plot_grid
def plot_grid(
    data, n_cols=3, y_name=None, x_name=None, figsize=None, ylim=None, line_color=None
):
    """Convenience function to create a grid plot using Jupyter-optimized settings."""
    # Create full Jupyter-optimized configuration (includes save settings)
    jupyter_config = PlotConfig.jupyter()

    # Create plot manager with Jupyter config
    plot_manager = PlotManager(jupyter_config)

    # Get the figure from the plot manager
    fig = plot_manager.plot_grid(
        data, n_cols, y_name, x_name, figsize, ylim, line_color
    )

    # Return the figure for Jupyter display
    return fig


# Convenience function for plot_lines_grayscale
def plot_lines_grayscale(
    df,
    x_axis=None,
    xlabel="Date",
    ylabel="Y",
    title="",
    filename="plot_output",
    max_xticks=12,
    y_limits=None,
    figsize=None,
    linewidth=None,
):
    """Convenience function to create a grayscale line plot using Jupyter-optimized settings."""
    # Create full Jupyter-optimized configuration (includes save settings)
    jupyter_config = PlotConfig.jupyter()

    # Create plot manager with Jupyter config
    plot_manager = PlotManager(jupyter_config)

    # Get the figure from the plot manager
    fig = plot_manager.plot_lines_grayscale(
        df,
        x_axis,
        xlabel,
        ylabel,
        title,
        filename,
        max_xticks,
        y_limits,
        figsize,
        linewidth,
    )

    # Return the figure for Jupyter display
    return fig


# Convenience function for plot_risk_matrix
def plot_risk_matrix(
    expected_returns,
    expected_covariance,
    colorscheme=None,
    figsize=None,
    fontsize=8,
    filename="risk_matrix",
    title="Risk Matrix",
):
    """Convenience function to create a risk matrix plot using Jupyter-optimized settings."""
    # Create full Jupyter-optimized configuration (includes save settings)
    jupyter_config = PlotConfig.jupyter()

    # Create plot manager with Jupyter config
    plot_manager = PlotManager(jupyter_config)

    # Get the figure from the plot manager
    fig = plot_manager.plot_risk_matrix(
        expected_returns,
        expected_covariance,
        colorscheme,
        figsize,
        fontsize,
        filename,
        title,
    )

    # Return the figure for Jupyter display
    return fig


# Convenience function for plot_hist_grid_compare
def plot_hist_grid_compare(
    dataframes,
    data_names,
    plot_kde=True,
    plot_hist=False,
    x_name="Return",
    y_name="Probability Density",
    n_cols=None,
    bins=50,
    figsize=None,
    log_y_scale=False,
    ylim=None,
    colors=None,
    markers=None,
    hist_alpha=0.3,
    legend=True,
    legend_loc="upper right",
    filename="hist_grid_compare",
):
    """Convenience function to create a histogram grid comparison plot using Jupyter-optimized settings."""
    # Create full Jupyter-optimized configuration (includes save settings)
    jupyter_config = PlotConfig.jupyter()

    # Create plot manager with Jupyter config
    plot_manager = PlotManager(jupyter_config)

    # Get the figure from the plot manager
    fig = plot_manager.plot_hist_grid_compare(
        dataframes,
        data_names,
        plot_kde,
        plot_hist,
        x_name,
        y_name,
        n_cols,
        bins,
        figsize,
        log_y_scale,
        ylim,
        colors,
        markers,
        hist_alpha,
        legend,
        legend_loc,
        filename,
    )

    # Return the figure for Jupyter display
    return fig


# Convenience function for plot_cdf_grid_compare
def plot_cdf_grid_compare(
    dataframes,
    data_names,
    x_name="Return",
    y_name="Cumulative Probability",
    n_cols=None,
    figsize=None,
    log_y_scale=False,
    ylim=None,
    colors=None,
    line_alpha=0.7,
    x_lim=None,
    filename="cdf_grid_compare",
):
    """Convenience function to create a CDF grid comparison plot using Jupyter-optimized settings."""
    # Create full Jupyter-optimized configuration (includes save settings)
    jupyter_config = PlotConfig.jupyter()

    # Create plot manager with Jupyter config
    plot_manager = PlotManager(jupyter_config)

    # Get the figure from the plot manager
    fig = plot_manager.plot_cdf_grid_compare(
        dataframes,
        data_names,
        x_name,
        y_name,
        n_cols,
        figsize,
        log_y_scale,
        ylim,
        colors,
        line_alpha,
        x_lim,
        filename,
    )

    # Return the figure for Jupyter display
    return fig


__all__ = [
    "PlotConfig",
    "PlotManager",
    "ConfidenceIntervalConfig",
    "LinePlotConfig",
    "DualAxisConfig",
    "GridPlotConfig",
    "SaveConfig",
    "MatplotlibConfig",
    "setup_jupyter_plotting",
    "get_jupyter_config",
    "plot_grid",
    "plot_lines_grayscale",
    "plot_risk_matrix",
    "plot_hist_grid_compare",
    "plot_cdf_grid_compare",
]
