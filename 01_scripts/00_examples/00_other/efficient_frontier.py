import pandas as pd
import numpy as np
import os
import sys
import qf

# Include ../../ to access the get_data and tickers modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../")))
import matplotlib.pyplot as plt
from pypfopt import expected_returns, risk_models
from pypfopt.efficient_frontier import EfficientFrontier
from pypfopt.plotting import plot_efficient_frontier

# Import PlotManager from qf
from qf.utils.plot import PlotManager, PlotConfig, MatplotlibConfig, SaveConfig

script_dir = os.path.dirname(os.path.abspath(__file__))


# seed the random number generator for reproducibility
np.random.seed(42)


def main():
    """
    Main function to calculate and visualize the efficient frontier for a given set of assets.
    Includes Monte Carlo simulation for random portfolios and highlights optimal portfolios.
    """

    # Configure PlotManager for efficient frontier plot
    matplotlib_config = MatplotlibConfig(
        figsize=(6, 6),  # Square figure for efficient frontier
        dpi=300,
        grid=True,
        grid_alpha=0.85,
        grid_linestyle="-",
        colormap="hsv",  # For Sharpe ratio coloring
        font_size=9,
        axes_titlesize=9,
        axes_labelsize=8,
        xtick_labelsize=8,
        ytick_labelsize=8,
        legend_fontsize=8,
        pgf_enabled=True,  # Enable PGF for LaTeX output
    )

    save_config = SaveConfig(
        save_dir=script_dir,
        save_formats=["png", "pgf"],
    )

    plot_config = PlotConfig(
        matplotlib=matplotlib_config,
        save=save_config,
    )

    # Initialize PlotManager
    plot_manager = PlotManager(plot_config)

    # Step 1: Define weight bounds and tickers
    weight_bounds = (0, 1)  # Bounds for weights: between 0% and 100%
    tickers = [
        "NVDA",
        "BLDR",
        "UBER",
        "WBD",
        #   "PLTR",
        #    "TSLA",
        #     "DXCM",
    ]  # Example tickers from different sectors

    # tickers = list(set(["NVDA", "BLDR", "UBER", "WBD"]))
    risk_free_rate = 0.00
    log_returns = True
    compounding = True

    data_config = qf.DataConfig(
        tickers=tickers,
        start="2000-01-01",
        end="2025-04-01",
        indicators=("Close",),
        imputation_method="keep_nan",
        backfill_method="bfill",
        n_trading_days=365,
    )

    # Step 2: Load historical price data
    df = qf.get_data(
        data_config=data_config,
    )

    # Specify tickers for 3 assets
    tickers2 = ["GBM2", "GBM3", "GBM1"]

    # Initial prices for each asset
    S0 = np.array([100, 100, 100])

    # Specify the mean (drift) for each asset (in linear return, then exponentiate for GBM)
    mu = np.array([0.1, 0.3, 0.2])  # example means
    mu = np.log(1 + mu)

    # Specify the standard deviation (sigma) for each asset
    sigmas = np.array([0.3, 0.3, 0.3])

    # Specify the off-diagonal correlations as a vector (row-wise upper triangle, excluding diagonal)
    # For 3 assets: [rho_12, rho_13, rho_23]
    rhos = np.array([0.001, 0.001, 0.001])

    # Build the full correlation matrix from the vector
    correlations = np.eye(3)
    correlations[0, 1] = correlations[1, 0] = rhos[0]  # rho_12
    correlations[0, 2] = correlations[2, 0] = rhos[1]  # rho_13
    correlations[1, 2] = correlations[2, 1] = rhos[2]  # rho_23

    # Build the covariance matrix: Cov(X, Y) = corr(X, Y) * sigma_X * sigma_Y
    sigma = np.outer(sigmas, sigmas) * correlations

    # check if the matrix is positive definite
    assert np.all(np.linalg.eigvals(sigma) > 0)

    # Instead lets construct correlated GBM data according to the covariance matrix
    df2 = qf.data.utils.generate_multivariate_geometric_brownian_motion(
        S0=S0,
        mu=mu,  # we need to convert them to log drift for GBM
        sigma=sigma,
        tickers=tickers2,
        T=1,
        N=255,
        seed=43,
    )

    # Remove the second level of the MultiIndex (the 'Close' indicator)
    df.columns = df.columns.droplevel(1)

    # Step 3: Calculate expected returns and covariance matrix
    mu = expected_returns.mean_historical_return(
        df, log_returns=log_returns, compounding=compounding
    )  # Expected returns
    S = risk_models.sample_cov(
        df, log_returns=log_returns, compounding=compounding
    )  # Covariance matrix

    # Step 4: Create an efficient frontier object
    ef = EfficientFrontier(mu, S, weight_bounds=weight_bounds)

    # Min Volatility Portfolio
    ef_min_vol = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    ef_min_vol.min_volatility()
    ret_mv, std_mv, _ = ef_min_vol.portfolio_performance()

    # Show Tangency Portfolio
    ef_tangency = EfficientFrontier(mu, S, weight_bounds=weight_bounds)
    tangency_weights = ef_tangency.max_sharpe(risk_free_rate=risk_free_rate)
    ret_tangency, std_tangency, _ = ef_tangency.portfolio_performance(
        risk_free_rate=risk_free_rate
    )

    # Print debug information
    print(f"Risk-free rate: {risk_free_rate:.3f}")
    print(f"Tangency portfolio return: {ret_tangency:.3f}")
    print(f"Tangency portfolio volatility: {std_tangency:.3f}")
    print(
        f"Tangency portfolio Sharpe ratio: {(ret_tangency - risk_free_rate) / std_tangency:.3f}"
    )
    print(f"Expected returns:\n{mu}")
    print(f"Tangency weights: {dict(zip(tickers, tangency_weights.values()))}")

    # Step 6: Plot Efficient Frontier
    fig, ax = plot_manager.create_figure()

    # Plot individual asset points in grayscale
    for i, ticker in enumerate(df.columns):
        asset_return = mu[ticker]
        asset_vol = np.sqrt(S.loc[ticker, ticker])
        ax.scatter(
            asset_vol,
            asset_return,
            marker=".",
            s=70,
            color="0.2",
            label=ticker,
            zorder=2,
        )
        ax.annotate(
            ticker,
            (asset_vol, asset_return),
            textcoords="offset points",
            xytext=(5, -2.5),
            ha="left",
            fontsize=9,
            color="0.2",
            zorder=4,
        )

    # Plot the efficient frontier
    ax2 = plot_efficient_frontier(
        ef, ax=ax, show_assets=False, color="black", linewidth=2, linestyle="--"
    )

    # Set the color of the plot (ax) to black
    ax2.spines["bottom"].set_color("black")

    # Step 7: Monte Carlo Simulation - Generate random portfolios
    n_portfolios = 2000  # Number of random portfolios
    mc_returns = []  # List to store portfolio returns
    mc_vols = []  # List to store portfolio volatilities
    mc_sharpes = []  # List to store Sharpe ratios

    for _ in range(n_portfolios):
        # Generate random weights
        weights = np.random.dirichlet(np.ones(len(tickers)), size=1)[0]
        port_return = np.dot(weights, mu)  # Portfolio return
        port_vol = np.sqrt(
            np.dot(weights.T, np.dot(S, weights))
        )  # Portfolio volatility
        sharpe = (
            (port_return - risk_free_rate) / port_vol if port_vol > 0 else np.nan
        )  # Sharpe ratio
        mc_returns.append(port_return)
        mc_vols.append(port_vol)
        mc_sharpes.append(sharpe)

    # Plot Monte Carlo portfolios, colored by Sharpe Ratio
    cmap = plot_manager.config.matplotlib.colormap  # Use configured colormap
    sharpe_normalized = (mc_sharpes - np.nanmin(mc_sharpes)) / (
        np.nanmax(mc_sharpes) - np.nanmin(mc_sharpes)
    )
    sc = ax.scatter(
        mc_vols,
        mc_returns,
        c=mc_sharpes,
        cmap=cmap,
        s=5,
        alpha=0.6,
        label="Monte Carlo Portfolios",
        zorder=1,  # Lowest z-order, behind everything
    )
    cbar = plt.colorbar(
        sc,
        ax=ax,
        label="Sharpe Ratio",
        orientation="horizontal",
        location="bottom",
        shrink=1,
        aspect=40,
        anchor=(0.5, 0.5),
    )

    # Highlight optimal portfolios
    # ax.scatter(std_ms, ret_ms, marker="*", s=100, color="red", label="Max Sharpe Ratio")
    ax.scatter(std_mv, ret_mv, color="red", marker=5, label="Min Volatility", zorder=3)
    ax.scatter(
        std_tangency,
        ret_tangency,
        marker="*",
        s=150,  # Make it larger
        color="blue",
        label="Tangency Portfolio",
        zorder=10,  # Highest z-order to appear on top
        edgecolors="white",  # Add white edge for better visibility
        linewidth=1.5,
    )

    # Add the tangency line (Capital Market Line)
    # The line goes from risk-free rate (0, risk_free_rate) to tangency portfolio
    # Calculate the slope of the tangency line
    slope = (ret_tangency - risk_free_rate) / std_tangency if std_tangency > 0 else 0

    # Extend the line beyond the tangency portfolio for better visualization
    x_start = 0  # Risk-free rate has zero volatility
    y_start = risk_free_rate
    x_end = std_tangency * 1.8  # Extend beyond tangency portfolio
    y_end = risk_free_rate + slope * x_end  # Use calculated slope

    # Also extend backwards to show the line from risk-free rate
    x_start_extended = 0
    y_start_extended = risk_free_rate

    ax.plot(
        [x_start_extended, x_end],
        [y_start_extended, y_end],
        color="gray",
        linestyle="--",
        linewidth=1,
        alpha=0.5,
        label="Capital Market Line",
        zorder=5,  # Above Monte Carlo and assets, but below portfolio points
    )

    # Print slope for verification
    print(f"Tangency line slope: {slope:.3f}")
    print(
        f"Sharpe ratio (should equal slope): {(ret_tangency - risk_free_rate) / std_tangency:.3f}"
    )

    # Add a marker for the risk-free rate point
    ax.scatter(
        x_start,
        y_start,
        marker=4,
        color="green",
        label=f"Risk-free Rate ({risk_free_rate:.1%})",
        zorder=8,  # High z-order, but below tangency portfolio
        linewidth=1.5,
    )

    # Step 8: Customize plot
    ax.set_xlabel(
        r"$\sigma_p = \sqrt{\mathbb{V}ar[r_p]}$ (Portfolio risk)"
    )  # X-axis label
    ax.set_ylabel(r"$\mathbb{E}[r_p]$ (Expected rate of return)")  # Y-axis label

    # Legend above the plot
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.3),
        ncol=4,
        frameon=True,
    )
    # ax.legend(loc="right", bbox_to_anchor=(0.5, 0.5, 0.1, 0.5), frameon=True)
    # Grid and layout adjustments
    ax.grid(
        plot_manager.config.matplotlib.grid,
        alpha=plot_manager.config.matplotlib.grid_alpha,
        linestyle=plot_manager.config.matplotlib.grid_linestyle,
    )
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    # ax.set_xlim((0.3, 0.48))  # Start from 0 to show risk-free rate point
    # ax.set_ylim((-0.2, 0.6))  # Adjust y-axis limits

    # Step 9: Save plot using PlotManager
    plot_manager.save_figure("efficient_frontier")

    # Show plot
    plot_manager.finish()


if __name__ == "__main__":

    # Run the main function
    main()
