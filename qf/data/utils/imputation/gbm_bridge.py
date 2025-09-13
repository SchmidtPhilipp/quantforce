import numpy as np
import pandas as pd


def gbm_bridge_impute(
    df: pd.DataFrame,
    n_simulations: int = 1000,
    seed: int = 42,
    method: str = "median",
    variance_mode: str = "unit_variance",
    window_size: int = 60,
) -> pd.DataFrame:
    """
    Imputes missing values in time series using a Geometric Brownian Motion (GBM) bridge or forward simulation.

    This function generates Monte Carlo GBM simulations to fill gaps in the time series.
    The behavior depends on the chosen variance_mode.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the time series (strictly positive prices).
    n_simulations : int
        Number of Monte Carlo simulation paths per gap.
    seed : int
        Random seed for reproducibility.
    method : str
        Aggregation method for the simulation results:
            - 'mean' : Fill missing values with the mean of all simulated paths.
            - 'median' : Fill missing values with the median of all simulated paths.
            - 'sample' : Fill missing values with a randomly selected simulated path.
    variance_mode : str
        Variance and drift estimation mode:
            - 'unit_variance' : Pure Brownian bridge with unit variance (no scaling or drift).
            - 'global' : GBM bridge using mu and sigma estimated from all log returns of the series.
            - 'local' : Bridge using variance estimated from log returns within a window around the gap (before and after).
            - 'rolling' : Bridge using variance estimated from log returns in a window of past data only (before the gap).
            - 'forward' : GBM forward simulation from the last known value before the gap (no fixed end point).
    window_size : int
        Size of the window (in number of data points) used for local or rolling variance estimation.
        Only relevant if variance_mode is 'local' or 'rolling'.

    Returns
    -------
    pd.DataFrame
        A DataFrame where missing values have been imputed using GBM simulations.
    """
    rng = np.random.default_rng(seed)
    df_imputed = df.copy()

    for col in df.columns:  # For every asset
        series = df[col]
        is_nan = series.isna()

        # Skip the calculation if none of the entries is nan.
        if is_nan.sum() == 0:
            continue

        # Get the logreturns
        series_without_nan = series.dropna()
        log_returns = np.log(series_without_nan / series_without_nan.shift(1)).dropna()

        # Estimate the global mu and sigma
        mu_global = log_returns.mean()
        sigma_global = log_returns.std()
        sigma_global = max(sigma_global, 1e-6)

        # ---
        # Groups consecutive indices with missing values (NaN) in the time series.
        # For each contiguous NaN gap, a list of indices is created and stored in 'groups'.
        # Example: [1,2,3] for a gap of three consecutive NaNs.
        # These groups are later used for imputation.
        groups = []
        current = []
        for i in series.index:
            if is_nan.loc[i]:
                current.append(i)
            elif current:
                groups.append(current)
                current = []
        if current:
            groups.append(current)

        for group in groups:
            try:
                start_idx = series.index.get_loc(
                    series.loc[: group[0]].last_valid_index()
                )
                end_idx = series.index.get_loc(
                    series.loc[group[-1] :].first_valid_index()
                )
            except (KeyError, AttributeError):
                continue

            S0 = series.iloc[start_idx]
            if S0 <= 0:
                continue

            n = len(group)
            t_steps = np.linspace(0, 1, n + 2)[1:-1]
            step_count = end_idx - start_idx

            # Variance Estimation
            if variance_mode == "unit_variance":
                mu = 0
                sigma = 1
            elif variance_mode == "global":
                mu = mu_global
                sigma = sigma_global
            elif variance_mode == "local":
                window_start = max(0, start_idx - window_size)
                window_end = min(len(series), end_idx + window_size)
                window_series = np.log(series.iloc[window_start:window_end].dropna())
                local_var = window_series.diff().var()
                sigma = np.sqrt(max(local_var, 1e-10))
                mu = 0
            elif variance_mode == "rolling":
                window_start = max(0, start_idx - window_size)
                window_end = start_idx
                window_series = np.log(series.iloc[window_start:window_end].dropna())
                local_var = window_series.diff().var()
                sigma = np.sqrt(max(local_var, 1e-10))
                mu = 0
            elif variance_mode == "forward":
                mu = mu_global
                sigma = sigma_global
            else:
                raise ValueError("Invalid variance_mode")

            # Simulation
            simulations = np.zeros((n_simulations, n))
            for sim in range(n_simulations):
                if variance_mode == "forward":
                    # Pure GBM forward simulation without fixed end
                    z = rng.standard_normal(n)
                    dt = 1.0  # You can adjust this to your time scale if needed
                    log_increments = (mu - 0.5 * sigma**2) * dt + sigma * z * np.sqrt(
                        dt
                    )
                    sim_log = np.log(S0) + np.cumsum(log_increments)
                    simulations[sim, :] = np.exp(sim_log)
                else:
                    # Bridge modes
                    z = rng.standard_normal(n)
                    B = np.cumsum(z) * np.sqrt(1 / n)
                    bridge = B - t_steps * B[-1]

                    if variance_mode in ["local", "rolling"]:
                        bridge_scaled = bridge * sigma
                        drift = (1 - t_steps) * np.log(S0) + t_steps * np.log(
                            series.iloc[end_idx]
                        )
                        # drift += (mu - 0.5 * sigma ** 2) * (t_steps * step_count)
                    elif variance_mode == "global":
                        bridge_scaled = bridge * sigma_global
                        drift = (1 - t_steps) * np.log(S0) + t_steps * np.log(
                            series.iloc[end_idx]
                        )
                        # drift += (mu - 0.5 * sigma ** 2) * (t_steps * step_count)
                    else:
                        bridge_scaled = bridge
                        drift = (1 - t_steps) * np.log(S0) + t_steps * np.log(
                            series.iloc[end_idx]
                        )
                        # drift += (mu - 0.5 * sigma ** 2) * (t_steps * step_count)

                    sim_log = drift + bridge_scaled
                    simulations[sim, :] = np.exp(sim_log)

            if method == "mean":
                fill_values = simulations.mean(axis=0)
            elif method == "median":
                fill_values = np.median(simulations, axis=0)
            elif method == "sample":
                fill_values = simulations[rng.integers(0, n_simulations), :]
            else:
                raise ValueError("Invalid method")

            for i, date in enumerate(group):
                series.loc[date] = fill_values[i]

        df_imputed[col] = series

    return df_imputed
