import numpy as np
import pandas as pd


def ekf_gbm_impute(
    df: pd.DataFrame,
    dt: float = 1.0,
    Q_mu: float = 1e-6,
    Q_sigma: float = 1e-6,
    R: float = 0.5,
    window_size: int = 60,
    n_simulations: int = 1000,
    method: str = "median",
    mode: str = "deterministic_global",
    seed: int = 42,
    stabilize: bool = True,
) -> pd.DataFrame:
    """
    Imputes missing values in price time series using an Extended Kalman Filter (EKF)
    for Geometric Brownian Motion (GBM).

    Supports:
    - 'deterministic_global': EKF with global fixed Q/R, no stochastic sampling.
    - 'stochastic_global': EKF with global fixed Q/R, stochastic sampling.
    - 'stochastic_local': EKF with local adaptive Q/R, stochastic sampling.

    Parameters
    ----------
    df : pd.DataFrame
        Time series of prices (strictly positive).
    dt : float
        Time increment (e.g. 1.0 = daily).
    Q_mu : float
        Process noise for mu.
    Q_sigma : float
        Process noise for sigma.
    R : float
        Measurement noise (used in global modes).
    window_size : int
        Window size for local Q/R estimation (only used if mode = 'stochastic_local').
    n_simulations : int
        Number of Monte Carlo samples for stochastic imputation (if applicable).
    method : str
        Aggregation method for stochastic imputation: 'median', 'mean', or 'sample'.
    mode : str
        Imputation mode: 'deterministic_global', 'stochastic_global', 'stochastic_local'.
    seed : int
        Random seed for reproducibility.
    stabilize : bool
        If True, apply numerical stabilizations (clipping sigma, Q, P, S_mat).

    Returns
    -------
    pd.DataFrame
        Imputed time series.
    """
    rng = np.random.default_rng(seed)
    imputations = pd.DataFrame(index=df.index, columns=df.columns)
    df = df.copy()

    if mode == "deterministic_global":
        _window_size = None
        _n_simulations = 0
    elif mode == "stochastic_global":
        _window_size = None
        _n_simulations = n_simulations
    elif mode == "stochastic_local":
        _window_size = window_size
        _n_simulations = n_simulations
    else:
        raise ValueError("Invalid mode")

    max_sigma = 1.0
    max_Q_logS = 0.1
    min_S_mat = 1e-6
    max_P_diag = 10.0

    for col in df.columns:
        series = df[col]
        n = len(series)
        x_hat = np.zeros((3, n))
        P = np.zeros((3, 3, n))

        first_valid_idx = series.first_valid_index()
        if first_valid_idx is None:
            imputations[col] = series
            continue

        first_idx = series.index.get_loc(first_valid_idx)
        init_price = max(series.iloc[first_idx], 1e-3)
        x_hat[:, first_idx] = [np.log(init_price), 0.0, 0.1]
        P[:, :, first_idx] = np.eye(3) * 1e-3

        imputed_series = series.copy()

        for t in range(first_idx + 1, n):
            logS_prev, mu_prev, sigma_prev = x_hat[:, t - 1]

            if stabilize:
                sigma_prev = np.clip(sigma_prev, 1e-6, max_sigma)

            f1 = logS_prev + (mu_prev - 0.5 * sigma_prev**2) * dt
            f2 = mu_prev
            f3 = sigma_prev
            x_pred = np.array([f1, f2, f3])

            F = np.array(
                [[1.0, dt, -sigma_prev * dt], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            )

            if _window_size is not None:
                t0 = max(0, t - _window_size)
                local_vals = np.log(series.iloc[t0:t].dropna())
                if len(local_vals) >= 2:
                    q_var = local_vals.diff().var()
                    Q_logS = q_var * dt if not np.isnan(q_var) else sigma_prev**2 * dt
                else:
                    Q_logS = sigma_prev**2 * dt
            else:
                Q_logS = sigma_prev**2 * dt

            if stabilize:
                Q_logS = np.clip(Q_logS, 1e-10, max_Q_logS)

            Q_mat = np.diag([Q_logS, Q_mu, Q_sigma])
            P_pred = F @ P[:, :, t - 1] @ F.T + Q_mat
            P_pred = (P_pred + P_pred.T) / 2

            if stabilize:
                P_diag = np.diag(P_pred)
                if np.any(P_diag > max_P_diag):
                    scale = max_P_diag / np.max(P_diag)
                    P_pred *= scale

            if _window_size is not None:
                residuals = series.iloc[t0:t] - np.exp(local_vals)
                r_var = residuals.var()
                R_local = (
                    r_var if len(residuals.dropna()) >= 2 and not np.isnan(r_var) else R
                )
                R_local = max(R_local, 1e-6)
            else:
                R_local = R

            if not np.isnan(series.iloc[t]):
                z_t = series.iloc[t]
                h = np.exp(x_pred[0])
                H = np.array([[h, 0.0, 0.0]])

                S_mat = H @ P_pred @ H.T + R_local
                if stabilize:
                    S_mat = max(S_mat, min_S_mat)

                K = (P_pred @ H.T) / S_mat
                y = z_t - h

                x_hat[:, t] = x_pred + (K.flatten() * y)
                P[:, :, t] = (np.eye(3) - K @ H) @ P_pred
            else:
                if _n_simulations > 0:
                    try:
                        L = np.linalg.cholesky(P_pred + 1e-10 * np.eye(3))
                    except np.linalg.LinAlgError:
                        diag_P = np.diag(np.maximum(np.diag(P_pred), 1e-10))
                        L = np.linalg.cholesky(diag_P)

                    samples = []
                    for _ in range(_n_simulations):
                        z = rng.standard_normal(3)
                        x_sample = x_pred + L @ z
                        val = np.exp(x_sample[0])
                        if not np.isnan(val) and not np.isinf(val):
                            samples.append(val)

                    if not samples:
                        imputed_value = np.exp(x_pred[0])
                    else:
                        if method == "median":
                            imputed_value = np.median(samples)
                        elif method == "mean":
                            imputed_value = np.mean(samples)
                        elif method == "sample":
                            imputed_value = rng.choice(samples)
                        else:
                            raise ValueError("Invalid method")
                else:
                    imputed_value = np.exp(x_pred[0])

                imputed_series.iloc[t] = imputed_value
                x_hat[:, t] = x_pred
                P[:, :, t] = P_pred

        imputations[col] = imputed_series

    return imputations
