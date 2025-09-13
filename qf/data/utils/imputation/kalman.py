import numpy as np
import pandas as pd


def _linear_kalman_1d_impute(
    series: pd.Series, Q: float, R: float, use_log: bool = True
) -> pd.Series:
    """
    Führt 1D-Kalman-Imputation für eine Serie durch.
    - Q: Prozessrauschen
    - R: Messrauschen
    - use_log: Falls True, arbeitet der Filter im Lograum
    """
    series = series.copy()
    n = len(series)
    x_hat = np.zeros(n)
    P = np.zeros(n)

    # Initialisierung
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        return series  # keine gültigen Werte vorhanden

    first_idx = series.index.get_loc(first_valid_idx)

    if use_log:
        if series.iloc[first_idx] <= 0:
            raise ValueError(
                f"Erster gültiger Preis {series.iloc[first_idx]} muss > 0 sein für log-Transformation."
            )
        x_hat[first_idx] = np.log(series.iloc[first_idx])
    else:
        x_hat[first_idx] = series.iloc[first_idx]

    P[first_idx] = 1.0

    for t in range(first_idx + 1, n):
        # PREDICT
        x_hat[t] = x_hat[t - 1]
        P[t] = P[t - 1] + Q

        if not np.isnan(series.iloc[t]):
            if use_log:
                if series.iloc[t] <= 0:
                    raise ValueError(
                        f"Ungültiger Preis {series.iloc[t]} bei Index {series.index[t]} für log-Transformation."
                    )
                z_t = np.log(series.iloc[t])
            else:
                z_t = series.iloc[t]

            # Kalman Gain
            K = P[t] / (P[t] + R)

            # UPDATE
            x_hat[t] += K * (z_t - x_hat[t])
            P[t] *= 1 - K

    # IMPUTATION
    for t in range(n):
        if np.isnan(series.iloc[t]):
            if use_log:
                series.iloc[t] = np.exp(x_hat[t])
            else:
                series.iloc[t] = x_hat[t]

    return series


def _linear_kalman_1d_velocity_impute(
    series: pd.Series, Q: float, R: float, use_log: bool = False
) -> pd.Series:
    """
    Kalman-Imputation mit Preis + Geschwindigkeit (konstante Geschwindigkeit Modell).
    - Q: Prozessrauschen (auf das Prozessrauschen der Geschwindigkeit bezogen)
    - R: Messrauschen (Messfehler des Preises)
    - use_log: Falls True, arbeitet der Filter im Lograum
    """
    series = series.copy()
    n = len(series)

    # Zustandsvektor: [Preis, Geschwindigkeit]
    x_hat = np.zeros((2, n))
    P = np.zeros((2, 2, n))  # Fehlerkovarianz

    # Systemmatrix (Delta t = 1 angenommen)
    A = np.array([[1, 1], [0, 1]])

    # Beobachtungsmatrix: Wir messen nur den Preis
    H = np.array([[1, 0]])

    # Prozessrauschkovarianz (hier nur Geschwindigkeit rauscht)
    Q_mat = Q * np.array([[0.25, 0.5], [0.5, 1]])

    # Messrauschkovarianz
    R_mat = np.array([[R]])

    # Initialisierung
    first_valid_idx = series.first_valid_index()
    if first_valid_idx is None:
        return series  # keine gültigen Werte vorhanden

    first_idx = series.index.get_loc(first_valid_idx)

    if use_log:
        if series.iloc[first_idx] <= 0:
            raise ValueError(
                f"Erster gültiger Preis {series.iloc[first_idx]} muss > 0 sein für log-Transformation."
            )
        price_0 = np.log(series.iloc[first_idx])
    else:
        price_0 = series.iloc[first_idx]

    x_hat[:, first_idx] = [price_0, 0.0]  # Anfangsgeschwindigkeit 0
    P[:, :, first_idx] = np.eye(2)  # Anfangsunsicherheit

    for t in range(first_idx + 1, n):
        # PREDICT
        x_pred = A @ x_hat[:, t - 1]
        P_pred = A @ P[:, :, t - 1] @ A.T + Q_mat

        if not np.isnan(series.iloc[t]):
            if use_log:
                if series.iloc[t] <= 0:
                    raise ValueError(
                        f"Ungültiger Preis {series.iloc[t]} bei Index {series.index[t]} für log-Transformation."
                    )
                z_t = np.array([[np.log(series.iloc[t])]])
            else:
                z_t = np.array([[series.iloc[t]]])

            # Kalman Gain
            S = H @ P_pred @ H.T + R_mat
            K = P_pred @ H.T @ np.linalg.inv(S)

            # UPDATE
            y = z_t - (H @ x_pred)  # Innovation
            x_hat[:, t] = x_pred + (K @ y).flatten()
            P[:, :, t] = (np.eye(2) - K @ H) @ P_pred
        else:
            # Keine Messung → nur Vorhersage
            x_hat[:, t] = x_pred
            P[:, :, t] = P_pred

    # IMPUTATION
    for t in range(n):
        if np.isnan(series.iloc[t]):
            if use_log:
                series.iloc[t] = np.exp(x_hat[0, t])
            else:
                series.iloc[t] = x_hat[0, t]

    return series


def linear_kalman_impute_global(
    df: pd.DataFrame,
    auto_estimate: bool = True,
    default_Q: float = 1e-5,
    default_R: float = 0.01,
    use_log: bool = True,
    velocity: bool = True,
) -> pd.DataFrame:
    """
    Kalman-Imputation mit GLOBALER Q/R-Schätzung je Spalte.
    Q = Prozessrauschen, R = Messrauschen.
    Für jede Spalte werden Q/R geschätzt oder Default-Werte genutzt.
    Dann wird die Kalman-Filter-Gleichung angewendet.
    Das Ergebnis ist eine DataFrame mit den imputierten Werten.
    """

    def impute_column(col):
        Q = _estimate_process_noise(col) if auto_estimate else default_Q
        R = _estimate_measurement_noise(col) if auto_estimate else default_R
        if velocity:
            return _linear_kalman_1d_velocity_impute(col, Q, R, use_log=use_log)
        else:
            return _linear_kalman_1d_impute(col, Q, R, use_log=use_log)

    return df.apply(impute_column, axis=0)


def _estimate_process_noise(series: pd.Series) -> float:
    diffs = series.dropna().diff().dropna()
    return np.var(diffs)


def _estimate_measurement_noise(series: pd.Series) -> float:
    residuals = series.dropna() - series.dropna().rolling(window=3, center=True).mean()
    residuals = residuals.dropna()
    return np.var(residuals)
