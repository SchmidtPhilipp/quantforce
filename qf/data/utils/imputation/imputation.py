import numpy as np
import pandas as pd

from qf.data.utils.imputation.extended_kalman import ekf_gbm_impute
from qf.data.utils.imputation.gbm_bridge import gbm_bridge_impute
from qf.data.utils.imputation.kalman import linear_kalman_impute_global
from qf.data.utils.imputation.least_squares import (
    lse_linear_impute_global,
    lse_loglinear_impute_global,
)


def impute(
    data: pd.DataFrame, imputation_method: str, start=None, end=None, **kwargs
) -> pd.DataFrame:
    """
    Führt die Imputation von fehlenden Werten in einem DataFrame mit verschiedenen Methoden durch.

    Diese Funktion bietet eine Vielzahl von einfachen und fortgeschrittenen Imputationsverfahren,
    darunter lineare und nichtlineare Interpolationen, Kalman- und Extended-Kalman-Filter, GBM-basierte
    Monte-Carlo- und Brückenmethoden sowie Least-Squares-Schätzungen.

    Parameter
    ----------
    data : pd.DataFrame
        Eingabedaten mit fehlenden Werten (NaN), die imputiert werden sollen.
    imputation_method : str
        Name der Imputationsmethode. Unterstützte Methoden sind:
        'ffill', 'linear_interpolation', 'log_interpolation', 'spline_interpolation',
        'polynomial_interpolation', 'quadratic_interpolation', 'cubic_interpolation',
        'keep_nan', 'shrinkage', 'linear_kalman_global', 'linear_kalman_velocity_global',
        'ekf_gbm_deterministic_global', 'ekf_gbm_stochastic_global', 'ekf_gbm_stochastic_local',
        'gbm_monte_carlo_unconstrained_end_median', 'gbm_monte_carlo_unconstrained_end_sample',
        'gbm_monte_carlo_unconstrained_end_mean', 'gbm_bridge_impute_with_unit_variance',
        'gbm_bridge_impute_with_global_variance', 'gbm_bridge_impute_with_local_variance',
        'gbm_bridge_impute_with_rolling_variance', 'lse_linear_impute_global',
        'lse_loglinear_impute_global'.
    start : optional
        Optionaler Startindex für die Imputation (wird aktuell nicht verwendet).
    end : optional
        Optionaler Endindex für die Imputation (wird aktuell nicht verwendet).
    **kwargs : dict
        Zusätzliche Parameter, die an die jeweilige Imputationsmethode weitergegeben werden.
        Beispiele: 'n_simulations', 'window_size', 'velocity', etc.

    Rückgabe
    -------
    pd.DataFrame
        DataFrame mit imputierten Werten entsprechend der gewählten Methode.

    Raises
    ------
    ValueError
        Falls eine unbekannte Imputationsmethode gewählt wird oder erforderliche Parameter fehlen.
    """
    method = "median"
    data = data.copy()

    ## Simple Imputation Methods
    if imputation_method == "ffill":
        data = data.ffill()
    elif imputation_method == "linear_interpolation":
        data = data.interpolate(method="linear")
    elif imputation_method == "log_interpolation":
        data = np.log(data)
        data = data.interpolate(method="linear")
        data = np.exp(data)
    elif imputation_method == "spline_interpolation":
        data = data.interpolate(method="spline", order=2)
    elif imputation_method == "polynomial_interpolation":
        data = data.interpolate(method="polynomial", order=2)
    elif imputation_method == "quadratic_interpolation":
        data = data.interpolate(method="quadratic", order=2)
    elif imputation_method == "cubic_interpolation":
        data = data.interpolate(method="cubic", order=2)
    elif imputation_method == "keep_nan":
        pass
    elif imputation_method == "shrinkage":
        # we remove the columns with a single nan value
        data = data.dropna(axis=1, how="all")

    ## Advanced Imputation Methods

    ####### State Space Methods / Predictive Methods #######
    ## Kalman Methods
    elif imputation_method == "linear_kalman_global":
        data = linear_kalman_impute_global(data, **kwargs)
    elif imputation_method == "linear_kalman_velocity_global":
        data = linear_kalman_impute_global(data, velocity=True, **kwargs)

    ## EKF Methods
    elif imputation_method == "ekf_gbm_deterministic_global":
        data = ekf_gbm_impute(data, mode="deterministic_global", **kwargs)
    elif imputation_method == "ekf_gbm_stochastic_global":
        if "n_simulations" not in kwargs:
            raise ValueError(
                "n_simulations must be provided for stochastic_global mode"
            )
        data = ekf_gbm_impute(data, mode="stochastic_global", **kwargs)
    elif imputation_method == "ekf_gbm_stochastic_local":
        if "window_size" not in kwargs:
            raise ValueError("window_size must be provided for stochastic_local mode")
        if "n_simulations" not in kwargs:
            raise ValueError("n_simulations must be provided for stochastic_local mode")
        data = ekf_gbm_impute(data, mode="stochastic_local", **kwargs)

    ## GBM Methods
    elif (
        imputation_method == "gbm_monte_carlo_unconstrained_end_median"
    ):  # method="median"
        data = gbm_bridge_impute(data, method=method, variance_mode="forward", **kwargs)
    elif imputation_method == "gbm_monte_carlo_unconstrained_end_sample":
        data = gbm_bridge_impute(data, method=method, variance_mode="forward", **kwargs)
    elif imputation_method == "gbm_monte_carlo_unconstrained_end_mean":
        data = gbm_bridge_impute(data, method=method, variance_mode="forward", **kwargs)

    # Bridge Methods
    elif imputation_method == "gbm_bridge_impute_with_unit_variance":
        data = gbm_bridge_impute(
            data, method=method, variance_mode="unit_variance", **kwargs
        )
    elif imputation_method == "gbm_bridge_impute_with_global_variance":
        data = gbm_bridge_impute(data, method=method, variance_mode="global", **kwargs)
    elif imputation_method == "gbm_bridge_impute_with_local_variance":
        data = gbm_bridge_impute(data, method=method, variance_mode="local", **kwargs)
    elif imputation_method == "gbm_bridge_impute_with_rolling_variance":
        data = gbm_bridge_impute(data, method=method, variance_mode="rolling", **kwargs)

    ## LSE Methods
    elif imputation_method == "lse_linear_impute_global":
        data = lse_linear_impute_global(data, **kwargs)
    elif imputation_method == "lse_loglinear_impute_global":
        data = lse_loglinear_impute_global(data, **kwargs)

    else:
        raise ValueError(
            f"Unknown imputation method: {imputation_method}. "
            + "Available methods: "
            + "ffill, linear_interpolation, log_interpolation, spline_interpolation, polynomial_interpolation, quadratic_interpolation, cubic_interpolation, keep_nan, shrinkage, linear_kalman_global, linear_kalman_velocity_global, ekf_gbm_deterministic_global, ekf_gbm_stochastic_global, ekf_gbm_stochastic_local, gbm_monte_carlo_unconstrained_end_median, gbm_monte_carlo_unconstrained_end_sample, gbm_monte_carlo_unconstrained_end_mean, gbm_bridge_impute_with_unit_variance, gbm_bridge_impute_with_global_variance, gbm_bridge_impute_with_local_variance, gbm_bridge_impute_with_rolling_variance, lse_linear_impute_global, lse_loglinear_impute_global"
        )

    return data
