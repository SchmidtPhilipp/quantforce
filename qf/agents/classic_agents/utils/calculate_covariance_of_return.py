import numpy as np
import pandas as pd


def calculate_covariance_of_return(
    returns,
    returns_are_log=True,
    trading_days=252,
    linearize=True,
    percentage=True,
    calculate_using_log_returns=False, 
    as_std_matrix=False
):
    """
    Berechnet die jährliche Kovarianzmatrix aus gegebenen Tagesrenditen.

    :param returns: DataFrame mit Tagesrenditen.
    :param returns_are_log: Gibt an, ob die Renditen logarithmisch sind.
    :param trading_days: Anzahl der Handelstage pro Jahr.
    :param linearize: Ob die Kovarianzmatrix auf lineare Renditen transformiert werden soll.
    :param percentage: Ob das Ergebnis in Prozentpunkten² ausgegeben wird.
    :return: DataFrame der annualisierten Kovarianzmatrix.
    """
    if returns_are_log:
        returns = returns.apply(lambda x: np.exp(x) - 1)

    if calculate_using_log_returns:
        returns = np.log(1 + returns)

    cov_log = returns.cov() * trading_days
    mu_log = returns.mean() * trading_days

    if linearize:
        mu_i = mu_log.values.reshape(-1, 1)
        mu_j = mu_log.values.reshape(1, -1)

        sigma_i = np.diag(cov_log).reshape(-1, 1)
        sigma_j = np.diag(cov_log).reshape(1, -1)

        cov_lin = (np.exp(cov_log.values) - 1) * np.exp(mu_i + mu_j + 0.5 * (sigma_i + sigma_j))
        cov = pd.DataFrame(cov_lin, index=returns.columns, columns=returns.columns)
    else:
        cov = cov_log

    if as_std_matrix:
        cov = np.sqrt((cov))

    if percentage:
        cov *= 100

    return cov