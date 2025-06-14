import numpy as np
import pandas as pd


def calculate_expected_return(
    returns,
    returns_are_log=True,
    trading_days=252,
    calculate_using_log_returns=False,
    compounding=True,
    percentage=True
):
    """
    Berechnet die erwartete jährliche Rendite aus gegebenen (log oder linearen) Tagesrenditen.

    :param returns: DataFrame mit Tagesrenditen.
    :param returns_are_log: Gibt an, ob die Renditen logarithmisch sind.
    :param trading_days: Anzahl der Handelstage pro Jahr.
    :param compounding: Ob geometrisches Mittel verwendet werden soll.
    :param percentage: Ob das Ergebnis in Prozent angegeben werden soll.
    :return: Series der erwarteten jährlichen Rendite pro Asset.
    """
    #returns = in_returns.copy()

    # Umwandlung der Renditen falls in logarthmischer Form gegeben. 
    if returns_are_log:
        returns = returns.apply(lambda x: np.exp(x) - 1)

    if calculate_using_log_returns:
       returns = np.log(1 + returns)

    if compounding:
        mu = (1 + returns).prod() ** (trading_days / returns.count()) - 1
    else:
        mu = returns.mean() * trading_days

    if percentage:
        mu *= 100

    return mu