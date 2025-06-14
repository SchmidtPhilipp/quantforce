import numpy as np
from qf import TimeBasedDataset  # Importiere das TimeBasedDataset
import matplotlib.pyplot as plt
from qf.utils.plot import plot_risk_matrix
plt.rcParams["text.usetex"] = True


def calculate_historical_correlation(dataset, tickers, window_size=60):
    """
    Berechnet die historischen Korrelationen zwischen zwei Tickersymbolen.

    Parameters:
        dataset (TimeBasedDataset): Das Dataset mit den historischen Daten.
        tickers (list): Liste der Tickersymbole (z. B. ["AAPL", "MSFT"]).
        window_size (int): Größe des gleitenden Fensters für die Korrelation.

    Returns:
        list: Liste der historischen Korrelationen.
    """
    # Index der Tickersymbole im Dataset finden
    ticker_indices = [dataset.tickers.index(ticker) for ticker in tickers]

    # Renditen berechnen
    data = dataset.data  # Pandas DataFrame mit MultiIndex (Ticker, Spalte)
    returns = data.xs("Close", level=1, axis=1).pct_change().dropna()  # Renditen berechnen

    # Extrahiere die Renditen der beiden Ticker
    returns_ticker_1 = returns.iloc[:, ticker_indices[0]].values
    returns_ticker_2 = returns.iloc[:, ticker_indices[1]].values

    # Historische Korrelationen berechnen
    historical_correlations = []
    for i in range(len(returns) - window_size + 1):
        window_1 = returns_ticker_1[i:i + window_size]
        window_2 = returns_ticker_2[i:i + window_size]

        # Kovarianz und Korrelation berechnen
        cov_matrix = np.cov(window_1, window_2)
        correlation = cov_matrix[0, 1] / (np.sqrt(cov_matrix[0, 0]) * np.sqrt(cov_matrix[1, 1]))
        historical_correlations.append(correlation)

    return historical_correlations

def calculate_correlation_matrix(dataset, tickers):
    """
    Berechnet die paarweisen Korrelationen zwischen mehreren Tickersymbolen über den gesamten Zeitraum.

    Parameters:
        dataset (TimeBasedDataset): Das Dataset mit den historischen Daten.
        tickers (list): Liste der Tickersymbole.

    Returns:
        pd.DataFrame: Korrelationsmatrix der Tickersymbole.
        pd.Series: Gesamtrenditen der Einzelrenditen.
    """
    # Renditen berechnen
    data = dataset.data  # Pandas DataFrame mit MultiIndex (Ticker, Spalte)
    returns = data.xs("Close", level=1, axis=1).pct_change().dropna()  # Renditen berechnen

    # Korrelationsmatrix berechnen
    correlation_matrix = returns.corr()

    # Gesamtrenditen berechnen (geometrische Rendite über den gesamten Zeitraum)
    cumulative_returns = (1 + returns).prod() - 1

    return correlation_matrix, cumulative_returns


def main():
    # Define 5 diverse tickers
    tickers = [
        "LMT",   # Lockheed Martin (Military)
        "AAPL",  # Apple (Tech)
        "ADM",   # Archer-Daniels-Midland (Agriculture)
        "GOLD",  # Barrick Gold (Gold)
        "JNJ"    # Johnson & Johnson (Medical)
    ]
    start_date = "2020-01-01"
    end_date = "2025-01-01"

    # Load the dataset
    dataset = TimeBasedDataset(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        window_size=60
    )

    # Calculate the correlation matrix and mean returns
    correlation_matrix, mean_returns = calculate_correlation_matrix(dataset, tickers)

    # Output the results
    print("Correlation matrix of tickers:")
    print(correlation_matrix)
    print("\nMean returns of individual tickers:")
    print(mean_returns)

    plot_risk_matrix(correlation_matrix, tickers, title="Correlation Matrix of Tickers")




if __name__ == "__main__":
    main()