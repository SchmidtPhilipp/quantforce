import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from data.dataset import TimeBasedDataset  # Importiere das TimeBasedDataset
import matplotlib.pyplot as plt

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
    # Definiere die Ticker und den Zeitraum
    tickers = [
        # 1. Nordamerika
        # Technologie
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet
        "AMZN",  # Amazon
        "NVDA",  # NVIDIA
        "META",  # Meta Platforms
        "TSLA",  # Tesla
        "ADBE",  # Adobe
        "ORCL",  # Oracle

        # Energie
        "XOM",   # Exxon Mobil
        "CVX",   # Chevron
        "COP",   # ConocoPhillips
        "SLB",   # Schlumberger
        "HAL",   # Halliburton

        # Finanzen
        "JPM",   # JPMorgan Chase
        "BAC",   # Bank of America
        "C",     # Citigroup
        "GS",    # Goldman Sachs
        "MS",    # Morgan Stanley

        # Gesundheitswesen
        "JNJ",   # Johnson & Johnson
        "PFE",   # Pfizer
        "MRK",   # Merck & Co.
        "ABBV",  # AbbVie
        "UNH",   # UnitedHealth Group

        # Konsumgüter
        "PG",    # Procter & Gamble
        "KO",    # Coca-Cola
        "PEP",   # PepsiCo
        "MCD",   # McDonald's
        "SBUX",  # Starbucks

        # Rohstoffe
        "GOLD",  # Barrick Gold
        "NEM",   # Newmont Corporation
        "FCX",   # Freeport-McMoRan

        # 2. Europa
        # Technologie
        "SAP",   # SAP SE, Deutschland
        "ASML",  # ASML Holding, Niederlande
        "NOK",   # Nokia, Finnland

        # Energie
        "BP",    # BP, Großbritannien
        "SHEL",  # Shell, Niederlande/Großbritannien
        "TOT",   # TotalEnergies, Frankreich

        # Finanzen
        "HSBC",  # HSBC Holdings, Großbritannien
        "CS",    # Credit Suisse, Schweiz
        "DB",    # Deutsche Bank, Deutschland

        # Gesundheitswesen
        "NVS",   # Novartis, Schweiz
        "RHHBY", # Roche, Schweiz
        "AZN",   # AstraZeneca, Großbritannien

        # Konsumgüter
        "NESN",  # Nestlé, Schweiz
        "UL",    # Unilever, Großbritannien/Niederlande
        "LVMH",  # LVMH Moët Hennessy Louis Vuitton, Frankreich

        # 3. Asien
        # Technologie
        "TCEHY", # Tencent, China
        "BABA",  # Alibaba, China
        "TSM",   # Taiwan Semiconductor Manufacturing, Taiwan
        "NTES",  # NetEase, China
        "INFY",  # Infosys, Indien

        # Energie
        "PTR",   # PetroChina, China
        "CNOOC", # China National Offshore Oil Corporation, China
        "ONGC",  # Oil and Natural Gas Corporation, Indien

        # Finanzen
        "ICBC",  # Industrial and Commercial Bank of China, China
        "HDB",   # HDFC Bank, Indien
        "MFG",   # Mizuho Financial Group, Japan

        # Konsumgüter
        "TM",    # Toyota, Japan
        "SNE",   # Sony, Japan
        "BYD",   # BYD Company, China

        # 4. Globale ETFs
        # Aktien-ETFs
        "SPY",   # SPDR S&P 500 ETF, USA
        "QQQ",   # Invesco QQQ ETF, USA
        "EFA",   # iShares MSCI EAFE ETF, Europa/Asien
        "VWO",   # Vanguard FTSE Emerging Markets ETF, Schwellenländer

        # Rohstoff-ETFs
        "GLD",   # SPDR Gold Shares, Gold
        "SLV",   # iShares Silver Trust, Silber
        "USO",   # United States Oil Fund, Öl

        # Anleihen-ETFs
        "TLT",   # iShares 20+ Year Treasury Bond ETF, USA
        "BND",   # Vanguard Total Bond Market ETF, USA
        "IEF",   # iShares 7-10 Year Treasury Bond ETF, USA

        # 5. Rohstoffe
        # Edelmetalle
        "XAU/USD", # Gold
        "XAG/USD", # Silber
        "XPT/USD", # Platin

        # Energie
        "WTI",     # West Texas Intermediate Öl
        "Brent",   # Brent Crude Öl
        "NG",      # Erdgas (Natural Gas)

        # Agrarrohstoffe
        "Wheat",   # Weizen
        "Corn",    # Mais
        "Soybeans" # Sojabohnen

        # 6. Kryptowährungen
        "BTC",     # Bitcoin
        "ETH",     # Ethereum
        "BNB",     # Binance Coin
        "XRP",     # Ripple
        "ADA",     # Cardano

        # 7. Immobilien (REITs)
        "VNQ",     # Vanguard Real Estate ETF, USA
        "SPG",     # Simon Property Group, USA
        "PLD",     # Prologis, USA
        "DLR"      # Digital Realty Trust, USA
    ]
    tickers = [
        "AAPL",  # Apple
        "MSFT",  # Microsoft
        "GOOGL", # Alphabet (Google)
        "GOLD",  # Barrick Gold
        "NEM",   # Newmont Corporation
        "XOM",   # Exxon Mobil
        "CVX",   # Chevron
        "TSLA",  # Tesla
        "ENPH"   # Enphase Energy
    ]
    start_date = "2020-01-01"
    end_date = "2025-01-01"

    # Lade das Dataset
    dataset = TimeBasedDataset(
        tickers=tickers,
        start_date=start_date,
        end_date=end_date,
        interval="1d",
        window_size=60
    )

    # Berechne die Korrelationsmatrix und die Erwartungswerte
    correlation_matrix, mean_returns = calculate_correlation_matrix(dataset, tickers)

    # Ausgabe der Ergebnisse
    print("Korrelationsmatrix der Tickersymbole:")
    print(correlation_matrix)
    print("\nErwartungswerte der Einzelrenditen:")
    print(mean_returns)

    # Erwartungswerte als zusätzliche Zeile zur Korrelationsmatrix hinzufügen
    extended_matrix = correlation_matrix.copy()
    extended_matrix["E[R]"] = mean_returns # Leere Spalte für die Erwartungswerte

    # Visualisierung der erweiterten Matrix mit matplotlib
    fig, ax = plt.subplots(figsize=(10, 9))  # Größeres Plot-Fenster für zusätzliche Zeile
    cax = ax.matshow(extended_matrix, cmap="Greys", vmin=-1, vmax=1)  # Graufarbene Darstellung mit begrenzter Farbskala
    fig.colorbar(cax)

    # Achsenticks und Labels
    ax.set_xticks(range(len(tickers) + 1))  # Zusätzliche Spalte für Erwartungswerte
    ax.set_yticks(range(len(tickers)))  # Zusätzliche Zeile für Erwartungswerte
    ax.set_xticklabels(tickers + ["E[R]"], rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(tickers, fontsize=8)

    # Zahlen in die Matrix schreiben (gerundet auf eine Nachkommastelle)
    for i in range(len(tickers)):
        for j in range(len(tickers)):
            value = correlation_matrix.iloc[i, j]
            color = "white" if value > 0 else "black"  # Weiß für Werte > 0, Schwarz für Werte <= 0
            ax.text(j, i, f"{value:.1f}", ha="center", va="center", color=color, fontsize=8)

    # Erwartungswerte in die letzte Zeile schreiben
    for j in range(len(tickers)):
        mean_value = mean_returns.iloc[j]
        value = mean_value
        color = "white" if value > 0 else "black"  # Weiß für Werte > 0, Schwarz für Werte <= 0
        ax.text(len(tickers), j, f"{mean_value:.2%}", ha="center", va="center", color=color, fontsize=8)

    # Titel und Layout
    plt.title(f"Korrelationsmatrix der Tickersymbole\nZeitraum: {start_date} bis {end_date}", pad=20)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()