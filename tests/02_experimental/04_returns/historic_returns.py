import sys
import os

# Include ../../ to access the get_data and tickers modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")))

from data.get_data import get_data
from utils.plot import plot_lines_grayscale

from data import DOWJONES

def main():
    
    tickers = DOWJONES
    # Fetch historical data
    start_date = "2000-01-01"
    end_date = "2019-12-31"
    data = get_data(tickers, start_date, end_date, indicators=("Close",))

    # Remove the second entry of the Multiindex
    data.columns = data.columns.droplevel(1)

    # Berechnung der Renditen der Wertpapiere R(T) = (S(T) - S(0)) / S(0)
    returns = (data.iloc[:] - data.iloc[0]) / data.iloc[0]


    # Plot the historic returns of 5 Assets
    plot_lines_grayscale(returns[tickers], xlabel="Date", ylabel="Return", filename="historic_returns",
                         figsize=(8, 2.5), max_xticks=12, save_dir="tests/04_returns", linewidth=1)
    



if __name__ == "__main__":
    main()