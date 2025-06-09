import os
import subprocess
import pandas as pd
import numpy as np
from qf.data import get_data
from qf.utils.plot import plot_dual_axis



def append_to_latex(self, content):
    """
    Fügt neuen Inhalt zur LaTeX-Code-Liste hinzu.
    
    :param content: Der LaTeX-Code, der hinzugefügt werden soll.
    """
    self.latex_code.append(content)


def plot_stockprice_asset_comparison(self):
    """
    Erstellt Plots für jeden Ticker: Preise vs. Zeit und Asset Holdings.
    Speichert die Plots in einem Ordner mit dem Präfix REPORT_... für jeden Run.
    Fügt die Plots auch in den LaTeX-Code ein.
    """

    self.append_to_latex(r"""
    \section{Stock Price and Asset Holdings Comparison}
    \label{sec:stock_price_asset_holdings}
    """)

    for investigation in self.investigations:

        self.append_to_latex(f"""
        \\subsection{{Investigation: {os.path.basename(investigation.train_path).replace("_", " ").upper()}}}
        """)
        # Erstelle den REPORT_ Ordner für die Untersuchung

        for run_type in ['train', 'eval']:

            self.append_to_latex(f"""
            \\subsubsection{{Run Type: {run_type.upper()}}}
            """)

            data = getattr(investigation, run_type)
            if not data:
                continue

            # Erstelle den REPORT_ Ordner basierend auf dem Run-Typ
            report_name = os.path.basename(investigation.train_path).replace("TRAIN_", "").replace("EVAL_", "")
            report_folder = os.path.join(self.runs_folder, f"REPORT_{report_name}")
            os.makedirs(report_folder, exist_ok=True)

            start_date = data['config'].get(f'{run_type}_start', None)
            end_date = data['config'].get(f'{run_type}_end', None)
            indicators = data['config'].get('indicators', [])
            window = data['config'].get('time_window_size', 1)
            asset_tracker = data['asset_tracker']
            config = data['config']
            tickers = config.get('tickers', [])

            for i, ticker in enumerate(tickers):
                # Preise abrufen
                ticker_price = get_data(ticker, start_date, end_date, indicators="Close")

                for actor in range(asset_tracker.n_agents):  # Beispiel: Preise
                    holdings = asset_tracker.get_episode_data("asset_holdings")[:, actor, i]

                    # DataFrame für den Plot erstellen
                    df = pd.DataFrame({
                        "Prices": (ticker_price[window-1:].values.flatten()),  # Preise
                        "Holdings": holdings.flatten(),  # Mittelwert über Episoden
                    })

                    df.index = ticker_price.index[window-1:]  # Setze den Index auf die Zeitstempel

                    # Plot erstellen
                    plot_filename = f"{run_type}_{ticker}_actor_{actor}"
                    plot_dual_axis(
                        df=df,
                        xlabel="Date",
                        ylabel_left="Price",
                        ylabel_right="Holdings",
                        linewidth=1,
                        #title=f"{run_type.upper()} - {ticker}",
                        filename=plot_filename,
                        save_dir=report_folder,
                        max_entries=1000,
                        max_xticks=15,
                    )

                    self.add_pgf_to_latex(plot_filename, report_folder, f"{run_type.upper()} - {ticker} (Actor {actor})")


def plot_price_action_comparison(self):
    """
    Erstellt Plots für jeden Ticker: Preise vs. Zeit und Asset Holdings.
    Speichert die Plots in einem Ordner mit dem Präfix REPORT_... für jeden Run.
    Fügt die Plots auch in den LaTeX-Code ein.
    """

    self.append_to_latex(r"""
    \section{Price Action Comparison}
    \label{sec:price_action_comparison}
    """)

    for investigation in self.investigations:

        self.append_to_latex(f"""
        \\subsection{{Investigation: {os.path.basename(investigation.train_path).replace("_", " ").upper()}}}
        """)
        # Erstelle den REPORT_ Ordner für die Untersuchung

        for run_type in ['train', 'eval']:

            self.append_to_latex(f"""
            \\subsubsection{{Run Type: {run_type.upper()}}}
            """)

            data = getattr(investigation, run_type)
            if not data:
                continue

            # Erstelle den REPORT_ Ordner basierend auf dem Run-Typ
            report_name = os.path.basename(investigation.train_path).replace("TRAIN_", "").replace("EVAL_", "")
            report_folder = os.path.join(self.runs_folder, f"REPORT_{report_name}")
            os.makedirs(report_folder, exist_ok=True)

            start_date = data['config'].get(f'{run_type}_start', None)
            end_date = data['config'].get(f'{run_type}_end', None)
            indicators = data['config'].get('indicators', [])
            window = data['config'].get('time_window_size', 1)
            asset_tracker = data['asset_tracker']
            config = data['config']
            tickers = config.get('tickers', [])

            for i, ticker in enumerate(tickers):
                # Preise abrufen
                ticker_price = get_data(ticker, start_date, end_date, indicators="Close")

                for actor in range(asset_tracker.n_agents):
                    # Preise abrufen
                    ticker_price = get_data(ticker, start_date, end_date, indicators="Close")

                    # DataFrame für den Plot erstellen
                    df = pd.DataFrame({
                        "Prices": (ticker_price[window-1:].values.flatten()),  # Preise
                        "Action": np.mean(asset_tracker.actions[:,:, actor, i], axis=0).flatten(),  # Aktionen
                    })
                    df.index = ticker_price.index[window-1:]  # Setze den Index auf die Zeitstempel
                    # Plot erstellen
                    plot_filename = f"{run_type}_{ticker}_actor_{actor}"
                    plot_dual_axis(
                        df=df,
                        xlabel="Date",
                        ylabel_left="Price",
                        ylabel_right="Action",
                        linewidth=1,
                        #title=f"{run_type.upper()} - {ticker}",
                        filename=plot_filename,
                        save_dir=report_folder,
                        max_entries=1000,
                        max_xticks=15,
                    )
                    self.add_pgf_to_latex(plot_filename, report_folder, f"{run_type.upper()} - {ticker} (Actor {actor})")



def add_pgf_to_latex(self, filename, report_folder, caption):
    """
    Fügt eine PGF-Datei in den LaTeX-Code ein.
    
    :param filename: Name der PGF-Datei.
    :param report_folder: Ordner, in dem die PGF-Datei gespeichert ist.
    :param caption: Bildunterschrift für die PGF-Datei.
    """
    self.append_to_latex(f"""
    \\begin{{figure}}[H]
        \\centering
        \\input{{{os.path.join(report_folder, filename + ".pgf")}}}        
        \\vspace{{-1cm}}
        \\caption{{{caption}}}
    \\end{{figure}}
    """)

