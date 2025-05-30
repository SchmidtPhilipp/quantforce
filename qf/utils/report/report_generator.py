import os
import json
import numpy as np
import pandas as pd
from utils.plot import plot_lines_grayscale, plot_dual_axis  # Importiere die Plot-Funktion
from utils.report.investigation import Investigation  # Importiere die Investigation-Klasse
from utils.tracker.assettracker import AssetTracker  # Importiere die AssetTracker-Klasse
from data.dataset import Dataset  # Importiere die Dataset-Klasse
from data.get_data import get_data
import subprocess

class ReportGenerator:
    def __init__(self, runs_folder):
        """
        Initialisiert den Report Generator.
        
        :param runs_folder: Ordner, der die Runs (TRAIN_... und EVAL_...) enthält.
        """
        self.runs_folder = runs_folder
        self.investigations = []
        self.latex_code = []  # Liste, die den LaTeX-Code speichert

    def load_investigations(self):
        """
        Lädt alle TRAIN_... und EVAL_... Ordner und erstellt Investigation-Objekte.
        """
        train_folders = [f for f in os.listdir(self.runs_folder) if f.startswith("TRAIN_")]
        eval_folders = [f for f in os.listdir(self.runs_folder) if f.startswith("EVAL_")]

        for train_folder in train_folders:
            eval_folder = train_folder.replace("TRAIN_", "EVAL_")
            train_path = os.path.join(self.runs_folder, train_folder)
            eval_path = os.path.join(self.runs_folder, eval_folder)

            if os.path.exists(train_path) and os.path.exists(eval_path):
                self.investigations.append(Investigation(train_path, eval_path))
            else:
                print(f"Fehlende Ordner: {train_path} oder {eval_path}")

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

    def tabular_metrics_comparison(self):
        pass




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


    def generate_latex_document(self, compile_pdf=True):
        """
        Generiert ein vollständiges LaTeX-Dokument aus dem gespeicherten LaTeX-Code und kompiliert es optional zu einem PDF.
        Speichert die Dateien in einem Ordner namens REPORT.
        
        :param compile_pdf: Ob die LaTeX-Datei zu einem PDF kompiliert werden soll.
        """
        # Erstelle den REPORT-Ordner
        report_folder = os.path.join(self.runs_folder, "REPORT")
        os.makedirs(report_folder, exist_ok=True)

        # Pfad für die .tex-Datei und das PDF
        tex_file = os.path.join(report_folder, "report.tex")
        pdf_file = os.path.join(report_folder, "report.pdf")

        # LaTeX-Dokument Header und Footer
        document_header = r"""
        \documentclass[a4paper,
            fleqn,            % Linksbündige Gleichungen
            12pt,             % Standard-Schriftgröße
            ngerman,          % Deutsche Sprache
            oneside,          % Doppelseitiges Layout
            chapterentrydots=true,  % Punkte in Inhaltsverzeichnis für Kapitel
            parskip=half      % Halber Zeilenabstand zwischen Absätzen
            ]{article}
        \usepackage{graphicx}   % Für Grafiken
        \usepackage{float}      % Für die Verwendung von [H] in \begin{figure}
        \usepackage{hyperref}   % Für Hyperlinks im PDF
        \hypersetup{
            colorlinks=true,
            linkcolor=blue,
            urlcolor=cyan,
        }
        \usepackage{booktabs}   % Für bessere Tabellen
        \usepackage{pgf}        % Für die Verwendung von .pgf-Dateien
        \usepackage{pgfplots}   % Für die Verwendung von pgfplots
        \usepackage{lmodern}    % Verbessert die Schriftart
        \usepackage{import}     % Importiere die pgf-Datei
        \def\mathdefault#1{#1}  % Verhindert Fehler bei der Verwendung von \mathdefault
        \pgfplotsset{compat=1.18}  % Setze die Kompatibilitätsversion
        \usepackage{amsmath}    % Für mathematische Formeln
        \usepackage{amssymb}    % Für mathematische Symbole
        """

        # Add the formattting of the page using the LaTeX package scrlayer-scrpage
        page_format = r"""
        \usepackage[a4paper, % Setze das Papierformat auf A4
        left=2cm, % Linker Rand
        right=2cm, % Rechter Rand
        top=1cm, % Oberer Rand
        bottom=1cm, % Unterer Rand
        includehead,  % Kopfzeilen einbeziehen
        includefoot, % Kopf- und Fußzeilen einbeziehen
        nomarginpar,% We don't want any margin paragraphs
        textwidth=10cm,% Set \textwidth to 10cm
        headheight=10mm,% Set \headheight to 10mm
        ]{geometry}
        \usepackage{fancyhdr}
        \pagestyle{fancy}
        \fancyhead{} % clear all header fields
        \fancyfoot{} % clear all footer fields
        \fancyfoot[LE,RO]{\thepage}
        \fancyhead[LE]{\leftmark} % left even page
        \fancyhead[RO]{\rightmark} % right odd page
        """

        # document beginning
        document_beginning = r"""
                \begin{document}
        \title{Report}
        \date{\today}
        \maketitle
        \newpage
        \tableofcontents
        \newpage
        """

        document_footer = r"""
        \end{document}
        """

        # Kombiniere Header, LaTeX-Code und Footer
        full_document = document_header + page_format+ document_beginning+ "\n".join(self.latex_code) + document_footer

        # Schreibe das LaTeX-Dokument in die .tex-Datei
        with open(tex_file, "w") as f:
            f.write(full_document)

        print(f"LaTeX-Dokument wurde unter {tex_file} gespeichert.")

        # Kompiliere die LaTeX-Datei zu einem PDF
        if compile_pdf:
            try:
                subprocess.run(["pdflatex", "-output-directory", report_folder, tex_file], check=True)
                # We have to run pdflatex twice to get the table of contents right
                subprocess.run(["pdflatex", "-output-directory", report_folder, tex_file], check=True)
                print(f"PDF wurde erfolgreich kompiliert: {pdf_file}")
            except subprocess.CalledProcessError as e:
                print(f"Fehler beim Kompilieren der LaTeX-Datei: {e}")
            except FileNotFoundError:
                print("Fehler: pdflatex ist nicht installiert oder nicht im PATH verfügbar.")