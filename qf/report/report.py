from qf.report.utils.latex.render import render
from qf.report.utils.latex.generate_plot_and_latex import plot_latex_single_axis

from qf.utils.tracker.tracker import Tracker
from qf.utils.metrics import Metrics
from qf.report.utils.latex.create_latex_table import create_latex_table

import qf

class EVALReport:
    def __init__(self, output_folder="tensorboard_reports"):
        """
        Initialisiert die Report-Klasse.

        :param output_folder: Pfad zum Ordner, in dem die Berichte gespeichert werden.
        """
        self.output_folder = output_folder
        self.latex = []

    def run(self, folders, names=None, save_csv=True, generate_summary=True, color='hsv'):
        """
        Führt den gesamten Prozess aus: Laden, Konvertieren, Speichern und Zusammenfassen.

        :param save_csv: Ob die Daten als CSV gespeichert werden sollen.
        :param generate_summary: Ob eine Zusammenfassung der Daten erstellt werden soll.
        """

        ##################################################################################
        # Load the tracker files from the folders
        balance_df = Tracker.get_df_from_trackers(folders, df_request="balance", col_names=names)
        

        ################################################################################
        # Load the metrics from the metrics files
        metric_files = Metrics.get_metrics_files(folders)
        metrics = [ Metrics.load(file) for file in metric_files ]

    
        ##########################################################################
        # Reports
        latex_data = []

        # Plot Portfolio Value
        latex_data.append(plot_latex_single_axis(balance_df, 
                                                 xlabel="Date", 
                                                 ylabel="Portfolio Value",
                                                 folder_name=self.output_folder,
                                                 filename="portfolio_value_paper",
                                                 colorscheme=color))
        latex_data.append(plot_latex_single_axis(balance_df, 
                                                 xlabel="Date", 
                                                 ylabel="Portfolio Value",
                                                 folder_name=self.output_folder,
                                                 filename="portfolio_value_beamer", 
                                                 figsize=qf.DEFAULT_FIGSIZE_BEAMER, 
                                                 colorscheme=color))


        # Create LaTeX table for metrics
        latex_data.append(create_latex_table(metrics, names, output_folder=self.output_folder, filename="metrics_table"))
        latex_data.append(create_latex_table(metrics, names, output_folder=self.output_folder, filename="metrics_table_transposed", transpose=True))

        #########################################################################
        # Render
        render(self.output_folder, latex_data, compile_pdf=True)

        print(f"✅ Report saved to {self.output_folder}")



