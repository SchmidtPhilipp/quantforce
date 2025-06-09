from qf.report.utils.select_folders import select_folders
from qf.report.utils.load_tensorboard_data import load_tensorboard_data
from qf.report.utils.tensorboard_to_dataframe import tensorboard_to_dataframe
from qf.report.utils.df_to_csv import df_to_csv


from qf.report.utils.latex.render import render
from qf.report.utils.latex.generate_plot_and_latex import generate_plot_and_latex

from qf.utils.tracker.tracker import Tracker


class EVALReport:
    def __init__(self, output_folder="tensorboard_reports"):
        """
        Initialisiert die Report-Klasse.

        :param output_folder: Pfad zum Ordner, in dem die Berichte gespeichert werden.
        """
        self.output_folder = output_folder
        self.latex = []

    def run(self, folders, save_csv=True, generate_summary=True):
        """
        Führt den gesamten Prozess aus: Laden, Konvertieren, Speichern und Zusammenfassen.

        :param save_csv: Ob die Daten als CSV gespeichert werden sollen.
        :param generate_summary: Ob eine Zusammenfassung der Daten erstellt werden soll.
        """
        

            
        tracker_files = get_tracker_files(folders)

        # For the Evaulation we can load the data from the files from the tracker. 
        tracker = Tracker.load(tracker_files[0])



        latex_data = []
        # Reports

        # TD_errors
        latex_data.append(generate_plot_and_latex(df))



        # Render
        render(self, self.output_folder, latex_data, compile_pdf=True)

import os
def get_tracker_files(folders):
    """
    Sammelt alle Tracker-Dateien aus den angegebenen Ordnern.

    :param folders: Liste von Ordnern, in denen nach Tracker-Dateien gesucht wird.
    :return: Liste von Pfaden zu den gefundenen Tracker-Dateien.
    """
    tracker_files = []
    for folder in folders:
        for root, _, files in os.walk(folder):
            for file in files:
                if file.endswith(".pt") and "tracker" in file.lower():
                    tracker_files.append(os.path.join(root, file))
    return tracker_files




if __name__ == "__main__":

    folders = ["./runs_base_config/2025-06-09_00-06-37_Tangency_default_config_EVAL_Harper"]

    report = EVALReport(output_folder="tensorboard_reports")
    report.run(folders)