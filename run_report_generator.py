from pathlib import Path
import json
import numpy as np
from utils.report.report_generator import ReportGenerator




# Beispiel für die Verwendung des ReportGenerators
if __name__ == "__main__":
    runs_folder = "runs_04_17_first_success"
    report_generator = ReportGenerator(runs_folder)
    
    # Investigations laden
    report_generator.load_investigations()
    
    # Plots erstellen und LaTeX-Code generieren
    report_generator.plot_stockprice_asset_comparison()
    report_generator.plot_price_action_comparison()
    
    # LaTeX-Dokument erstellen und kompilieren
    report_generator.generate_latex_document(compile_pdf=True)
    
    print("Report wurde erstellt und kompiliert.")