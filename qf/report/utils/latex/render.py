import os
import subprocess

def render(self, folder, latex_data, compile_pdf=True):
    """
    Generiert ein vollständiges LaTeX-Dokument aus dem gespeicherten LaTeX-Code und kompiliert es optional zu einem PDF.
    Speichert die Dateien in einem Ordner namens REPORT.
    
    :param compile_pdf: Ob die LaTeX-Datei zu einem PDF kompiliert werden soll.
    """
    # Erstelle den REPORT-Ordner
    report_folder = os.path.join(folder, "REPORT")
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
    full_document = document_header + page_format+ document_beginning+ "\n".join(latex_data) + document_footer

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