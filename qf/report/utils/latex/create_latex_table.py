import os

def create_latex_table(metrics, names, output_folder="latex_tables", filename="metrics_table", std_dev=False, transpose=False):
    """
    Erstellt eine LaTeX-Tabelle basierend auf den Metriken und speichert sie in einer Datei.

    :param metrics: Liste von Metrics-Instanzen.
    :param names: Liste von Namen, die den Spalten der Tabelle entsprechen.
    :param output_folder: Ordner, in dem die Tabelle gespeichert wird.
    :param filename: Name der LaTeX-Datei (ohne Erweiterung).
    :param std_dev: Ob die Standardabweichung in der Tabelle angezeigt werden soll.
    :param transpose: Ob die Tabelle transponiert werden soll.
    :return: LaTeX-Code der Tabelle als String.
    """
    # Erstelle den Ordner, falls er nicht existiert
    os.makedirs(output_folder, exist_ok=True)

    # Extrahiere die formatierten Metriken
    formatted_metrics = [metric.formated(std_dev) for metric in metrics]
    metric_names = list(formatted_metrics[0].keys())  # Zeilen (Metriken)

    # Erstelle den LaTeX-Code für die Tabelle
    if not transpose:
        # Standardformat (Metriken als Zeilen, Namen als Spalten)
        latex_code = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{l" + "c" * len(names) + "}\n"
        latex_code += "\\toprule\n"
        latex_code += "Metric & " + " & ".join(names) + " \\\\\n"
        latex_code += "\\midrule\n"

        for metric in metric_names:
            row = [formatted_metrics[i][metric] for i in range(len(names))]
            latex_code += f"{metric} & " + " & ".join(row) + " \\\\\n"

        latex_code += "\\bottomrule\n\\end{tabular}\n\\caption{Metrics Table}\n\\label{tab:metrics_table}\n\\end{table}"
    else:
        # Transponiertes Format (Namen als Zeilen, Metriken als Spalten)
        latex_code = "\\begin{table}[H]\n\\centering\n\\begin{tabular}{|l|" + "c|" * len(metric_names) + "}\n"
        latex_code += "\\toprule\n"
        latex_code += "\\rotatebox{90}{Name} & \\rotatebox{90}{" + "} & \\rotatebox{90}{".join(metric_names) + "} \\\\\n"
        latex_code += "\\midrule\n"

        for i, name in enumerate(names):
            row = [formatted_metrics[i][metric] for metric in metric_names]
            latex_code += f"{name} & " + " & ".join(row) + " \\\\\n"

        latex_code += "\\bottomrule\n\\end{tabular}\n\\caption{Metrics Table (Transposed)}\n\\label{tab:metrics_table_transposed}\n\\end{table}"

    # Speichere die Tabelle in einer Datei
    filepath = os.path.join(output_folder, f"{filename}.tex")
    with open(filepath, "w") as f:
        f.write(latex_code)

    print(f"✅ LaTeX-Tabelle gespeichert unter: {filepath}")
    return latex_code

