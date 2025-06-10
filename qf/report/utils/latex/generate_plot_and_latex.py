import os
from qf.utils.plot import plot_dual_axis, plot_lines
import pandas as pd
import qf


def plot_latex_dual_axis(
    df: pd.DataFrame,
    xlabel: str = "X-Axis",
    ylabel_left: str = "Y-Axis (Left)",
    ylabel_right: str = None,
    title: str = None,
    linewidth: float = 2.0,
    folder_name: str = "plots",
    filename: str = "plot",
    max_xticks: int = 12,
    max_entries: int = None,
    num_yticks: int = 5,
    round_base: int = 100,
    verbosity: int = 0,
    y_limits_left: tuple[float, float] = None,
    y_limits_right: tuple[float, float] = None,
):
    """
    Erstellt einen Plot basierend auf den angegebenen Parametern und gibt den LaTeX-Code zurück.

    :param df: Pandas DataFrame mit den Daten.
    :param x_col: Spaltenname für die X-Achse.
    :param y_left_col: Spaltenname für die linke Y-Achse.
    :param y_right_col: (Optional) Spaltenname für die rechte Y-Achse.
    :param xlabel: Beschriftung der X-Achse.
    :param ylabel_left: Beschriftung der linken Y-Achse.
    :param ylabel_right: (Optional) Beschriftung der rechten Y-Achse.
    :param title: (Optional) Titel des Plots.
    :param colorscheme: (Optional) Farbschema für den Plot.
    :param linewidth: Breite der Linien.
    :param figsize: Größe des Plots (Breite, Höhe).
    :param folder_name: Ordner, in dem der Plot gespeichert wird.
    :param filename: Name der Datei (ohne Erweiterung).
    :param max_xticks: Maximale Anzahl der X-Ticks.
    :param max_entries: Maximale Anzahl der Einträge im Plot.
    :param num_yticks: Anzahl der Y-Ticks.
    :param round_base: Basis für das Runden der Achsenlimits.
    :param verbosity: Verbositätslevel.
    :param y_limits_left: Y-Achsenlimits für die linke Achse.
    :param y_limits_right: Y-Achsenlimits für die rechte Achse.
    :return: LaTeX-Code für den Plot.
    """
    # Erstelle den Ordner, falls er nicht existiert
    os.makedirs(folder_name, exist_ok=True)

    # Erstelle den Plot
    plot_dual_axis(
        df=df,
        xlabel=xlabel,
        ylabel_left=ylabel_left,
        ylabel_right=ylabel_right,
        title=title,
        filename=filename,
        save_dir=folder_name,
        max_xticks=max_xticks,
        max_entries=max_entries,
        linewidth=linewidth,
        num_yticks=num_yticks,
        round_base=round_base,
        verbosity=verbosity,
        y_limits_left=y_limits_left,
        y_limits_right=y_limits_right,
    )

    # Generiere den LaTeX-Code
    latex_code = f"""
    \\begin{{figure}}[H]
        \\centering
        \\input{{{os.path.join(folder_name, filename + ".pgf")}}}
        \\caption{{{title if title else ''}}}
        \\label{{fig:{filename}}}
    \\end{{figure}}
    """
    return latex_code


def plot_latex_single_axis(
    df: pd.DataFrame,
    xlabel: str = "X-Axis",
    ylabel: str = "Y-Axis",
    title: str = None,
    linewidth: float = 1.0,
    folder_name: str = "plots",
    filename: str = "plot",
    max_xticks: int = 12,
    figsize: tuple[float, float] = qf.DEFAULT_FIGSIZE_PAPER,
    colorscheme: str = "jet",
):
    """
    Erstellt einen Plot mit einer einzelnen Y-Achse und gibt den LaTeX-Code zurück.

    :param df: Pandas DataFrame mit den Daten.
    :param xlabel: Beschriftung der X-Achse.
    :param ylabel: Beschriftung der Y-Achse.
    :param title: (Optional) Titel des Plots.
    :param linewidth: Breite der Linien.
    :param folder_name: Ordner, in dem der Plot gespeichert wird.
    :param filename: Name der Datei (ohne Erweiterung).
    :param max_xticks: Maximale Anzahl der X-Ticks.
    :param max_entries: Maximale Anzahl der Einträge im Plot.
    :param num_yticks: Anzahl der Y-Ticks.
    :param round_base: Basis für das Runden der Achsenlimits.
    :param verbosity: Verbositätslevel.
    :return: LaTeX-Code für den Plot.
    """     
    # Erstelle den Ordner, falls er nicht existiert
    os.makedirs(folder_name, exist_ok=True)

    plot_lines(
        df=df,
        xlabel=xlabel,
        ylabel=ylabel,
        title=title,
        filename=filename,
        save_dir=folder_name,
        max_xticks=max_xticks,
        y_limits=None,  # Set to None for automatic limits
        figsize=figsize,
        linewidth=linewidth,
        colorscheme=colorscheme
    )

    # Generiere den LaTeX-Code
    latex_code = f"""
    \\begin{{figure}}[H]
        \\centering
        \\input{{{os.path.join(folder_name, filename + ".pgf")}}}
        \\caption{{{title if title else ''}}}
        \\label{{fig:{filename}}}
    \\end{{figure}}
    """
    return latex_code