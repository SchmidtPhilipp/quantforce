import os
from qf.utils.plot import plot_dual_axis


def generate_plot_and_latex(
    df,
    x_col,
    y_left_col,
    y_right_col=None,
    xlabel="X-Axis",
    ylabel_left="Y-Axis (Left)",
    ylabel_right=None,
    title=None,
    colorscheme=None,
    linewidth=1,
    figsize=(10, 6),
    folder_name="plots",
    filename="plot",
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
        x_col=x_col,
        y_left_col=y_left_col,
        y_right_col=y_right_col,
        linewidth=linewidth,
        figsize=figsize,
        title=title,
        filename=filename,
        save_dir=folder_name,
        colorscheme=colorscheme,
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