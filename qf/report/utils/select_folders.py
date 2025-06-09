import os
from tkinter import Tk, filedialog, messagebox


def select_folders():
    """
    Öffnet ein Finder-Fenster (macOS) oder Datei-Explorer (Windows/Linux) für die Auswahl mehrerer Ordner.
    Der Benutzer kann nacheinander mehrere Ordner auswählen. Das Fenster öffnet sich im aktuellen Arbeitsverzeichnis.
    """
    root = Tk()
    root.withdraw()  # Versteckt das Hauptfenster
    root.title("Ordnerauswahl")

    selected_folders = []
    current_location = os.getcwd()  # Aktuelles Arbeitsverzeichnis abrufen

    while True:
        # Öffne ein Dialogfenster zur Auswahl eines Ordners
        folder_path = filedialog.askdirectory(title="Wähle einen Ordner aus", initialdir=current_location)
        if not folder_path:
            # Wenn kein Ordner ausgewählt wurde, breche die Schleife ab
            break

        # Füge den Ordner zur Liste hinzu, wenn er noch nicht ausgewählt wurde
        if folder_path not in selected_folders:
            selected_folders.append(folder_path)

        # Frage den Benutzer, ob er weitere Ordner auswählen möchte
        if not messagebox.askyesno("Weitere Ordner?", "Möchten Sie weitere Ordner auswählen?"):
            # Close messagebox
            break

    if not selected_folders:
        print("Keine Ordner ausgewählt.")
        return None

    return selected_folders