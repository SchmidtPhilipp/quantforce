import os
import glob
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
from tqdm import tqdm

def load_tensorboard_data(folders):
    """
    Lädt TensorBoard-Daten aus den ausgewählten Ordnern.
    """
    runs_data = {}

    for run_path in tqdm(folders, desc="Lade TensorBoard-Daten"):
        print(f"Lade Daten aus: {run_path}")
        event_files = glob.glob(os.path.join(run_path, "events.out.tfevents.*"))
        if not event_files:
            print(f"Keine TensorBoard-Dateien gefunden in {run_path}")
            continue

        # Lade TensorBoard-Daten
        event_accumulator = EventAccumulator(run_path)
        event_accumulator.Reload()

        # Extrahiere alle verfügbaren Skalar-Daten
        scalars = {}
        for tag in event_accumulator.Tags()["scalars"]:
            scalars[tag] = event_accumulator.Scalars(tag)

        # Speichere die Daten für diesen Run
        runs_data[os.path.basename(run_path)] = scalars

    return runs_data
