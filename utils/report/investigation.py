from utils.tracker.assettracker import AssetTracker
import os
import json

class Investigation:
    def __init__(self, train_path, eval_path):
        """
        Initialisiert eine Untersuchung mit TRAIN- und EVAL-Daten.
        
        :param train_path: Pfad zum TRAIN-Ordner.
        :param eval_path: Pfad zum EVAL-Ordner.
        """
        self.train_path = train_path
        self.eval_path = eval_path
        self.train = self._load_data(train_path)
        self.eval = self._load_data(eval_path)

        self.identifier = os.path.basename(train_path).replace("TRAIN_", "")

    def _load_data(self, folder_path):
        """
        Lädt die Konfiguration und Asset-Daten aus einem Ordner.
        
        :param folder_path: Pfad zum Ordner (TRAIN oder EVAL).
        :return: Ein Dictionary mit 'config' und 'asset_tracker'.
        """
        data = {}
        try:
            # Lade die Konfiguration
            config_path = os.path.join(folder_path, "config.json")
            with open(config_path, 'r') as f:
                data['config'] = json.load(f)

            # Lade die Asset-Daten
            npz_path = os.path.join(folder_path, "asset_data.npz")  # Immer "data.npz"
            data['asset_tracker'] = AssetTracker.load(npz_path)
        except Exception as e:
            print(f"Fehler beim Laden der Daten aus {folder_path}: {e}")
        return data
    
    def get_id(self):
        """
        Gibt die ID der Untersuchung zurück.
        
        :return: Die ID der Untersuchung.
        """
        return self.identifier