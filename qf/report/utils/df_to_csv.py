import os
from tqdm import tqdm
import pandas as pd


def df_to_csv(dataframes, output_folder):
    """
    Speichert die konvertierten DataFrames als CSV-Dateien.
    """
    if output_folder is None:
        raise ValueError("output_folder not defined")


    os.makedirs(output_folder, exist_ok=True)

    for run_name, df in tqdm(dataframes.items(), desc="Speichere DataFrames als CSV"):
        output_path = os.path.join(output_folder, f"{run_name}.csv")
        df.to_csv(output_path, index=False)
        print(f"Gespeichert: {output_path}")