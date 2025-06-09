import pandas as pd
from tqdm import tqdm


def tensorboard_to_dataframe(data):
    """
    Konvertiert TensorBoard-Daten in Pandas DataFrames.
    """

    dataframes = {}
    for run_name, scalars in tqdm(data.items(), desc="Konvertiere Daten in DataFrames"):
        print(f"Konvertiere Daten für Run: {run_name}")
        run_df = pd.DataFrame()

        for tag, events in scalars.items():
            tag_df = pd.DataFrame(events)
            tag_df["tag"] = tag
            run_df = pd.concat([run_df, tag_df], ignore_index=True)

        dataframes[run_name] = run_df

    return dataframes