import argparse
import os
import shutil
import random
import numpy as np
import torch

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

        

# Set a different seed for each training run
set_seed(random.randint(0, 10000))

from train.process import process_config

def main():
    # CLI arg: --config_folder configs/pending --processed_folder configs/processed
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_folder", type=str, default="configs/pending", help="Folder containing pending configuration files")
    parser.add_argument("--processed_folder", type=str, default="configs/processed", help="Folder to move processed configuration files")
    args = parser.parse_args()

    config_folder = args.config_folder
    processed_folder = args.processed_folder

    # Ensure the processed folder exists
    os.makedirs(processed_folder, exist_ok=True)

    # Get all JSON files in the config folder
    config_files = [os.path.join(config_folder, f) for f in os.listdir(config_folder) if f.endswith(".json")]

    if not config_files:
        print(f"No configuration files found in {config_folder}. Exiting.")
        return

    print(f"Found {len(config_files)} configuration files in {config_folder}.")

    # Process each configuration file
    for config_path in config_files:
        
        process_config(config_path)

        # Move the processed config to the processed folder
        shutil.move(config_path, os.path.join(processed_folder, os.path.basename(config_path)))
        print(f"Configuration {config_path} processed and moved to {processed_folder}.")
        try:
            continue
        except Exception as e:
            print(f"Error processing configuration {config_path}: {e}")
            continue

    print("All configurations processed.")


if __name__ == "__main__":
    main()




