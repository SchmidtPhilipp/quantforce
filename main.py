import argparse
from qf import process_config


def main():
    # CLI arg: --config configs/example_config.json
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to the configuration file")
    args = parser.parse_args()

    # Use the provided config file or fall back to a default config
    if args.config:
        config_path = args.config
    else:
        config_path = "configs/default/MADDPG_minimal.json"
        print(f"No configuration file provided. Using default configuration: {config_path}")


    process_config(config_path)



if __name__ == "__main__":
    main()




