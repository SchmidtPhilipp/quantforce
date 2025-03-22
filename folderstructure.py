import os

def create_structure(base_path, structure):
    for key, value in structure.items():
        # For the key create a folder
        folder_path = os.path.join(base_path, key)
        os.makedirs(folder_path, exist_ok=True)

        print(value)
        # For each value we may decide
        for item in value:
            try:
                sub = value[item]
            except TypeError:
                sub = None

            if sub is not None:
                create_structure(folder_path, {item: sub})
            else:
                print(folder_path, item)
                file_path = os.path.join(folder_path, item)
                with open(file_path, 'w'):
                    pass


project_structure = {
    "rl_portfolio_framework": {
        "config": {"settings.yaml"},
        "data": ["data_loader.py", "feature_engineering.py"],
        "envs": ["base_env.py", "portfolio_env.py", "marl_env.py"],
        "agents": [
            "base_agent.py", "dqn_agent.py", "ppo_agent.py", "a2c_agent.py",
            "maddpg_agent.py", "qmix_agent.py", "marl_utils.py"
        ],
        "models": ["networks.py", "replay_buffer.py"],
        "trainer": ["trainer.py", "marl_trainer.py", "logger.py"],
        "utils": ["metrics.py", "evaluation.py", "seed.py"],
        "": ["main.py", "requirements.txt", "README.md"]
    }
}

if __name__ == "__main__":
    create_structure(".", project_structure)
    print("✅ Folder structure created successfully.")
