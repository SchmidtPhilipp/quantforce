# Import the root folder of this folder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qf

if __name__ == "__main__":

    folders = ["./runs_base_config/2025-06-09_15-06-10_SPQLAgent_default_config_EVAL_Daisy", 
                "./runs_base_config/2025-06-09_14-35-10_PPOAgent_default_config_EVAL_Jacob",
               "./runs_base_config/2025-06-09_14-51-50_TD3Agent_default_config_EVAL_Frank",
              "./runs_base_config/2025-06-09_15-39-53_DDPGAgent_default_config_EVAL_Uma",
              "./runs_base_config/2025-06-09_16-00-30_SACAgent_default_config_EVAL_Mason",
               "./runs_base_config/2025-06-09_15-26-09_Tangency_default_config_EVAL_Will"]

    names = ["SPQL",
             "PPO", 
             "TD3", 
             "DDPG",
             "SAC",
             "Tangency"]

    report = qf.EVALReport(output_folder="REPORT")
    report.run(folders, names=names, color='gist_rainbow')