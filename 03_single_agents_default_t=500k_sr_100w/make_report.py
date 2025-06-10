# Import the root folder of this folder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qf

if __name__ == "__main__":

    folders = ["./runs/2025-06-10_03-50-48_SPQLAgent_default_config_t=500k_sr_100w_EVAL_Kyle",
               "./runs/2025-06-10_02-46-42_PPOAgent_default_config_t=500k_sr_100w_EVAL_Yasmine",
               "./runs/2025-06-10_02-20-35_TD3Agent_default_config_t=500k_sr_100w_EVAL_Mona",
               "./runs/2025-06-09_23-32-50_DDPGAgent_default_config_t=500k_sr_100w_EVAL_Bianca",
               "./runs/2025-06-10_01-07-04_SACAgent_default_config_t=500k_sr_100w_EVAL_Adrian",
               "./runs/2025-06-10_03-50-59_ClassicOnePeriodMarkovitzAgent_default_config_t=500k_sr_100w_EVAL_Steve"]
               
    names = ["SPQL",
             "PPO", 
             "TD3", 
             "DDPG",
             "SAC",
             "Tangency"]

    report = qf.EVALReport(output_folder="REPORT")
    report.run(folders, names=names, color='gist_rainbow')