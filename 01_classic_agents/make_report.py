# Import the root folder of this folder
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import qf

if __name__ == "__main__":

    folders = ["./runs/2025-06-09_17-47-26_ClassicOnePeriodMarkovitzAgent_risk_model=sample_cov_log_returns=True_EVAL_George",
                #"./runs/2025-06-09_17-47-48_ClassicOnePeriodMarkovitzAgent_risk_model=sample_cov_log_returns=False_EVAL_Uma",
               "./runs/2025-06-09_17-48-10_ClassicOnePeriodMarkovitzAgent_risk_model=exp_cov_log_returns=True_EVAL_Ethan",
              #"./runs/2025-06-09_17-48-32_ClassicOnePeriodMarkovitzAgent_risk_model=exp_cov_log_returns=False_EVAL_Charlie",
              "./runs/2025-06-09_17-48-54_ClassicOnePeriodMarkovitzAgent_risk_model=ledoit_wolf_log_returns=True_EVAL_Sebastian",
              # "./runs/2025-06-09_17-49-16_ClassicOnePeriodMarkovitzAgent_risk_model=ledoit_wolf_log_returns=False_EVAL_Leo",
               "./runs/2025-06-09_17-49-38_ClassicOnePeriodMarkovitzAgent_risk_model=ledoit_wolf_constant_variance_log_returns=True_EVAL_Hank",
              # "./runs/2025-06-09_17-50-00_ClassicOnePeriodMarkovitzAgent_risk_model=ledoit_wolf_constant_variance_log_returns=False_EVAL_Uma",
               "./runs/2025-06-09_17-50-22_ClassicOnePeriodMarkovitzAgent_risk_model=ledoit_wolf_single_factor_log_returns=True_EVAL_Quinn",
               #"./runs/2025-06-09_17-50-44_ClassicOnePeriodMarkovitzAgent_risk_model=ledoit_wolf_single_factor_log_returns=False_EVAL_Paul",
               "./runs/2025-06-09_17-51-06_ClassicOnePeriodMarkovitzAgent_risk_model=ledoit_wolf_constant_correlation_log_returns=True_EVAL_Samuel",
               #"./runs/2025-06-09_17-51-28_ClassicOnePeriodMarkovitzAgent_risk_model=ledoit_wolf_constant_correlation_log_returns=False_EVAL_Madeline",
               "./runs/2025-06-09_17-51-50_ClassicOnePeriodMarkovitzAgent_risk_model=oracle_approximating_log_returns=True_EVAL_Wendy",
               #"./runs/2025-06-09_17-52-12_ClassicOnePeriodMarkovitzAgent_risk_model=oracle_approximating_log_returns=False_EVAL_Elliot"
               "./runs/2025-06-09_17-52-34_ClassicOnePeriodMarkovitzAgent_risk_model=ML_brownian_motion_logreturn_log_returns=True_EVAL_Harper"
               #"./runs/2025-06-09_17-52-56_ClassicOnePeriodMarkovitzAgent_risk_model=ML_brownian_motion_logreturn_log_returns=False_EVAL_Zane"
               ]
    
    #01_classic_agents/runs/2025-06-10_13-08-19_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=True_TRAIN_Abigail 01_classic_agents/runs/2025-06-10_13-08-19_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=True_TRAIN_Abigail/events.out.tfevents.1749553699.MacBook-Pro.local.56886.0 01_classic_agents/runs/2025-06-10_13-08-52_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=True_TRAIN_Yara 01_classic_agents/runs/2025-06-10_13-09-43_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=True_TRAIN_Katherine 01_classic_agents/runs/2025-06-10_13-10-07_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=True_TRAIN_Gabriella 01_classic_agents/runs/2025-06-10_13-10-27_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=True_EVAL_Alice 01_classic_agents/runs/2025-06-10_13-10-35_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=False_TRAIN_Jacob 01_classic_agents/runs/2025-06-10_13-10-37_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=False_EVAL_Yara 01_classic_agents/runs/2025-06-10_13-10-44_ClassicOnePeriodMarkovitzAgent_risk_model=stepwise_statistics_log_returns=True_TRAIN_Ethan 01_classic_agents/runs/2025-06-10_13-10-46_ClassicOnePeriodMarkovitzAgent_risk_model=stepwise_statistics_log_returns=True_EVAL_Paul 01_classic_agents/runs/2025-06-10_13-10-53_ClassicOnePeriodMarkovitzAgent_risk_model=stepwise_statistics_log_returns=False_TRAIN_Gavin 01_classic_agents/runs/2025-06-10_13-10-55_ClassicOnePeriodMarkovitzAgent_risk_model=stepwise_statistics_log_returns=False_EVAL_Elliot 01_classic_agents/runs/2025-06-10_13-11-02_ClassicOnePeriodMarkovitzAgent_risk_model=ML_brownian_motion_logreturn_log_returns=True_TRAIN_Bella 01_classic_agents/runs/2025-06-10_13-11-04_ClassicOnePeriodMarkovitzAgent_risk_model=ML_brownian_motion_logreturn_log_returns=True_EVAL_Lucas 01_classic_agents/runs/2025-06-10_13-11-11_ClassicOnePeriodMarkovitzAgent_risk_model=ML_brownian_motion_logreturn_log_returns=False_TRAIN_Steve 01_classic_agents/runs/2025-06-10_13-11-12_ClassicOnePeriodMarkovitzAgent_risk_model=ML_brownian_motion_logreturn_log_returns=False_EVAL_Carter
        
    folders = [
               #"runs/2025-06-10_13-10-07_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=True_TRAIN_Gabriella",
               "runs/2025-06-10_13-10-27_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=True_EVAL_Alice",
               #"runs/2025-06-10_13-10-35_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=False_TRAIN_Jacob",
               #"runs/2025-06-10_13-10-37_ClassicOnePeriodMarkovitzAgent_risk_model=terminal_statistics_log_returns=False_EVAL_Yara",
               #"runs/2025-06-10_13-10-44_ClassicOnePeriodMarkovitzAgent_risk_model=stepwise_statistics_log_returns=True_TRAIN_Ethan",
               "runs/2025-06-10_13-10-46_ClassicOnePeriodMarkovitzAgent_risk_model=stepwise_statistics_log_returns=True_EVAL_Paul",
               #"runs/2025-06-10_13-10-53_ClassicOnePeriodMarkovitzAgent_risk_model=stepwise_statistics_log_returns=False_TRAIN_Gavin",
               #"runs/2025-06-10_13-10-55_ClassicOnePeriodMarkovitzAgent_risk_model=stepwise_statistics_log_returns=False_EVAL_Elliot",
               #"runs/2025-06-10_13-11-02_ClassicOnePeriodMarkovitzAgent_risk_model=ML_brownian_motion_logreturn_log_returns=True_TRAIN_Bella",
               "runs/2025-06-10_13-11-04_ClassicOnePeriodMarkovitzAgent_risk_model=ML_brownian_motion_logreturn_log_returns=True_EVAL_Lucas",
               #"runs/2025-06-10_13-11-11_ClassicOnePeriodMarkovitzAgent_risk_model=ML_brownian_motion_logreturn_log_returns=False_TRAIN_Steve",
               #"runs/2025-06-10_13-11-12_ClassicOnePeriodMarkovitzAgent_risk_model=ML_brownian_motion_logreturn_log_returns=False_EVAL_Carter"
              ]
        
        
        

    # Generate short names for the Legend
    names = ["SC", # Sample Covariance, Log Returns
             #"FC", # Sample Covariance, Price Returns
             "EC", # Exponential Covariance, Log Returns
             #"FE", # Exponential Covariance, Price Returns
             "LW", # Ledoit Wolf, Log Returns
             #"LE", # Ledoit Wolf, Price Returns
             "LWC", # Ledoit Wolf Constant Variance, Log Returns
             #"LWE", # Ledoit Wolf Constant Variance, Price Returns
             "LWSF", # Ledoit Wolf Single Factor, Log Returns
             #"LWSFE", # Ledoit Wolf Single Factor, Price Returns
             "LWCC", # Ledoit Wolf Constant Correlation, Log Returns
             #"LWCCE", # Ledoit Wolf Constant Correlation, Price Returns
             "OA", # Oracle Approximating, Log Returns
             #"OAE", # Oracle Approximating, Price Returns
             "MLBM" # ML Brownian Motion Log Return Model
             #"MLBME" # ML Brownian Motion Price Return Model
            ]

    names = ["TS","SS", "MLBM"]


    report = qf.EVALReport(output_folder="REPORT")
    report.run(folders, names=names, color='gist_rainbow')
