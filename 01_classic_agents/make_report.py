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

    report = qf.EVALReport(output_folder="REPORT")
    report.run(folders, names=names, color='gist_rainbow')
