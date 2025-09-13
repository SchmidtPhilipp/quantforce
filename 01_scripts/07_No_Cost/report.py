import qf
import pandas as pd
from typing import List

# Predefined list of all available agent types in the framework
AVAILABLE_AGENT_TYPES: List[str] = [
    # Classic Agents
    "ClassicOnePeriodMarkowitzAgent",
    "OneOverNPortfolioAgent",
    # Modern Agents
    "HJBPortfolioAgent",
    "HJBPortfolioAgentWithCosts",
    # SB3 Agents (Stable Baselines 3)
    "SACAgent",
    "PPOAgent",
    "DDPGAgent",
    "A2CAgent",
    "TD3Agent",
    # Tensor Agents
    "MADDPGAgent",
    "DQNAgent",
    "SPQLAgent",
    "RandomAgent",
    "Tangencytest",
]


def main():
    """Main function to run the report analysis."""
    # Example usage - replace your manual loading with this:
    runs_dir = "./runs"

    # You can control whether to include seed information in run names:
    # - remove_seed_from_name=True: "SAC", "PPO", "DQN" (default)
    # - remove_seed_from_name=False: "SAC_seed_1", "SAC_seed_2", etc.

    # Load all runs for all available agent types
    all_runs = qf.Run.load_runs_by_agent(
        runs_dir,
        agent_types=AVAILABLE_AGENT_TYPES,  # Use predefined agent types
        phases=["VAL"],  # Only include VAL runs
        exclude_phases=[
            #           "PPO",
            #           "DDPG",
            #           "MADDPG",
            "EVAL",
            "TRAIN",
            "test",
        ],  # Explicitly exclude TRAIN_EVAL
        remove_seed_from_name=True,  # Set to False to keep seed information in names
    )

    # Create a list of frames for easy comparison
    frames = []

    # Automatically create frames for all available agent types
    for agent_type, runs in all_runs.items():
        frame = qf.Run.combine_runs(runs)
        frames.append(frame)
        print(f"Added frame for {agent_type} with {len(runs)} runs")

    # Combine all frames into a single frame for overall comparison
    combined_frame = qf.PlotFrame.combine_frames(frames)

    # Plot the results - now you have [tangency_frame, maddpg_frame] in frames list
    if frames:
        # Create metrics table comparison
        print("\n" + "=" * 80)
        print("METRICS TABLE COMPARISON")
        print("=" * 80)

        # Generate and print the metrics table
        metrics_table = qf.PlotFrame.metrics_table_comparison(
            frame=combined_frame,
            periods_per_year=252,  # Assuming daily data
            risk_free_rate=0.0,
            tranposed=True,
        )

        print(metrics_table)

        # Optionally save the table to a file
        with open("metrics_table.tex", "w") as f:
            f.write(metrics_table)
        print(f"\nMetrics table saved to metrics_table.tex")

        print("\n" + "=" * 80)
        print("PLOTS")
        print("=" * 80)
        combined_frame.plot_balance()

        # qf.PlotFrame.plot_confidence_rewards(frames, mean_of_level=["run"])
        qf.PlotFrame.plot_confidence_balance(frames, mean_of_level=["run"])
        # qf.PlotFrame.plot_confidence_actions(
        #    frames,
        #    mean_of_level=["run"],
        #    plot_config=qf.PlotConfig(
        #        confidence_interval=qf.ConfidenceIntervalConfig(
        #            legend_loc=None,
        #            fill_alpha=0.1,
        #        ),
        #        matplotlib=qf.MatplotlibConfig(figsize=(7.5 / 2, 3 / 2)),
        #    ),
        # )
        # qf.PlotFrame.plot_confidence_asset_holdings(frames, mean_of_level=["run"])
        # qf.PlotFrame.plot_confidence_cash(frames, mean_of_level=["run"])
        qf.PlotFrame.plot_confidence_cumulative_rewards(frames, mean_of_level=["run"])

        combined_frame.plot_actions(
            plot_config=qf.PlotConfig.slim(),
            y_limits=(0, 0.2),
        )

    else:
        print("No runs found to plot")


if __name__ == "__main__":
    main()
