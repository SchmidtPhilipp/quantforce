# QuantForce

## Overview

QuantForce is a comprehensive, configuration-driven machine learning framework that improves existing frameworks like FinRL [1] by unifying classical portfolio theory, modern continuous-time optimization, and reinforcement learning approaches within a single experimental environment. Building upon the solid foundation that FinRL provides for deep reinforcement learning in quantitative finance, QuantForce extends and enhances these capabilities to address the broader needs of academic portfolio optimization research.

### Relationship to FinRL

FinRL [1] has made significant contributions to applying deep reinforcement learning in quantitative finance, providing a solid foundation for automated trading research. QuantForce builds upon this foundation while extending the scope to include classical and modern portfolio theory methods alongside reinforcement learning approaches.

**QuantForce's Specific Contributions:**
- **Multi-Paradigm Integration**: Integrates classical (Markowitz) and modern (HJB) portfolio methods alongside reinforcement learning for comprehensive comparison within a single framework
- **Academic Research Focus**: Provides tools specifically designed for rigorous academic research workflows and publication-ready outputs
- **Advanced Financial Time Series Processing**: Includes sophisticated missing data imputation using Kalman filters, GBM bridges, and 15+ specialized financial methods
- **Configuration-Driven Approach**: Enables researchers to focus on methodology through hierarchical configuration objects rather than implementation details
- **Publication Tools**: Offers direct LaTeX/TikZ output for academic papers and automated metrics table generation

The framework supports:
- **Classical Methods**: Markowitz mean-variance optimization, 1/N portfolio, random portfolios
- **Modern Methods**: Hamilton-Jacobi-Bellman (HJB) optimal control with and without transaction costs
- **Reinforcement Learning**: Deep Q-Networks (DQN), Soft Actor-Critic (SAC), Proximal Policy Optimization (PPO), Deep Deterministic Policy Gradient (DDPG), Twin Delayed DDPG (TD3), and more
- **Multi-Agent Systems**: MADDPG for multi-agent portfolio optimization
- **Advanced Data Processing**: Sophisticated missing data imputation using GBM bridges, Kalman filters, and other methods
- **Hyperparameter Optimization**: Grid search and Optuna-based optimization
- **Publication-Ready Visualization**: LaTeX/TikZ integration for academic papers

## Key Features

- **Unified Interface**: Single API for classical, modern, and RL approaches
- **Configuration-Driven**: Easy experimentation through configuration objects
- **Academic Focus**: Built specifically for portfolio optimization research
- **Publication Ready**: LaTeX integration for academic papers
- **Comprehensive**: 15+ portfolio optimization algorithms
- **Scalable**: Multi-agent and multi-seed experimental capabilities
- **Advanced Data Processing**: Sophisticated missing data handling
- **Performance Optimized**: Tensor operations and memory pooling
- **Extensible**: Easy to add new agents and environments

## Installation

The installation requirements for the subpackages is quite tight. 
Ensure to use python=3.10 otherwise some packages are incompatible. 
For generating LaTeX plots a local LaTeX installation is necessary. 

```bash
# Clone the repository
git clone https://github.com/SchmidtPhilipp/quantforce.git
cd quantforce

# Create and activate conda environment
conda env create -f qf/requirements/environment.yml
conda activate quantforce

# Install the package in development mode
pip install -e .
```

## Usage

QuantForce is a configuration-driven machine learning tool that enables researchers to easily set up, run, and compare different portfolio optimization approaches through simple configuration objects.

### Quick Start Example

### Colab Example

<a href="https://colab.research.google.com/github/SchmidtPhilipp/quantforce/blob/main/Quantforce_colab_example.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>


### (local) Example

```python
import qf

# Create environment with default configuration
train_env = qf.MultiAgentPortfolioEnv(
    "TRAIN", n_agents=1, env_config=qf.EnvConfig.get_default_train()
)

# Create a Markowitz agent with tangency portfolio configuration
agent = qf.ClassicOnePeriodMarkowitzAgent(
    train_env, 
    config=qf.agents.ClassicOnePeriodMarkowitzAgentConfig(
        target="Tangency",
        risk_model="sample_cov"
    )
)

# Train the agent
agent.train(total_timesteps=1000)

# Evaluate on test data
eval_env = qf.MultiAgentPortfolioEnv(
    "EVAL", n_agents=1, env_config=qf.EnvConfig.get_default_eval()
)
agent.evaluate(episodes=1, eval_env=eval_env)

# Visualize results
run = eval_env.data_collector
frame = run.get_frame()
frame.plot_balance()
frame.plot_rewards()
frame.plot_actions()
```

### Configuration System

The framework uses a hierarchical configuration system with three main components:

#### 1. Data Configuration
```python
# Configure data sources and preprocessing
data_config = qf.DataConfig(
    tickers=["AAPL", "MSFT", "GOOGL", "AMZN"],
    start="2018-01-01",
    end="2023-01-01",
    indicators=["rsi", "sma", "macd"],
    imputation_method="gbm_bridge_impute_with_global_variance",
    backfill_method="shrinkage"
)
```

#### 2. Environment Configuration
```python
# Configure trading environment
env_config = qf.EnvConfig(
    data_config=data_config,
    obs_window_size=60,
    initial_balance=1_000_000,
    trade_cost_percent=0.01,  # 1% transaction cost
    trade_cost_fixed=1.0,
    reward_function_config=qf.envs. CostAdjustedSharpeRatioConfig(
        type="cost_adjusted_sharpe_ratio",
        past_window=0,
        future_window=60,
        use_log_returns=False,
    )
)
```

#### 3. Agent Configuration
```python
# Configure different types of agents

# Classical Markowitz
markowitz_config = qf.agents.ClassicOnePeriodMarkowitzAgentConfig(
    target="Tangency",
    risk_model="ledoit_wolf",
    risk_free_rate=0.02
)

# HJB Optimal Control
hjb_config = qf.agents.HJBPortfolioAgentConfig(
    risk_aversion=2.0,
    time_horizon=252,
    solver_method="analytical" # Uses Merton solution
)

# Reinforcement Learning
sac_config = qf.agents.SACConfig(
    learning_rate=3e-4,
    batch_size=256,
    tau=0.005,
    gamma=0.99
)
```

### Multi-Agent Comparison

```python
# Compare multiple strategies
agents_to_compare = [
    ("Tangency", qf.ClassicOnePeriodMarkowitzAgent, markowitz_config),
    ("HJB Merton", qf.HJBPortfolioAgent, hjb_config),
    ("SAC", qf.SACAgent, sac_config),
    ("1/N", qf.OneOverNPortfolioAgent, qf.agents.OneOverNPortfolioAgentConfig())
]

results = []
for name, agent_class, config in agents_to_compare:
    agent = agent_class(train_env, config=config)
    agent.train(total_timesteps=10000)
    agent.evaluate(episodes=1, eval_env=eval_env)
    
    run = eval_env.data_collector
    run.rename(name)
    results.append(run)

# Create comparison plots
frames = [run.get_frame() for run in results]
comparison = qf.PlotFrame(pd.concat(frames, axis=1))
comparison.plot_balance()
comparison.metrics_table_comparison(frames)
```

### Hyperparameter Optimization

QuantForce provides powerful hyperparameter optimization capabilities:

#### Grid Search Optimization
```python
# Define hyperparameter space
agent_space = {
    "learning_rate": {
        "type": "float",
        "low": 1e-5,
        "high": 1e-2,
        "n_points": 5
    },
    "batch_size": {
        "type": "categorical",
        "choices": [64, 128, 256]
    }
}

env_space = {
    "trade_cost_percent": {
        "type": "float",
        "low": 0.0,
        "high": 0.02,
        "n_points": 3
    }
}

# Initialize optimizer
optimizer = qf.GridSearchOptimizer(
    agent_classes=[qf.SACAgent],
    agent_config=[agent_space],
    env_hyperparameter_space=env_space,
    optim_config={
        "objective": "avg_reward",
        "max_timesteps": 10000,
        "episodes": 5
    }
)

# Run optimization
results = optimizer.optimize()
optimizer.visualize_results()
optimizer.save_results()
```

#### Optuna-based Optimization
```python
from qf.optim.hyperparameter_optimizer import HyperparameterOptimizer

optimizer = HyperparameterOptimizer(
    agent_classes=[qf.SACAgent, qf.PPOAgent],
    optim_config={
        "objective": "avg_reward",
        "max_timesteps": 50000,
        "episodes": 10
    }
)

# Run Bayesian optimization
results = optimizer.optimize(n_trials=100)
optimizer.visualize_results()
```

### Advanced Features

#### Multi-Seeded Experiments
```python
# Run experiments with multiple random seeds for statistical significance
agents, train_runs, train_eval_runs, eval_runs = agent.multi_seeded_run(
    total_timesteps=100000,
    eval_env_config=eval_config,
    eval_every_n_steps=10000,
    n_eval_episodes=5,
    seeds=list(range(10))  # 10 different random seeds
)
```

#### LaTeX Integration for Publications
```python
# Generate publication-ready LaTeX plots
# Requirements: Latex

# After training or evaluating the agent get the data collector
run = eval_env.data_collector
run.rename(name) # You may rename the run name 

# Create comparison plots
frames = [run.get_frame() for run in results]
comparison = qf.PlotFrame(pd.concat(frames, axis=1))
comparison.plot_balance()
comparison.plot_rewards()
comparison.plot_actions()
comparison.plot_asset_holdings()
comparison.plot_cash()
comparison.plot_cumulative_rawards()

# For multi seeded runs one can also use a list of runs having the same "run" name and plot the confidence intervals of the runs. 
qf.PlotFrame.plot_confidence_rewards(frames, mean_of_level=["run"])
qf.PlotFrame.plot_confidence_balance(frames, mean_of_level=["run"])
qf.PlotFrame.plot_confidence_actions(frames, mean_of_level=["run"])
qf.PlotFrame.plot_confidence_asset_holdings(frames, mean_of_level=["run"])
qf.PlotFrame.plot_confidence_cash(frames, mean_of_level=["run"])
qf.PlotFrame.plot_confidence_cumulative_rewards(frames, mean_of_level=["run"])

# Generate a LaTeX table
comparison.metrics_table_comparison(frames)
comparison.metrics_table_comparison(frames, transpose=True) # You may also transpose the table. 

```

Without a local LaTeX installation it is still possible to make custom plots of the results using matplotlib.
Any contributions to the plotting functionalities are highly welcome! 

For more detailed examples, see the `01_scripts/00_examples/` directory:
- `01_basic_example.py` - Simple single-agent setup
- `02_multiple_agents.py` - Multi-agent comparison
- `03_random_portfolios/` - Large-scale random portfolio analysis

## Examples

The framework includes comprehensive examples in the `01_scripts/` directory:

- **`00_examples/`**: Basic usage examples and tutorials
- **`07_No_Cost/`**: Portfolio optimization without transaction costs
- **`08_Constant_Cost/`**: Portfolio optimization with constant transaction costs  
- **`09_Full_Cost/`**: Portfolio optimization with full transaction cost modeling

Each example directory contains:
- `start.py`: Main execution script
- `report.py`: Results analysis and visualization
- `plots/`: Generated visualizations in PNG and PGF formats
- `runs/`: Saved experimental results

## Citation

If you use QuantForce in your academic research, please cite:

```bibtex
@software{quantforce2024,
  title={QuantForce: A Unified Framework for Multi-Paradigm Portfolio Optimization Research},
  author={Schmidt, Philipp},
  year={2024},
  url={https://github.com/SchmidtPhilipp/quantforce},
  note={Academic portfolio optimization framework advancing beyond FinRL}
}
```

## References

[1] Liu, X. Y., Yang, H., Chen, Q., Zhang, R., Yang, L., Xiao, B., & Wang, C. D. (2020). FinRL: A deep reinforcement learning library for automated stock trading in quantitative finance. *Deep RL Workshop, NeurIPS 2020*.

[2] Markowitz, H. (1952). Portfolio selection. *The Journal of Finance*, 7(1), 77-91.

[3] Merton, R. C. (1971). Optimum consumption and portfolio rules in a continuous-time model. *Journal of Economic Theory*, 3(4), 373-413.

[4] Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O. (2017). Proximal policy optimization algorithms. *arXiv preprint arXiv:1707.06347*.

[5] Haarnoja, T., Zhou, A., Abbeel, P., & Levine, S. (2018). Soft actor-critic: Off-policy maximum entropy deep reinforcement learning with a stochastic actor. *International Conference on Machine Learning*, 1861-1870.

---

*QuantForce is designed for academic research in portfolio optimization and quantitative finance. For questions, collaborations, or contributions, please open an issue or pull request.*