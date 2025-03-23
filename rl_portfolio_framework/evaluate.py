from trainer.logger import Logger
from utils.metrics import (
    calculate_returns, sharpe_ratio, sortino_ratio,
    max_drawdown, volatility, cumulative_return,
    annualized_return, calmar_ratio
)
from utils.log_weights import log_weights
import numpy as np


def runs_single_evaluation(env, agent):
    state = env.reset()
    done = False
    balances = [env.balance]
    weights_over_time = []

    while not done:
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        balances.append(env.balance)
        state = next_state
        weights_over_time.append(action)

    return balances, weights_over_time


def evaluate_agent(env, agent, config, run_name=None, n_runs=100):
    tickers = config["tickers"]
    run_name = "EVAL_" + (run_name or "default")
    logger = Logger(run_name=run_name)

    metrics = {
        "sharpe": [],
        "sortino": [],
        "drawdown": [],
        "volatility": [],
        "cumulative": [],
        "annualized": [],
        "calmar": [],
    }

    for run in range(n_runs):
        balances, weights_over_time = runs_single_evaluation(env, agent)

        # Store data
        logger.add_run_data(balances, weights_over_time)

        # Log portfolio value at last timestep
        logger.log_scalar("02_eval/portfolio_value_final", balances[-1])

        # Log final weights
        log_weights(logger, tickers, weights_over_time[-1])

        logger.next_step()
    
        # Compute metrics
        returns = calculate_returns(balances)
        drawdown = max_drawdown(balances)
        annual_ret = annualized_return(balances)

        metrics["sharpe"].append(sharpe_ratio(returns))
        metrics["sortino"].append(sortino_ratio(returns))
        metrics["drawdown"].append(drawdown)
        metrics["volatility"].append(volatility(returns))
        metrics["cumulative"].append(cumulative_return(balances))
        metrics["annualized"].append(annual_ret)
        metrics["calmar"].append(calmar_ratio(annual_ret, drawdown))

    # Log portfolio mean and std over time
    balances_array = np.array(logger.balances)  # shape: (n_runs, n_steps)
    mean_balances = np.mean(balances_array, axis=0)
    std_balances = np.std(balances_array, axis=0)

    logger.step = 0

    for t in range(len(mean_balances)):
        logger.log_scalar("02_eval/portfolio_value_mean", mean_balances[t])
        logger.log_scalar("02_eval/portfolio_value_std", std_balances[t])
        logger.next_step()

    logger.step = 0
    
    # Log metrics as scalars + emulated histograms
    logger.log_metrics(metrics)
    for name, values in metrics.items():
        logger.log_emulated_histogram(name, values, bins=10)

    # Save everything to disk
    logger.save_evaluation_data(config=config)
    logger.close()

    # Print summary
    print(f"\n📊 Evaluation Summary ({n_runs} runs):")
    for name, values in metrics.items():
        print(f"{name.capitalize():20s}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print("✅ Evaluation completed, logged, and saved.")
    return metrics
