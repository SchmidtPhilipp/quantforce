import numpy as np

def log_weights(logger, tickers, action, step=None):
    """
    Logs the portfolio weights for each ticker and cash.

    Parameters:
        logger (Logger): The logger instance.
        tickers (list[str]): List of ticker symbols.
        action (list[float]): Portfolio weights (including cash).
        step (int, optional): The current step in the evaluation. Default is None.
    """
    # Normalize weights to ensure they sum to 1
    weights = np.clip(action, 0, 1)
    weights /= np.sum(weights) + 1e-8

    # Separate cash weight and asset weights
    cash_weight = weights[-1]
    asset_weights = weights[:-1]

    # Log cash weight
    if step is not None:
        logger.log_scalar("03_weights/cash", cash_weight, step=step)
    else:
        logger.log_scalar("03_weights/cash", cash_weight)

    # Log asset weights
    for i, ticker in enumerate(tickers):
        if step is not None:
            logger.log_scalar(f"03_weights/{ticker}", asset_weights[i], step=step)
        else:
            logger.log_scalar(f"03_weights/{ticker}", asset_weights[i])


