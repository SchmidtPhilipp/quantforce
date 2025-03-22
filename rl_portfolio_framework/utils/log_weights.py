import numpy as np

def log_weights(logger, tickers, action):
    weights = np.clip(action, 0, 1)
    weights /= np.sum(weights) + 1e-8

    cash_weight = weights[-1]
    asset_weights = weights[:-1]

    logger.log_scalar("03_weights/cash", cash_weight)
    for i, ticker in enumerate(tickers):
        logger.log_scalar(f"03_weights/{ticker}", asset_weights[i])


