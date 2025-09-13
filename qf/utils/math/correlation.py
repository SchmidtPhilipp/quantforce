import torch


def correlation(a_i: torch.Tensor, a_j: torch.Tensor) -> torch.Tensor:
    """
    Computes the correlation between two action vectors.

    Parameters:
        a_i (torch.Tensor): Action vector of the first agent.
        a_j (torch.Tensor): Action vector of the second agent.

    Returns:
        torch.Tensor: Squared correlation between the two action vectors.
    """
    mean_i = a_i.mean()
    mean_j = a_j.mean()
    std_i = a_i.std()
    std_j = a_j.std()

    # Avoid division by zero
    if std_i > 0 and std_j > 0:
        correlation = ((a_i - mean_i) * (a_j - mean_j)).mean() / (std_i * std_j)
        return correlation**2
    return torch.tensor(0.0)
