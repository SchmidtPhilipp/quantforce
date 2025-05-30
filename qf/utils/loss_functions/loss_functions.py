import torch
import torch.nn.functional as F
from qf.utils.correlation import compute_correlation

def weighted_mse_correlation_loss(predicted, target, actions, lambda_=1):
    """
    Custom loss function combining MSE and a correlation penalty.

    Parameters:
        predicted (torch.Tensor): Predicted Q-values.
        target (torch.Tensor): Target Q-values.
        actions (torch.Tensor): Actions taken by all agents (batch_size, n_agents, action_dim).
        lambda_ (float): Weighting factor between MSE and correlation penalty.

    Returns:
        torch.Tensor: Combined loss value.
    """
    # Compute the MSE loss
    mse_loss = F.mse_loss(predicted, target)

    # Compute the correlation penalty for each batch sample
    batch_size, n_agents, action_dim = actions.shape
    correlation_penalties = torch.zeros(batch_size, device=actions.device)

    penalty = torch.zeros(batch_size, device=actions.device)
    

    for b in range(batch_size):
        penalty = 0.0

        # Compute pairwise correlations between agents
        for i in range(n_agents):
            for j in range(i + 1, n_agents):

                # Use the dedicated correlation function
                penalty += compute_correlation(actions[b,i],actions[b,j])


        correlation_penalties[b] = penalty

    # Average the correlation penalties across the batch
    avg_correlation_penalty = correlation_penalties.mean()

    # Combine MSE loss and correlation penalty
    loss = lambda_ * mse_loss + (1 - lambda_) * avg_correlation_penalty
    return loss