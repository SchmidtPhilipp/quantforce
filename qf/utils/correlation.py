import torch

def compute_correlation(a_i, a_j):
    """
    Computes the correlation between two action vectors.

    Parameters:
        a_i (torch.Tensor): Action vector of the first agent.
        a_j (torch.Tensor): Action vector of the second agent.

    Returns:
        float: Squared correlation between the two action vectors.
    """
    mean_i = a_i.mean()
    mean_j = a_j.mean()
    std_i = a_i.std()
    std_j = a_j.std()

    # Avoid division by zero
    if std_i > 0 and std_j > 0:
        correlation = ((a_i - mean_i) * (a_j - mean_j)).mean() / (std_i * std_j)
        return correlation**2
    return 0.0


def per_sample_correlation_penalty(actions: torch.Tensor, i: int) -> torch.Tensor:
    """
    Compute per-sample correlation penalty between agent i and all others (j ≠ i).
    
    Parameters:
        actions: Tensor of shape [batch_size, n_agents, action_dim]
        i: Index of the reference agent (int)

    Returns:
        correlation_penalty: Tensor of shape [batch_size]
    """
    batch_size, n_agents, action_dim = actions.shape

    # Center actions per sample (zero-mean across action_dim)
    actions_centered = actions - actions.mean(dim=2, keepdim=True)  # [B, A, D]

    # Normalize per sample (across action_dim)
    norm = actions_centered.norm(dim=2, keepdim=True) + 1e-8
    actions_normalized = actions_centered / norm  # [B, A, D]

    # Select agent i: [B, D]
    a_i = actions_normalized[:, i, :]  # [B, D]

    # Compute dot product with all agents: [B, D] × [B, A, D] -> [B, A]
    dot_products = torch.sum(a_i.unsqueeze(1) * actions_normalized, dim=2)  # [B, A]

    # Mask out self-correlation at position i
    mask = torch.ones(n_agents, device=actions.device)
    mask[i] = 0.0

    # Apply mask and sum over agents j ≠ i → [B]
    correlation_penalty = torch.sum(dot_products * mask, dim=1)

    return correlation_penalty  # shape: [batch_size]