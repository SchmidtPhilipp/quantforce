import numpy as np
import torch


def risk_limiting_strategy(actions, last_actions, n_agents):
    epsilon = 0.01  # Initial epsilon value
    max_epsilon = 0.1
    epsilon_growth_rate = 0.001

    constrained_actions = []
    for i in range(n_agents):
        lower_bound = last_actions[i] - epsilon
        upper_bound = last_actions[i] + epsilon

        constrained_action = torch.clamp(actions[i], lower_bound, upper_bound)
        constrained_actions.append(constrained_action)

        deviation = torch.abs(constrained_action - last_actions[i]).mean()
        epsilon = max(0.1, epsilon + epsilon_growth_rate - deviation.item() * 0.05)
        epsilon = min(epsilon, max_epsilon)

    return torch.stack(constrained_actions)

class TimeVariantActionFilter:
    def __init__(self, n_agents, n_assets, epsilon=0.2, alpha=0.01):
        self.n_agents = n_agents
        self.epsilon_max = epsilon # maximum epsilon value
        self.epsilon = epsilon # Epsilon schlauch
        self.alpha = alpha # acceleration of the filter
        self.n_assets = n_assets

        self.epsilon = np.ones((n_agents, n_assets)) * epsilon # Epsilon schlauch for each agent
    
    def filter(self, actions: torch.Tensor, last_actions: torch.Tensor, n_agents, epsilon=0.1):

        for i in range(n_agents):
            for a in range(len(actions[i])):
                last_action = last_actions[i][a]
                action = actions[i][a]
                if action > last_action:
                    # we map the distance between the last action and 1 to the distance between the last action + epsilon and 1
                    # which will map the current action into the range [last_actionm, last_action + epsilon]
                    actions[i][a] = last_action + epsilon * (action - last_action) / (1 - last_action)
                else: # action < last_action:
                    # we map the distance between the last action and 0 to the distance between the last action - epsilon and 0
                    # which will map the current action into the range [last_action - epsilon, last_action]
                    actions[i][a] = last_action - epsilon * (last_action - action) / last_action

                distance = actions[i][a] - last_action

                # if the distance is larger than half epsilon then we decrease epsilon for the next step
                if abs(distance) > epsilon / 2:
                    self.epsilon = max(0.1, self.epsilon - self.alpha * abs(distance))
                # if the distance is smaller than half epsilon then we increase epsilon for the next step
                else:
                    self.epsilon = min(self.epsilon_max, self.epsilon + self.alpha * abs(distance))

            # nomalize the actions
            actions[i] = actions[i] / actions[i].sum()
        return actions

# Implement CPPI risk limiting strategy
#def cppi_risk_limiting_strategy(actions, last_actions, n_agents, cushion=0.1):


class CPPIActionFilter:
    def __init__(self, n_agents, cushion=0.1):
        self.n_agents = n_agents
        self.cushion = cushion

    def filter(self, actions, last_actions, n_agents):
        for i in range(n_agents):
            # Calculate the cushion
            cushion = np.maximum(0, last_actions[i] - self.cushion)
            # Apply the CPPI strategy
            actions[i] = actions[i] * (1 + cushion)
            # Normalize the actions
            actions[i] = actions[i] / np.sum(actions[i])
        return actions

