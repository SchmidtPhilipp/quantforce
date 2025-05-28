from agents.base_agent import BaseAgent
import torch
import torch.optim as optim
import numpy as np
import random
from agents.model_builder import ModelBuilder
from agents.buffers.replay_buffer import ReplayBuffer


# Attention this is Soft-DQN
class DQNAgent(BaseAgent):
    def __init__(self, obs_dim, act_dim, actor_config=None, lr=1e-3, gamma=0.99, batch_size=32, buffer_max_size=100000, device="cpu"):
        super().__init__()

        # Use the provided network architecture or a default one
        default_architecture = [
            {"type": "Linear", "params": {"in_features": obs_dim, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 64}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 64, "out_features": act_dim}}
        ]
        actor_config = actor_config or default_architecture
        self.n_agents = 1
        self.device = device
        # Use ModelBuilder to create the models
        self.model = ModelBuilder(actor_config).build().to(self.device)
        self.target_model = ModelBuilder(actor_config).build().to(self.device)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.batch_size = batch_size
        self.loss_fn = torch.nn.MSELoss()
        self.memory = ReplayBuffer(capacity=buffer_max_size)  # Initialize the replay buffer

        self.buffer_max_size = buffer_max_size

    def act(self, state: torch.Tensor, epsilon: float = 0.0) -> torch.Tensor:
        """
        Gibt eine Wahrscheinlichkeitsverteilung (Länge = act_dim, Summe = 1)
        """
        state = state.to(self.device)#.unsqueeze(0)  # [1, obs_dim]

        with torch.no_grad():
            logits = self.model(state)  # [1, act_dim]

            if random.random() < epsilon:
                # Uniforme Zufallsverteilung
                probs = torch.rand_like(logits)
                probs = probs / probs.sum(dim=1, keepdim=True)
            else:
                probs = torch.softmax(logits, dim=1)  # Softmax Q → Verteilung

        return probs#.squeeze(0)  # [act_dim]

    def store(self, transition):
        self.memory.store(transition)


    def train(self):
        if len(self.memory) < self.batch_size:
            return
        
        # Sample a batch from memory
        states, actions, rewards, next_states, _ = self.memory.sample(self.batch_size)

        # Ensure states, actions, rewards, and next_states are tensors
        if isinstance(states[0], torch.Tensor):
            states = torch.stack(states).to(self.device)
            actions = torch.stack(actions).to(self.device)
            rewards = torch.stack(rewards).to(self.device)
            next_states = torch.stack(next_states).to(self.device)
        else:
            states = torch.FloatTensor(np.array(states)).to(self.device)
            actions = torch.LongTensor(np.array(actions)).to(self.device)
            rewards = torch.FloatTensor(np.array(rewards)).to(self.device)
            next_states = torch.FloatTensor(np.array(next_states)).to(self.device)

        # Compute predictions and targets
        pred = self.model(states)
        with torch.no_grad():
            target = rewards + self.gamma * self.target_model(next_states).max(dim=1).values

        # Compute loss
        loss = self.loss_fn(pred.max(dim=1).values, target)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
