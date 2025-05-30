import torch
import torch.optim as optim
import numpy as np
import random
from tqdm import tqdm
from qf.agents.utils.model_builder import ModelBuilder
from qf.agents.buffers.replay_buffer import ReplayBuffer
from qf.agents.agent import Agent

import qf as qf

# Attention this is Soft-DQN
class DQNAgent(Agent):
    def __init__(self, 
                env,
                actor_config=None, 
                lr=qf.DEFAULT_DQN_LR, 
                gamma=qf.DEFAULT_DQN_GAMMA, 
                batch_size=qf.DEFAULT_DQN_BATCH_SIZE, 
                buffer_max_size=qf.DEFAULT_DQN_BUFFER_MAX_SIZE, 
                device=qf.DEFAULT_DEVICE):
        
        super().__init__(env=env)

        # Use the provided network architecture or a default one
        default_architecture = [
            {"type": "Linear", "params": {"in_features": self.obs_dim, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 128}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 128, "out_features": 64}, "activation": "ReLU"},
            {"type": "Linear", "params": {"in_features": 64, "out_features": self.act_dim}}
        ]
        actor_config = actor_config or default_architecture

        # Single-agent environment setup
        self.n_agents = 1
        # Check if the environment agent settings are compatible
        if hasattr(env, 'n_agents') and env.n_agents != self.n_agents:
            raise ValueError(f"Environment is configured for {env.n_agents} agents, but DQNAgent is set up for {self.n_agents} agents.")

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

    def train(self, episodes=10, use_tqdm=True):
        """
        Trains the agent for a specified number of episodes.
        Parameters:
            episodes (int): Number of episodes to train the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
        """
        progress = tqdm(range(episodes), desc="Training DQNAgent") if use_tqdm else range(episodes)

        for episode in progress:
            state = self.env.reset()
            done = False
            total_reward = 0
            epsilon = max(0.1, 1 - episode / episodes)  # Linear epsilon decay

            while not done:
                action = self.act(state, epsilon)
                next_state, reward, done, _ = self.env.step(action)
                self.memory.store((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                # Perform training step
                self._train_step()

            if use_tqdm:
                progress.set_postfix({"Episode Reward": total_reward})

    def _train_step(self):
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

    def evaluate(self, eval_env, episodes=1, use_tqdm=True):
        """
        Evaluates the agent for a specified number of episodes on the environment.
        Parameters:
            eval_env: The environment used for evaluation.
            episodes (int): Number of episodes to evaluate the agent.
            use_tqdm (bool): If True, use tqdm for progress tracking; otherwise, print episode summaries.
        Returns:
            float: The average reward over the evaluation episodes.
        """
        total_rewards = []
        progress = tqdm(range(episodes), desc="Evaluating DQNAgent", ncols=80) if use_tqdm else range(episodes)

        for episode in progress:
            state = eval_env.reset()
            done = False
            episode_reward = 0

            while not done:
                # Select action using the trained model (no exploration during evaluation)
                action = self.act(state, epsilon=0.0)
                next_state, reward, done, _ = eval_env.step(action)
                state = next_state
                episode_reward += reward

            total_rewards.append(episode_reward)

            if use_tqdm:
                progress.set_postfix({"Episode Reward": episode_reward})

        eval_env.print_metrics()
        avg_reward = np.mean(total_rewards)
        print(f"Average reward over {episodes} episodes: {avg_reward}")
        return avg_reward

    def save(self, path):
        """
        Saves the agent's model to a file.
        Parameters:
            path (str): Path to save the model.
        """
        torch.save(self.model.state_dict(), path)

    def load(self, path):
        """
        Loads the agent's model from a file.
        Parameters:
            path (str): Path to load the model from.
        """
        self.model.load_state_dict(torch.load(path, map_location=self.device))
        self.model.to(self.device)
