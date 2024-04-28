import torch
import torch.optim as optim
import torch.nn.functional as F
from ..networks.mlp import MLP
from ..utils.schedule import REGISTRY as SCHEDULE_REGISTRY
import numpy as np
from ..agents.base import BaseAgent
import os

class DQNAgent(BaseAgent):
    """
    Implements a Deep Q-Network (DQN) agent with separate target and behavior models,
    epsilon-greedy exploration, and configurable scheduling for learning rate and epsilon.

    Attributes:
        model (MLP): The main neural network model used for Q-value predictions.
        target_model (MLP): A copy of the main model used to generate target Q-values for stability.
        optimizer (torch.optim.Optimizer): Optimizer for the main network.
        gamma (float): Discount factor for future rewards.
        lr_schedule (Schedule): Learning rate schedule.
        epsilon_schedule (Schedule): Epsilon (exploration rate) schedule.
    """

    def __init__(self, config):
        # Extract configuration and initialize schedules
        self.state_size = config.get('state_size', 6)
        self.action_size = config.get('action_size', 2)
        lr_schedule = SCHEDULE_REGISTRY[config.get('lr_schedule_decay', 'linear')]
        self.lr_schedule = lr_schedule(config.get('lr_start', 0.002),
                                       config.get('lr_end', 0.001),
                                       config.get('lr_anneal', 100_000))
        self.gamma = config.get('gamma', 1.0)
        eps_schedule = SCHEDULE_REGISTRY[config.get('epsilon_schedule_decay', 'linear')]
        self.epsilon_schedule = eps_schedule(config.get('epsilon_start', 1),
                                             config.get('epsilon_end', 0.1),
                                             config.get('epsilon_anneal', 100_000))

        # Initialize neural networks and optimizer
        self.model = MLP(self.state_size, self.action_size, config)
        self.target_model = MLP(self.state_size, self.action_size, config)
        self.sync()
        self.optimizer = optim.Adam(self.model.parameters())
        self.set_lr(0)

    def sync(self):
        """Synchronize the target model's weights with the main model's weights."""
        self.target_model.load_state_dict(self.model.state_dict())

    def set_lr(self, time: int):
        """Updates the learning rate in the optimizer according to the schedule."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_schedule(time)

    def act(self, state, action_mask, **kwargs):
        """
        Selects an action using an epsilon-greedy strategy.

        Args:
            state: The current state of the environment.
            action_mask: A mask indicating valid actions.
            **kwargs: Additional parameters, e.g., current timestep or test mode flag.

        Returns:
            The action selected by the agent.
        """
        t = kwargs['t']
        test = kwargs.get('test', False)
        eps = self.epsilon_schedule.get(t) if not test else 0
        if np.random.random() < eps:
            return np.random.choice(np.flatnonzero(action_mask))

        # Compute action values and apply mask
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mask = torch.BoolTensor(action_mask).unsqueeze(0)
        q_values = self.model(state)
        q_values = torch.where(action_mask, q_values, torch.tensor(float('-inf')))
        return q_values.argmax().item()

    def learn(self, batch):
        """
        Updates the network weights based on a batch of experience.

        Args:
            batch: A batch of experience tuples.
        """
        obs = torch.stack([torch.FloatTensor(t['observations']) for t in batch])
        actions = torch.LongTensor([t['actions'] for t in batch])
        rewards = torch.FloatTensor([t['rewards'] for t in batch])
        next_obs = torch.stack([torch.FloatTensor(t['next_observations']) for t in batch])
        dones = torch.FloatTensor([t['dones'] for t in batch])

        # Compute target Q values
        with torch.no_grad():
            q_values_next = self.target_model(next_obs).max(1)[0]
        q_targets = rewards + self.gamma * q_values_next * (1 - dones)

        # Update model
        q_values = self.model(obs).gather(1, actions.unsqueeze(1)).squeeze(1)
        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, directory: str):
        """
        Saves the model's weights to a file.

        Args:
            directory (str): The directory path where the model weights will be saved.
        """
        os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(directory, 'q_network.pt'))
