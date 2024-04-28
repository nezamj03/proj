import torch
import torch.optim as optim
import torch.nn.functional as F
from ..networks.mlp import MLP
from ..utils.buffer import EpisodeBuffer
from ..utils.schedule import REGISTRY as SCHEDULE_REGISTRY
import numpy as np
from ..agents.base import BaseAgent
import os

class DQNAgent(BaseAgent):
    def __init__(self, config):

        self.state_size = config.get('state_size', 6)
        self.action_size = config.get('action_size', 2)
        lr_schedule = SCHEDULE_REGISTRY[config.get('lr_schedule_decay', 'linear')]
        self.lr_schedule = lr_schedule(config.get('lr_start', 0.002),
                              config.get('lr_end', 0.001),
                              config.get('lr_anneal', 100_000) )
        self.gamma = config.get('gamma', 1)
        eps_schedule = SCHEDULE_REGISTRY[config.get('epsilon_schedule_decay', 'linear')]
        self.epsilon_schedule = eps_schedule(config.get('epsilon_start', 1),
                                             config.get('epsilon_end', 0.1),
                                             config.get('epsilon_anneal', 100_000))

        # Network and Optimizer
        self.model = MLP(self.state_size, self.action_size, config)
        self.target_model = MLP(self.state_size, self.action_size, config)
        self.sync()
        self.optimizer = optim.Adam(self.model.parameters())
        self.set_lr(0)

    def sync(self):
        self.target_model.load_state_dict(self.model.state_dict())

    def set_lr(self, time):
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_schedule(time)

    def act(self, state, action_mask, **kwargs):
        t = kwargs['t']
        test = kwargs.get('test', False)
        
        eps = self.epsilon_schedule.get(t) if not test else 0
        rand = np.random.random()
        if rand < eps:
            return np.random.choice(np.flatnonzero(action_mask))
        
        with torch.no_grad():
            state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
            action_mask = torch.BoolTensor(action_mask).unsqueeze(0)
            q_values = self.model(state)
            q_values = torch.where(action_mask != 0, q_values, torch.tensor(float('-inf')))
        return q_values.argmax().item()
        

    def learn(self, batch):

        obs = torch.stack([torch.FloatTensor(timestep['observations']) for timestep in batch])
        actions = torch.LongTensor([timestep['actions'] for timestep in batch])
        rewards = torch.FloatTensor([timestep['rewards'] for timestep in batch])
        next_obs = torch.stack([torch.FloatTensor(timestep['next_observations']) for timestep in batch])
        dones = torch.FloatTensor([timestep['dones'] for timestep in batch])

        with torch.no_grad():  # No gradients needed when calculating the next Q values from the target model
            q_values_next = self.target_model(next_obs).max(1)[0]        
        q_targets = rewards + self.gamma * q_values_next * (1 - dones)
        q_values = self.model(obs).gather(1, actions.unsqueeze(1)).squeeze(1)

        loss = F.mse_loss(q_values, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def save(self, directory):
        """
        Saves the model's actor and critic networks to the specified directory.

        Args:
            directory (str): The directory path to save the model files.
        """
        os.makedirs(directory, exist_ok=True)
        torch.save(self.model.state_dict(), os.path.join(directory, 'q_network.pt'))
