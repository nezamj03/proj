import torch
from ..networks.mlp import MLP
import torch.optim as optim
import torch.nn.functional as F
import os

class PPOAgent:
    def __init__(self, config):
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.gamma = config.get('gamma', 0.99)
        self.clip_param = config.get('clip_param', 0.2)
        self.update_epochs = config.get('update_epochs', 5)
        self.lr = config.get('lr', 0.0003)
        
        # Policy network (assumed MLP model)
        self.policy = MLP(self.state_size, self.action_size, config)
        self.critic = MLP(self.state_size, 1, config)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.critic.parameters()), lr=self.lr)

    def act(self, state, action_mask):
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mask = torch.FloatTensor(action_mask).unsqueeze(0)
        
        with torch.no_grad():  # No gradient computation for action selection
            logits = self.policy(state)
            masked_logits = logits.masked_fill(action_mask == 0, float('-inf'))
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state).squeeze()

        return action.item(), log_prob, value.item()



    def learn(self, batch):

        obs = torch.stack([torch.FloatTensor(state) for episode in batch for state in episode['observations']])
        actions = torch.LongTensor([action for episode in batch for action in episode['actions']])
        rewards = torch.FloatTensor([rewards for episode in batch for rewards in episode['rewards']])
        next_obs = torch.stack([torch.FloatTensor(state) for episode in batch for state in episode['next_observations']])
        dones = torch.FloatTensor([dones for episode in batch for dones in episode['dones']])
        old_log_probs = torch.FloatTensor([log_probs for episode in batch for log_probs in episode['log_probs']])
        values = torch.FloatTensor([values for episode in batch for values in episode['values']])
        action_masks = torch.stack([torch.BoolTensor(mask) for episode in batch for mask in episode['action_masks']])

        with torch.no_grad():  # Disable gradient computation for target value calculation
            next_values = self.critic(next_obs).detach().squeeze()
            
        # Calculate returns and advantages
        returns, advantages = self._calculate_advantages(rewards, dones, values, next_values)

        for _ in range(self.update_epochs):
            # Recompute log_probs, values for the updated policy and critic
            logits = self.policy(obs)
            logits = logits.masked_fill(action_masks == 0, float('-inf'))  # Apply the mask
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            # Compute ratios
            ratio = (new_log_probs - old_log_probs).exp()

            # Compute objective functions
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()

            critic_loss = F.mse_loss(values, returns)

            # Total loss
            loss = actor_loss + 0.5 * critic_loss

            # Backpropagation
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _calculate_advantages(self, rewards, dones, values, next_values):
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        for i in reversed(range(len(rewards))):
            advantages[i] = last_advantage = deltas[i] + self.gamma * last_advantage * (1 - dones[i])

        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return returns, advantages

    def save(self, directory):
        """
        Saves the model's actor and critic networks to the specified directory.

        Args:
            directory (str): The directory path to save the model files.
        """
        os.makedirs(directory, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(directory, 'policy.pt'))
        torch.save(self.critic.state_dict(), os.path.join(directory, 'critic.pt'))
