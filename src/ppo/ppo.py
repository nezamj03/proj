import os
import torch
import torch.optim as optim
import torch.nn.functional as F
from ..networks.mlp import MLP
from ..utils.schedule import REGISTRY as SCHEDULE_REGISTRY

class PPOAgent:
    """
    Proximal Policy Optimization (PPO) Agent with separate policy and critic networks.

    Attributes:
        state_size (int): Dimensionality of the state space.
        action_size (int): Dimensionality of the action space.
        gamma (float): Discount factor for future rewards.
        clip_param (float): The PPO clipping parameter used to bound the objective function.
        update_epochs (int): Number of epochs to update the networks for each sampled batch.
        policy (MLP): The policy network that predicts action probabilities.
        critic (MLP): The value network that estimates state values.
        optimizer (torch.optim.Optimizer): Optimizer for both policy and critic networks.
    """
    def __init__(self, config):
        self.state_size = config['state_size']
        self.action_size = config['action_size']
        self.gamma = config.get('gamma', 0.99)
        self.clip_param = config.get('clip_param', 0.2)
        self.update_epochs = config.get('update_epochs', 5)
        
        # Initialize the learning rate schedule
        lr_schedule = SCHEDULE_REGISTRY[config.get('lr_schedule_decay', 'linear')]
        self.lr_schedule = lr_schedule(config.get('lr_start', 0.002),
                                       config.get('lr_end', 0.001),
                                       config.get('lr_anneal', 100_000))
        
        # Initialize policy and critic networks
        self.policy = MLP(self.state_size, self.action_size, config)
        self.critic = MLP(self.state_size, 1, config)
        self.optimizer = optim.Adam(list(self.policy.parameters()) + list(self.critic.parameters()))
        self.set_lr(0)  # Set the initial learning rate

    def act(self, state, action_mask):
        """
        Generate an action, log probability, and value estimate for the given state and action mask.

        Args:
            state (list or np.array): Current environment state.
            action_mask (list or np.array): Mask that indicates valid actions.

        Returns:
            tuple: Tuple containing the action, log probability of the action, and value estimate.
        """
        state = torch.FloatTensor(state).unsqueeze(0)
        action_mask = torch.FloatTensor(action_mask).unsqueeze(0)

        with torch.no_grad():
            logits = self.policy(state)
            masked_logits = logits.masked_fill(action_mask == 0, float('-inf'))
            probs = F.softmax(masked_logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            value = self.critic(state).squeeze()

        return action.item(), log_prob, value.item()

    def set_lr(self, time):
        """
        Adjusts the learning rate of the optimizer according to the scheduled learning rate.
        
        Args:
            time (int): Current timestep used to determine the learning rate.
        """
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr_schedule(time)

    def learn(self, batch):
        """
        Updates policy and critic networks using the provided batch of experience.

        Args:
            batch (list): Batch of experiences to update the networks.
        """
        obs = torch.stack([torch.FloatTensor(timestep['observations']) for timestep in batch])
        actions = torch.LongTensor([timestep['actions'] for timestep in batch])
        rewards = torch.FloatTensor([timestep['rewards'] for timestep in batch])
        next_obs = torch.stack([torch.FloatTensor(timestep['next_observations']) for timestep in batch])
        dones = torch.FloatTensor([timestep['dones'] for timestep in batch])
        old_log_probs = torch.FloatTensor([timestep['log_probs'] for timestep in batch])
        values = torch.FloatTensor([timestep['values'] for timestep in batch])
        action_masks = torch.stack([torch.FloatTensor(timestep['action_masks']) for timestep in batch])

        # No gradient computation needed here
        with torch.no_grad():
            next_values = self.critic(next_obs).detach().squeeze()

        returns, advantages = self._calculate_advantages(rewards, dones, values, next_values)

        for _ in range(self.update_epochs):
            logits = self.policy(obs)
            logits = logits.masked_fill(action_masks == 0, float('-inf'))
            probs = F.softmax(logits, dim=-1)
            dist = torch.distributions.Categorical(probs)
            new_log_probs = dist.log_prob(actions)

            ratio = (new_log_probs - old_log_probs).exp()
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_param, 1 + self.clip_param) * advantages
            actor_loss = -torch.min(surr1, surr2).mean()
            critic_loss = F.mse_loss(values, returns)
            entropy = dist.entropy().mean()
            loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def _calculate_advantages(self, rewards, dones, values, next_values):
        """
        Calculates the generalized advantage estimates (GAE) for the batch of experience.

        Args:
            rewards (torch.Tensor): Tensor of rewards.
            dones (torch.Tensor): Tensor indicating if an episode has ended.
            values (torch.Tensor): Tensor of value estimates.
            next_values (torch.Tensor): Tensor of next value estimates.

        Returns:
            tuple: Tuple containing returns and advantages as tensors.
        """
        deltas = rewards + self.gamma * next_values * (1 - dones) - values
        advantages = torch.zeros_like(rewards)
        last_advantage = 0
        for i in reversed(range(len(rewards))):
            last_advantage = deltas[i] + self.gamma * last_advantage * (1 - dones[i])
            advantages[i] = last_advantage
        returns = advantages + values
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-10)
        return returns, advantages

    def save(self, directory):
        """
        Saves the policy and critic network models to the specified directory.

        Args:
            directory (str): Path to save the model files.
        """
        os.makedirs(directory, exist_ok=True)
        torch.save(self.policy.state_dict(), os.path.join(directory, 'policy.pt'))
        torch.save(self.critic.state_dict(), os.path.join(directory, 'critic.pt'))
