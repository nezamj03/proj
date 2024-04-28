import os
from datetime import datetime
from .ppo import PPOAgent
from ..utils.buffer import TimestepBuffer
from ..agents.random import RandomAgent
from ..agents.always_offer import AlwaysOfferAgent

SAVE_PATH = 'res/'

class PPORunner:
    """
    Orchestrates the setup, execution, and training of PPO agents within a specified environment.
    Supports different configurations of agent interactions and training regimes.

    Attributes:
        config (dict): Configuration settings for the runner and agents.
        logger (Logger): Logger for recording training progress and statistics.
        env_fn (callable): Function to initialize the environment.
        agents (dict): Dictionary of agents participating in the environment.
        learners (set): Set of agents that are learning (i.e., being trained).
        replays (dict): Dictionary of replay buffers for learning agents.
        selected_learner (tuple): Tuple containing the key and the instance of the primary learner.
    """
    def __init__(self, env_fn, config, logger):
        self.env_fn = env_fn
        self.config = config
        self.logger = logger
        self.batch_size = config['batch_size']
        self.total_train_t = config["total_train_t"]
        self.n_agents = config["env_args"]["num_players"]
        self.setup()

    def setup(self):
        """
        Initializes the environment, agents, and any necessary variables.
        """
        self.env = self.env_fn(**self.config['env_args'])
        self.env.reset()
        self.build_agents()
        self.train_t = 0

    def build_agents(self):
        """
        Initializes agents based on configuration for self-play or interaction with different types of agents.
        """
        standard_learner = 'player_0'
        agent_configs = {
            'self_play': (PPOAgent, self.env.agents),
            'vs_random': (RandomAgent, self.env.agents.difference({standard_learner})),
            'vs_always': (AlwaysOfferAgent, self.env.agents.difference({standard_learner}))
        }
        agent_class, non_learner_agents = agent_configs.get(self.config.get('agents', 'self_play'), (PPOAgent, self.env.agents))

        self.agents = {agent: agent_class(self.config) if agent == standard_learner or agent_class == PPOAgent else agent_class()
                       for agent in self.env.agents}
        self.learners = {standard_learner} if agent_class != PPOAgent else self.env.agents
        self.replays = {agent: TimestepBuffer(self.config['buffer']) for agent in self.learners}
        self.selected_learner = standard_learner, self.agents[standard_learner]

    def rollout(self):
        """
        Executes one complete episode of interaction in the environment.
        """
        obs, _ = self.env.reset()
        history = {agent: {'observations': [], 'actions': [], 'rewards': [], 'next_observations': [], 'dones': [],
                           'log_probs': [], 'values': [], 'action_masks': []} for agent in self.learners}
        while self.env.agents:
            actions, log_probs, values = {}, {}, {}
            for agent in self.env.agents:
                result = self.agents[agent].act(state=obs[agent]["observation"], action_mask=obs[agent]["action_mask"])
                if agent in self.learners:
                    actions[agent], log_probs[agent], values[agent] = result
                else:
                    actions[agent] = result

            next_obs, rewards, terminations, truncations, _ = self.env.step(actions)
            for agent in self.env.agents:
                if agent in self.learners:
                    self.update_history(history, agent, obs, actions, rewards, next_obs, terminations, truncations, log_probs, values)

            self.train_t += 1
            obs = next_obs
        return history

    def update_history(self, history, agent, obs, actions, rewards, next_obs, terminations, truncations, log_probs, values):
        """
        Updates the history dictionary with new timestep information for a given agent.
        """
        history[agent]["observations"].append(obs[agent]["observation"])
        history[agent]["actions"].append(actions[agent])
        history[agent]["rewards"].append(rewards[agent])
        history[agent]["next_observations"].append(next_obs[agent]["observation"])
        history[agent]["dones"].append(terminations[agent] or truncations[agent])
        history[agent]["log_probs"].append(log_probs[agent])
        history[agent]["values"].append(values[agent])
        history[agent]["action_masks"].append(obs[agent]["action_mask"])

    
    def train(self):

        self.logger.info("Starting Training")

        self.train_t = 0
        self.episode = 0

        episode_returns = {agent : [] for agent in self.agents.keys()}

        while self.train_t <= self.total_train_t:

            episode_batch = self.rollout()
            self.episode += 1

            for agent in episode_batch:
                if agent in self.learners:
                    self.replays[agent].insert_episode(episode_batch[agent])
                episode_returns[agent].append(sum(episode_batch[agent]["rewards"]))

            if self.train_t % self.config['learn_freq'] == 0:
                for agent in self.learners:
                    if len(self.replays[agent]) > self.batch_size:
                        batch = self.replays[agent].sample(self.batch_size)
                        self.agents[agent].set_lr(self.train_t)
                        self.agents[agent].learn(batch)

            if self.config['save'] and \
                    self.train_t > 0 and \
                    self.train_t % (self.total_train_t // self.config['save_count']) == 0:
                token = f'{datetime.now().strftime("%Y%m%d")}'
                save_path = os.path.join(SAVE_PATH, "models", "PPO", token, str(self.train_t))
                self.selected_learner[1].save(save_path)
                self.logger.info(f"Saved PPO Model at t={self.train_t}/{self.total_train_t}")

            if self.train_t > 0 and self.train_t % self.config['stats_freq'] == 0:
                timestep = f"episode = {self.episode}, time = {self.train_t} of {self.total_train_t}"
                self.logger.stat(timestep, episode_returns[self.selected_learner[0]])

        self.logger.info("Finished Training")
        return episode_returns[self.selected_learner[0]]
