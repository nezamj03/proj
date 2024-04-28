import os
from datetime import datetime
from ..agents.random import RandomAgent
from ..agents.always_offer import AlwaysOfferAgent
from ..utils.buffer import TimestepBuffer
from ..utils.schedule import REGISTRY as SCHEDULE_REGISTRY
from ..agents.always_offer import AlwaysOfferAgent
from ..agents.random import RandomAgent
from ..dqn.dqn import DQNAgent

SAVE_PATH = 'res/'

class DQNRunner:
    """
    Manages the setup, execution, and training of DQN agents within a specified environment.
    Supports different configurations of agent interactions and training regimes.

    Attributes:
        config (dict): Configuration settings for the runner and agents.
        logger (SimpleLogger): Logger for recording training progress and statistics.
        env_fn (callable): A function to initialize the environment.
        agents (dict): Dictionary of agents participating in the environment.
        learners (set): Set of agents that are learning (i.e., being trained).
        replays (dict): Dictionary of replay buffers for learning agents.
        selected_learner (tuple): Tuple containing the key and the instance of the primary learner.
    """
    def __init__(self, env_fn, config, logger):
        self.config = config
        self.logger = logger
        self.batch_size = config['batch_size']
        self.total_train_t = config["total_train_t"]
        self.n_agents = config["env_args"]["num_players"]
        self.env_fn = env_fn
        self.env = self.env_fn(**self.config['env_args'])  # Initialize environment immediately
        self.build_agents()

    def build_agents(self):
        """
        Initializes agents based on configuration, creating either self-play, competitive, or cooperative setups.
        """
        standard_learner = 'player_0'
        agent_type = self.config.get('agents', 'all_dqn')
        agent_class = {
            'self_play': DQNAgent,
            'vs_random': RandomAgent,
            'vs_always': AlwaysOfferAgent
        }.get(agent_type, DQNAgent)

        if agent_type == 'self_play':
            agent = DQNAgent(self.config)
            self.agents = {a: agent for a in self.env.agents}
        else:
            self.agents = {a: agent_class() for a in self.env.agents if a != standard_learner}
            self.agents[standard_learner] = DQNAgent(self.config)

        self.learners = {standard_learner} if agent_type in ['vs_random', 'vs_always'] else set(self.agents.keys())
        self.replays = {a: TimestepBuffer(self.config['buffer']) for a in self.learners}
        self.selected_learner = standard_learner, self.agents[standard_learner]

    def rollout(self, test=False):
        """
        Executes one complete episode of interaction in the environment.

        Args:
            test (bool): Whether to run the rollout in test mode (i.e., no exploration).

        Returns:
            dict: A history of observations, actions, rewards, and states for each agent.
        """
        obs, _ = self.env.reset()
        history = {agent: {"observations": [], "actions": [], "rewards": [], "next_observations": [], "dones": []}
                   for agent in self.agents}

        while self.env.agents:
            actions = {agent: self.agents[agent].act(state=obs[agent]["observation"],
                                                     action_mask=obs[agent]["action_mask"],
                                                     t=self.train_t,
                                                     test=test)
                       for agent in self.env.agents}

            next_obs, rewards, terminations, truncations, _ = self.env.step(actions)
            for agent in self.agents:
                history[agent]["observations"].append(obs[agent]["observation"])
                history[agent]["actions"].append(actions[agent])
                history[agent]["rewards"].append(rewards[agent])
                history[agent]["next_observations"].append(next_obs[agent]["observation"])
                history[agent]["dones"].append(terminations[agent] or truncations[agent])

            self.train_t += 1
            obs = next_obs

        return history


    def train(self, test=False):

        self.logger.info("Starting Training")

        self.train_t = 0
        self.episode = 0

        episode_returns = {agent : [] for agent in self.agents.keys()}

        while self.train_t <= self.total_train_t:

            # Run for a whole episode at a time
            episode_batch = self.rollout(test=test)
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
            if self.train_t % self.config['sync_freq'] == 0:
                # self.logger.info(f"Policy and Target Network Syncing @ {self.t}")
                for agent in self.learners:
                    self.agents[agent].sync()

            if self.config['save'] and \
                    self.train_t > 0 and \
                    self.train_t % (self.total_train_t // self.config['save_count']) == 0:
                time = f'{datetime.now().strftime("%Y%m%d")}'
                token = f'{self.config["save_token"]}'
                save_path = os.path.join(SAVE_PATH, "models", "DQN", time, str(self.train_t), token)
                self.selected_learner[1].save(save_path)
                self.logger.info(f"Saved DQN Model at t={self.train_t}/{self.total_train_t}")

            if self.train_t > 0 and self.train_t % self.config['stats_freq'] == 0:
                timestep = f"episode = {self.episode}, time = {self.train_t} of {self.total_train_t}"
                self.logger.stat(timestep, episode_returns[self.selected_learner[0]])

        self.logger.info("Finished Training")
        return episode_returns[self.selected_learner[0]]