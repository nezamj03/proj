import os
from .dqn import DQNAgent
from ..utils.buffer import TimestepBuffer
from datetime import datetime
from ..agents.random import RandomAgent
from ..agents.always_offer import AlwaysOfferAgent

SAVE_PATH = 'res/'

class DQNRunner:

    def __init__(self, env_fn, config, logger):
        self.config = config
        self.logger = logger
        self.batch_size = config['batch_size']
        self.total_train_t = config["total_train_t"]
        self.n_agents = config["env_args"]["num_players"]
        self.env_fn = env_fn

    def setup(self):

        self.env = self.env_fn(**self.config['env_args'])
        self.env.reset()
        self.build_agents()
        self.train_t = 0

    def build_agents(self):
        standard_learner = 'player_0'
        if self.config['agents'] == 'self_play':
            agent = DQNAgent(self.config)
            self.agents = {a : agent for a in self.env.agents}
            self.learners = set(self.agents.keys())
        elif self.config['agents'] == 'vs_random':
            self.agents =  {a : RandomAgent() for a in self.env.agents}
            self.agents[standard_learner] = DQNAgent(self.config)
            self.learners = set([standard_learner])
        elif self.config['agents'] == 'vs_always':
            self.agents = {a : AlwaysOfferAgent() for a in self.env.agents}
            self.agents[standard_learner] = DQNAgent(self.config)
            self.learners = set([standard_learner])
        else:
            self.agents = {a : DQNAgent(self.config) for a in self.env.agents}
            self.learners = set(self.agents.keys())

        self.replays = {
            a : TimestepBuffer(self.config['buffer']) for a in self.learners
        }

        self.selected_learner = standard_learner, self.agents[standard_learner]

    def rollout(self, test=False):
        self.env = self.env_fn(**self.config['env_args'])
        obs, infos = self.env.reset()
        history = {
            agent : {
                "observations": [],
                "actions": [],
                "rewards": [],
                "next_observations": [],
                "dones": []
                } for agent in self.agents
            }

        while self.env.agents:
            actions = {
                agent: 
                self.agents[agent].act(
                    state= obs[agent]["observation"], action_mask= obs[agent]["action_mask"], t= self.train_t, test=test
                    ) for agent in self.env.agents
                }
            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

            for agent in infos.keys():
                history[agent]["observations"].append(obs[agent]["observation"])
                history[agent]["actions"].append(actions[agent])
                history[agent]["rewards"].append(rewards[agent])
                history[agent]["next_observations"].append(next_obs[agent]["observation"])
                history[agent]["dones"].append(terminations[agent] or truncations[agent])
            
            self.train_t += 1
            obs = next_obs
        
        return history

    def train(self):

        self.logger.info("Starting Training")

        self.train_t = 0
        self.episode = 0

        episode_returns = {agent : [] for agent in self.agents.keys()}

        while self.train_t <= self.total_train_t:

            # Run for a whole episode at a time
            episode_batch = self.rollout(test=False)

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
                print(save_path)
                self.selected_learner[1].save(save_path)
                self.logger.info(f"Saved DQN Model at t={self.train_t}/{self.total_train_t}")

            if self.train_t > 0 and self.train_t % self.config['stats_freq'] == 0:
                timestep = f"episode = {self.episode}, time = {self.train_t} of {self.total_train_t}"
                self.logger.stat(timestep, episode_returns[self.selected_learner[0]])

        self.logger.info("Finished Training")
        return episode_returns[self.selected_learner[0]]