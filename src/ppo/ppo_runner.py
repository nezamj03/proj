import numpy as np
from .ppo import PPOAgent
from ..utils.buffer import EpisodeBuffer
from ..agents.random import RandomAgent
from ..agents.always_offer import AlwaysOfferAgent
from datetime import datetime 
import os

SAVE_PATH = 'res/'

class PPORunner:
    def __init__(self, env_fn, config, logger):
        self.config = config
        self.logger = logger
        self.batch_size = config['batch_size']
        self.total_train_t = config["total_train_t"]
        self.total_eval_t = config.get("total_eval_t", 50_000)
        self.n_agents = config["n_agents"]
        self.env_fn = env_fn

    def setup(self):

        self.env = self.env_fn(**self.config['env_args'])
        self.env.reset()
        self.build_agents()

    def build_agents(self):
        standard_learner = 'player_0'
        if self.config['agents'] == 'self_play':
            agent = PPOAgent(self.config)
            self.agents = {a : agent for a in self.env.agents}
            self.learners = set(self.agents.keys())
        if self.config['agents'] == 'vs_random':
            self.agents =  {a : RandomAgent() for a in self.env.agents}
            self.agents[standard_learner] = PPOAgent(self.config)
            self.learners = set([standard_learner])
        if self.config['agents'] == 'vs_always':
            self.agents = {a : AlwaysOfferAgent() for a in self.env.agents}
            self.agents[standard_learner] = PPOAgent(self.config)
            self.learners = set([standard_learner])
        else:
            self.agents = {a : PPOAgent(self.config) for a in self.env.agents}
            self.learners = set(self.agents.keys())

        self.replays = {
            a : EpisodeBuffer(self.config['buffer']) for a in self.learners
        }
        self.selected_learner = standard_learner, self.agents[standard_learner]        

    def rollout(self):

        self.env = self.env_fn(**self.config['env_args'])
        obs, infos = self.env.reset()
        history = {
            agent : {
                'observations': [],
                'actions': [],
                'rewards': [],
                'next_observations': [],
                'dones': [],
                'log_probs': [],
                'values': [],
                'action_masks': []
                } for agent in self.learners
            }
        
        while self.env.agents:
            actions = {}
            log_probs = {}
            res = {agent : self.agents[agent].act(
                    state= obs[agent]["observation"], action_mask= obs[agent]["action_mask"]) 
                    for agent in self.env.agents}
            actions, log_probs, values = {}, {}, {}
            
            for agent in self.env.agents:
                if agent in self.learners:
                    actions[agent] = res[agent][0]
                    log_probs[agent] = res[agent][1]
                    values[agent] = res[agent][2]
                else:
                    actions[agent] = res[agent]

            next_obs, rewards, terminations, truncations, infos = self.env.step(actions)

            for agent in infos.keys():
                if agent in self.learners:
                    history[agent]["observations"].append(obs[agent]["observation"])
                    history[agent]["actions"].append(actions[agent])
                    history[agent]["rewards"].append(rewards[agent])
                    history[agent]["next_observations"].append(next_obs[agent]["observation"])
                    history[agent]["dones"].append(terminations[agent] or truncations[agent])
                    history[agent]["log_probs"].append(log_probs[agent])
                    history[agent]["values"].append(values[agent])
                    history[agent]["action_masks"].append(obs[agent]["action_mask"])

            self.train_t += 1
            obs = next_obs
        
        return history
    
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

    def evaluate(self):

        self.logger.info("Starting Evaluation")
        
        self.eval_t = 0
        self.episode = 0

        episode_returns = {agent : [] for agent in self.agents.keys()}

        while self.eval_t <= self.total_eval_t:

            episode_batch = self.rollout()
            self.episode += 1

            for agent in episode_batch:
                episode_returns[agent].append(sum(episode_batch[agent]["rewards"]))

            if self.config['save'] and \
                    self.eval_t > 0 and \
                    self.eval_t % (self.total_eval_t // self.config['count_save']):
                token = f'{datetime.now().strftime("%Y%m%d")}'
                save_path = os.path.join(SAVE_PATH, "models", "PPO", token, str(self.eval_t))
                self.selected_learner[1].save(save_path)
                self.logger.info(f"Saved DQN Model at t={self.eval_t}/{self.total_eval_t}")

            # if self.train_t > 0 and self.train_t % self.config['stats_freq']:
            #     self.logger.log_stat("episode", self.episode, self.train_t)
            #     self.logger.print_recent_stats(episode_returns)

        self.logger.info("Finished Training")
