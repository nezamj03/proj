import yaml
from .ppo_runner import PPORunner
from js_env.js_env_v0 import create_js_v0
from ..utils.log import SimpleLogger
from ..utils.parse_config import load_config
from ..utils.plots import plot_returns
import numpy as np
import torch
import os
from datetime import datetime

np.random.seed(56)
torch.manual_seed(56)
SAVE_PATH = 'res/'
log = SimpleLogger()

def train(num_epochs):

    config = load_config('src/ppo/config/default.yaml')
    res = []
    for _ in range(num_epochs):
        runner = PPORunner(create_js_v0, config, log)
        runner.setup()
        returns = runner.train()
        res.append(returns)
    
    time = f'{datetime.now().strftime("%Y%m%d")}'
    token = f'{config["save_token"]}'
    path = os.path.join(SAVE_PATH, "figures", "PPO", time, token)
    os.makedirs(path, exist_ok=True)
    plot_returns(res, path)

if __name__ == '__main__':

    num_epochs = 1
    train(num_epochs)
    

