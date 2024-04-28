import yaml
from .dqn.dqn_runner import DQNRunner
from .ppo.ppo_runner import PPORunner
from js_env.js_env_v0 import create_js_v0
from .utils.log import SimpleLogger
import numpy as np
import os

np.random.seed(42)
SAVE_PATH = 'res/'

log = SimpleLogger()

def load_config(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)
    
def eval(runner, num_epoch):
    
    res = []
    for _ in num_epoch:
        res.append(np.cumsum(runner.eval()))
    res = np.mean(res, axis=0)
    
    





    
def main():

    num_epochs = 10
    log = SimpleLogger()
    config = load_config('src/dqn/config/default.yaml')
    runner = DQNRunner(create_js_v0, config, log)
    runner.setup()
    returns = runner.train()

    # config = load_config('src/ppo/config/default.yaml')
    # runner = PPORunner(create_js_v0, config, log)
    # runner.setup()
    # runner.train()

if __name__ == '__main__':
    main()

