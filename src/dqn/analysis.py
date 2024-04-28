import numpy as np
import torch
from ..networks.mlp import MLP
from ..utils.theoretical import VsAlways
from ..utils.parse_config import load_config

config = load_config('config/default.yaml')
num_players = config['env_args']['num_players']
state_size = config['state_size']
action_size = config['action_size']

def generate(t):
    if t == 0:
        return [0, num_players, 0, 0, 0]
    res = [0, 0, 0, 0, 0]
    res[0] = t
    res[1] = max(num_players - (t * 4), 2)
    res[2] = max(1, max(num_players - ((t - 1) * 4), 2) - 1)
    res[3] = 0
    res[4] = 1
    return res

STATES = [
    generate(0),
    generate(1),
    generate(2),
    generate(3),
    generate(4)
]

model = MLP(state_size, action_size, config)
path = f"/Users/nezamjazayeri/Documents/neu/cs5180/proj/res/models/DQN/20240428/{config['total_train_t']}/dummy/q_network.pt"
model.load_state_dict(torch.load(path))

def get_qvalues(state):
    with torch.no_grad():
        state = torch.FloatTensor(state).unsqueeze(0)  # Convert state to tensor and add batch dimension
        return model(state)
    

