from .base import BaseAgent
import numpy as np 

class GTOAgent(BaseAgent):

    def __init__(self):
        self.policy = {
               0 : [1, 0],
               1 : [1, 0],
               2 : [1/55, 54/55],
               3 : [0, 1],
               4 : [0, 1]
        }
        

    def act(self, state, action_mask, **kwargs):
        
        t, _, _, _, _, winner = state
        if winner == 1:
            return 0
        else:
            return np.random.choice(np.arange(2), self.policy[t])
