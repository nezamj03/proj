from .base import BaseAgent
import numpy as np 

class RandomAgent(BaseAgent):

    def act(self, state, action_mask, **kwargs):
        return np.random.choice(np.flatnonzero(action_mask))