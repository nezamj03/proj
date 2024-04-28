from .base import BaseAgent
import numpy as np 

class AlwaysOfferAgent(BaseAgent):

    def act(self, state, action_mask, **kwargs):
        if action_mask[-1] == 1: 
            return 1
        else: return 0