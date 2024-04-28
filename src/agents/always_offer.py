from .base import BaseAgent
import numpy as np

class AlwaysOfferAgent(BaseAgent):
    """
    An agent that follows a simple, deterministic policy in a decision-making environment.

    The AlwaysOfferAgent class inherits from BaseAgent and implements a straightforward strategy:
    it always attempts to choose a specific action (offer action) when that action is permissible
    according to the action mask.

    The primary goal of this agent is to always opt for the last action (e.g., offering) if it is
    allowed by the current state's action mask.
    """

    def act(self, state, action_mask, **kwargs):
        """
        Decides an action based on the provided action mask. The agent will always choose the
        last action in the list (usually an offer) if it is allowed; otherwise, it selects the
        first action.

        Parameters:
            state (tuple): The current state of the environment. This parameter is not utilized
                           directly in this method but is included to match the expected method
                           signature.
            action_mask (list[int]): A list indicating which actions are allowed (1) and which
                                     are not (0).
            **kwargs: Additional keyword arguments.

        Returns:
            int: The chosen action index. It returns 1 if the last action is allowed, otherwise 0.
        """
        if action_mask[-1] == 1:
            return 1  # Choose the last action if it's permissible
        else:
            return 0  # Otherwise, default to the first action
