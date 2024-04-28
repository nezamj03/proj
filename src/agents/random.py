from .base import BaseAgent
import numpy as np

class RandomAgent(BaseAgent):
    """
    An agent that selects actions randomly from those available as per the action mask.

    The RandomAgent class inherits from BaseAgent and implements a policy of choosing actions at random,
    but only within the constraints provided by the action mask. This ensures that the agent's actions
    are always valid within the current state of the environment.

    This class is suitable for environments where the set of possible actions can vary dynamically
    based on the state of the system.
    """

    def act(self, state, action_mask, **kwargs):
        """
        Decides an action randomly from the list of allowed actions indicated by the action mask.

        Parameters:
            state (tuple): The current state of the environment. This parameter is not directly utilized
                           in this method but is included to match the expected method signature.
            action_mask (list[int]): A list indicating which actions are allowed (1) and which are not (0).
                                     Each index with a value of 1 represents an allowable action.
            **kwargs: Additional keyword arguments.

        Returns:
            int: The index of the chosen action, selected randomly from the indices of allowed actions.
        """
        # Use np.flatnonzero to find indices of non-zero elements in the action_mask, which are allowable actions
        allowed_actions = np.flatnonzero(action_mask)
        return np.random.choice(allowed_actions)  # Randomly select from the allowable actions
