from .base import BaseAgent
import numpy as np

class GTOAgent(BaseAgent):
    """
    A specific agent that implements a predetermined strategy or policy in a game, assuming
    Game Theoretic Optimal (GTO) play.

    The GTOAgent class inherits from BaseAgent and implements a fixed policy based on predefined
    probabilities for each game state represented by time `t`.

    Attributes:
        policy (dict): A dictionary mapping each game state (`t`) to a list of probabilities
                       for choosing between two actions. The policy provides the optimal strategy
                       for the agent in different situations.
    """

    def __init__(self):
        """
        Initializes the GTOAgent with a predefined policy.
        """
        self.policy = {
            0: [1, 0],
            1: [1, 0],
            2: [1/55, 54/55],
            3: [0, 1],
            4: [0, 1]
        }

    def act(self, state, action_mask=None, **kwargs):
        """
        Decides an action based on the current game state, specifically using the time `t` and
        winner information within the state tuple. Chooses action based on the predefined
        policy probabilities.

        Parameters:
            state (tuple): The current state of the game, expected to be a tuple containing:
                           (time `t`, other state information..., `winner`)
            action_mask (Optional[list]): A mask that can be applied to actions. Default is None.
            **kwargs: Additional keyword arguments.

        Returns:
            int: The index of the chosen action, either 0 or 1.
        """
        t, _, _, _, _, winner = state
        if winner == 1:
            return 0  # Return the first action if the winner is player 1
        else:
            # Randomly choose an action based on the policy probabilities for time `t`
            return np.random.choice(np.arange(2), p=self.policy[t])
