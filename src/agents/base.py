from abc import ABC, abstractmethod

class BaseAgent(ABC):
    """
    Abstract base class for a reinforcement learning agent.

    This class ensures that all subclasses implement the `act` method and provide a `learner` property.
    """

    @abstractmethod
    def act(self, state, action_mask, **kwargs):
        """
        Decide an action based on the given state.

        Parameters:
        - state: The current state of the environment.

        Returns:
        - The action to be taken.
        """
        pass