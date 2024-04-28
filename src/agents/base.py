from abc import ABC, abstractmethod
from typing import Any, Optional

class BaseAgent(ABC):
    """
    Abstract base class for a reinforcement learning agent.

    This class ensures that all subclasses implement the `act` method, which is responsible for
    determining the next action the agent will take based on the current state of the environment
    and an action mask that may restrict possible actions.

    Attributes:
        learner: A property that must be provided by subclasses, typically representing the learning
                 component of the agent that updates its policy based on received experiences.

    Methods:
        act(state, action_mask, **kwargs): Abstract method that must be implemented by all subclasses.
                                           It defines how the agent chooses an action based on the
                                           current state and a provided action mask.
    """

    @abstractmethod
    def act(self, state: Any, action_mask: Optional[Any] = None, **kwargs) -> Any:
        """
        Decide an action based on the given state and an optional action mask.

        Parameters:
            state (Any): The current state of the environment.
            action_mask (Optional[Any]): An optional mask applied to actions, allowing for the filtering
                                         of actions based on current state conditions. Defaults to None.
            **kwargs: Additional keyword arguments that might be needed for decision-making.

        Returns:
            Any: The action to be taken by the agent.
        """
        pass
