from abc import ABC, abstractmethod
import numpy as np

class Schedule(ABC):
    """
    An abstract base class for implementing schedules for parameters like epsilon in reinforcement learning.
    
    Attributes:
        start_val (float): Initial value of the parameter at the start of the schedule.
        end_val (float): Final value of the parameter at the end of the schedule.
        anneal_time (int): Number of timesteps over which the parameter value will be annealed.
    """

    def __init__(self, schedule_start_value: float, schedule_end_value: float, schedule_anneal_time: int):
        """
        Initializes the schedule with the starting value, ending value, and the total anneal time.
        
        Parameters:
            schedule_start_value (float): The starting value of the schedule.
            schedule_end_value (float): The ending value of the schedule.
            schedule_anneal_time (int): The total time over which the schedule is annealed.
        """
        self.start_val = schedule_start_value
        self.end_val = schedule_end_value
        self.anneal_time = schedule_anneal_time

    @abstractmethod
    def get(self, timestep: int) -> float:
        """
        Abstract method to calculate the parameter value at a given timestep.

        Parameters:
            timestep (int): The current timestep in the simulation or training.

        Returns:
            float: The current value of the parameter.
        """
        pass

    def __call__(self, timestep: int) -> float:
        """
        Allows the schedule instance to be called directly to get the parameter value.

        Parameters:
            timestep (int): The current timestep.

        Returns:
            float: The value of the parameter at the given timestep.
        """
        return self.get(timestep)

class ExponentialSchedule(Schedule):
    """
    A schedule that decreases the parameter value exponentially from the start value to the end value over the specified anneal time.
    """

    def __init__(self, schedule_start_value: float, schedule_end_value: float, schedule_anneal_time: int):
        super().__init__(schedule_start_value, schedule_end_value, schedule_anneal_time)
        self.decay_rate = -np.log(self.end_val / self.start_val) / self.anneal_time

    def get(self, timestep: int) -> float:
        """
        Returns the parameter value at the given timestep, calculated exponentially.

        Parameters:
            timestep (int): The current timestep.

        Returns:
            float: The exponentially decreased parameter value.
        """
        if timestep >= self.anneal_time:
            return self.end_val
        return self.start_val * np.exp(-self.decay_rate * timestep)

class LinearSchedule(Schedule):
    """
    A schedule that decreases the parameter value linearly from the start value to the end value over the specified anneal time.
    """

    def __init__(self, schedule_start_value: float, schedule_end_value: float, schedule_anneal_time: int):
        super().__init__(schedule_start_value, schedule_end_value, schedule_anneal_time)
        self.epsilon_drop = (self.start_val - self.end_val) / self.anneal_time

    def get(self, timestep: int) -> float:
        """
        Returns the parameter value at the given timestep, calculated linearly.

        Parameters:
            timestep (int): The current timestep.

        Returns:
            float: The linearly decreased parameter value.
        """
        if timestep >= self.anneal_time:
            return self.end_val
        return np.maximum(self.end_val, self.start_val - self.epsilon_drop * timestep)
    
REGISTRY = {
    'linear': LinearSchedule,
    'exponential': ExponentialSchedule
}

__all__ = [REGISTRY]
