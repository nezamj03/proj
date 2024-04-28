from abc import ABC, abstractmethod
import numpy as np

class Schedule(ABC):
    def __init__(self, schedule_start_value, schedule_end_value, schedule_anneal_time):

        self.start_val = schedule_start_value
        self.end_val = schedule_end_value
        self.anneal_time = schedule_anneal_time

    @abstractmethod
    def get(self, timestep):
        """
        Calculate the epsilon value for a given timestep.

        Parameters:
        - timestep (int): The current timestep in the simulation or training.

        Returns:
        - float: The current value of epsilon.
        """
        pass

    def __call__(self, timestep):
        return self.get(timestep)

class ExponentialSchedule(Schedule):

    def __init__(self, schedule_start_value, schedule_end_value, schedule_anneal_time):
        super().__init__(schedule_start_value, schedule_end_value, schedule_anneal_time)
        self.decay_rate = -np.log(self.end_val / self.start_val) / self.anneal_time

    def get(self, timestep):
        if timestep >= self.anneal_time:
            return self.end_val
        return self.start_val * np.exp(-self.decay_rate * timestep)


class LinearSchedule(Schedule):

    def __init__(self, schedule_start_value, schedule_end_value, schedule_anneal_time):
        super().__init__(schedule_start_value, schedule_end_value, schedule_anneal_time)
        self.epsilon_drop = (self.start_val - self.end_val) / self.anneal_time

    def get(self, timestep):
        if timestep >= self.anneal_time:
            return self.end_val
        return np.maximum(self.end_val, self.start_val - self.epsilon_drop * timestep)
    
REGISTRY = {
    'linear': LinearSchedule,
    'exponential': ExponentialSchedule
}

__all__ = [REGISTRY]