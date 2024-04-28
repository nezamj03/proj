import numpy as np
from collections import deque
from typing import Dict, List, Any

class EpisodeBuffer:
    """
    A buffer for storing and managing episodes in a reinforcement learning environment. Each episode
    is stored as a dictionary of sequences such as states, actions, rewards, etc.

    Attributes:
        capacity (int): The maximum number of episodes the buffer can hold.
        buffer (deque): A deque used for efficient FIFO operations while maintaining the capacity limit.
        required_keys (List[str]): List of keys required in each episode dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the EpisodeBuffer with a specific configuration.

        Parameters:
            config (Dict[str, Any]): Configuration dictionary containing 'buffer_size' and 'required_keys'.
        """
        self.capacity: int = config['buffer_size']
        self.buffer: deque = deque(maxlen=self.capacity)
        self.required_keys: List[str] = config['required_keys']

    def insert_episode(self, episode_data: Dict[str, Any]) -> None:
        """
        Inserts an episode into the buffer after validating it contains the required keys.

        Parameters:
            episode_data (Dict[str, Any]): The episode data containing sequences for states, actions, rewards, etc.
        
        Raises:
            ValueError: If episode_data is not a dictionary.
            KeyError: If episode_data does not contain all the required keys.
        """
        if not isinstance(episode_data, dict):
            raise ValueError("Episode data must be a dictionary")
        if not all(key in episode_data for key in self.required_keys):
            raise KeyError(f"Episode data must contain the following keys: {self.required_keys}")
        self.buffer.append(episode_data)

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Samples a batch of episodes from the buffer.

        Parameters:
            batch_size (int): The number of episodes to sample.
        
        Returns:
            List[Dict[str, Any]]: A batch of episodes, each formatted as a dictionary.
        
        Raises:
            ValueError: If the requested batch size is larger than the buffer.
        """
        if batch_size > len(self.buffer):
            raise ValueError("Requested batch size is larger than the number of episodes in the buffer")
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.

        Returns:
            int: The number of episodes currently in the buffer.
        """
        return len(self.buffer)

class TimestepBuffer:
    """
    A buffer similar to EpisodeBuffer but designed to store individual timesteps instead of whole episodes.
    This is useful for fine-grained sampling or when dealing with environments where actions are decided per timestep.

    Attributes:
        capacity (int): Maximum number of timesteps the buffer can hold.
        buffer (deque): A deque for efficient FIFO operations while maintaining the capacity limit.
        required_keys (List[str]): List of keys required in each timestep dictionary.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the TimestepBuffer with a specific configuration.

        Parameters:
            config (Dict[str, Any]): Configuration dictionary containing 'buffer_size' and 'required_keys'.
        """
        self.capacity: int = config['buffer_size']
        self.buffer: deque = deque(maxlen=self.capacity)
        self.required_keys: List[str] = config['required_keys']

    def insert_episode(self, episode_data: Dict[str, Any]) -> None:
        """
        Converts an episode into individual timesteps and inserts them into the buffer.

        Parameters:
            episode_data (Dict[str, Any]): The episode data containing sequences for states, actions, rewards, etc.
        
        Raises:
            ValueError: If episode_data is not a dictionary.
            KeyError: If episode_data does not contain all the required keys.
        """
        if not isinstance(episode_data, dict):
            raise ValueError("Episode data must be a dictionary")
        if not all(key in episode_data for key in self.required_keys):
            raise KeyError(f"Episode data must contain the following keys: {self.required_keys}")
        
        for group in zip(*episode_data.values()):
            self.buffer.append({k: v for k, v in zip(episode_data.keys(), group)})

    def sample(self, batch_size: int) -> List[Dict[str, Any]]:
        """
        Samples a batch of timesteps from the buffer.

        Parameters:
            batch_size (int): The number of timesteps to sample.
        
        Returns:
            List[Dict[str, Any]]: A batch of timesteps, each formatted as a dictionary.
        
        Raises:
            ValueError: If the requested batch size is larger than the buffer.
        """
        if batch_size > len(self.buffer):
            raise ValueError("Requested batch size is larger than the number of timesteps in the buffer")
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self) -> int:
        """
        Returns the current size of the buffer.

        Returns:
            int: The number of timesteps currently in the buffer.
        """
        return len(self.buffer)
