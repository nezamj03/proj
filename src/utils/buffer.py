import numpy as np
from collections import deque

class EpisodeBuffer:
    def __init__(self, config):
        """
        Initializes the EpisodeBuffer with a specified capacity.
        
        Parameters:
        - capacity (int): The maximum number of episodes the buffer can hold.
        """
        self.capacity = config['buffer_size']
        self.buffer = deque(maxlen=self.capacity)  # Use deque for efficient pop from left operation
        self.required_keys = config['required_keys']

    def insert_episode(self, episode_data):
        """
        Inserts an episode into the buffer. Each episode is expected to be a dictionary
        containing keys like 'states', 'actions', 'rewards', 'next_states', and 'dones'.
        
        Parameters:
        - episode_data (dict): The episode data containing sequences for states, actions, rewards, etc.
        """
        if not isinstance(episode_data, dict):
            raise ValueError("Episode data must be a dictionary")
        if not all(key in episode_data for key in self.required_keys):
            raise KeyError(f"Episode data must contain the following keys: {self.required_keys}")
        self.buffer.append(episode_data)

    def sample(self, batch_size):
        """
        Samples a batch of episodes from the buffer. This method randomly selects episodes.
        
        Parameters:
        - batch_size (int): The number of episodes to sample.
        
        Returns:
        - list of dict: A batch of episodes, each formatted as a dictionary of sequences.
        """
        if batch_size > len(self.buffer):
            raise ValueError("Requested batch size is larger than the number of episodes in the buffer")
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]

    def __len__(self):
        """
        Returns the current size of the buffer.
        
        Returns:
        - int: The number of episodes currently in the buffer.
        """
        return len(self.buffer)

class TimestepBuffer:

    def __init__(self, config):
        """
        Initializes the EpisodeBuffer with a specified capacity.
        
        Parameters:
        - capacity (int): The maximum number of episodes the buffer can hold.
        """
        self.capacity = config['buffer_size']
        self.buffer = deque(maxlen=self.capacity)  # Use deque for efficient pop from left operation
        self.required_keys = config['required_keys']

    def insert_episode(self, episode_data):
        """
        Inserts an episode into the buffer. Each episode is expected to be a dictionary
        containing keys like 'states', 'actions', 'rewards', 'next_states', and 'dones'.
        
        Parameters:
        - episode_data (dict): The episode data containing sequences for states, actions, rewards, etc.
        """
        if not isinstance(episode_data, dict):
            raise ValueError("Episode data must be a dictionary")
        if not all(key in episode_data for key in self.required_keys):
            raise KeyError(f"Episode data must contain the following keys: {self.required_keys}")
    
        res = []
        for group in zip(*episode_data.values()):
            self.buffer.append({k: v for k, v in zip(episode_data.keys(), group)}) 
             
    def sample(self, batch_size):
        """
        Samples a batch of episodes from the buffer. This method randomly selects episodes.
        
        Parameters:
        - batch_size (int): The number of episodes to sample.
        
        Returns:
        - list of dict: A batch of episodes, each formatted as a dictionary of sequences.
        """
        if batch_size > len(self.buffer):
            raise ValueError("Requested batch size is larger than the number of episodes in the buffer")
        indices = np.random.choice(len(self.buffer), size=batch_size, replace=False)
        return [self.buffer[idx] for idx in indices]
    

    def __len__(self):
        """
        Returns the current size of the buffer.
        
        Returns:
        - int: The number of episodes currently in the buffer.
        """
        return len(self.buffer)