import numpy as np
import torch

STATES = [
    np.array([0, 45, 4, 0, 0]),
    np.array([1, 41, 3, 44, 0]),
    np.array([2, 37, 2, 40, 0]),
    np.array([3, 33, 1, 36, 0]),
]

def get_qvalues(state):
    