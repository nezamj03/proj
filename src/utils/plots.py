import matplotlib.pyplot as plt
import numpy as np
import os
from typing import Tuple

def plot_returns(returns: np.ndarray, path: str) -> None:
    """
    Plots the cumulative average reward of an array of returns and saves the plot to a specified directory.

    Parameters:
        returns (np.ndarray): A 2D array where each row represents a sequence of rewards for an episode.
        path (str): The directory path where the plot image will be saved.
    """
    plt.figure(figsize=(14, 8))

    cum_winrate, error = _prepare_cum_winrate(returns)
    plt.subplot(212)
    plt.plot(cum_winrate, color='black')
    wr_bounds = (cum_winrate - error, cum_winrate + error)
    plt.fill_between(range(len(cum_winrate)), *wr_bounds, color='gray', alpha=0.2)
    plt.xlabel('Episode Number')
    plt.ylabel('Cumulative Mean Reward')
    plt.title('Cumulative Average Reward')
    plt.savefig(os.path.join(path, 'stats.png'))
    plt.close()

def _prepare_cum_returns(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the cumulative returns, their mean and the standard error of the mean across episodes.

    Parameters:
        returns (np.ndarray): A 2D array of returns where each row corresponds to an episode.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the mean of cumulative returns and the standard error of the mean.
    """
    cumulative_returns = np.cumsum(returns, axis=1)
    mean = np.mean(cumulative_returns, axis=0)
    std = np.std(cumulative_returns, axis=0)
    sem = std / np.sqrt(len(returns))
    return mean, sem

def _prepare_cum_winrate(returns: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Calculates the cumulative win rate over episodes and computes the standard error of the mean.

    Parameters:
        returns (np.ndarray): A 2D array of returns where each row corresponds to an episode.

    Returns:
        Tuple[np.ndarray, np.ndarray]: A tuple containing the mean of cumulative win rates and the standard error of the mean.
    """
    cumulative_returns = np.cumsum(returns, axis=1)
    win_rate = cumulative_returns / np.arange(1, len(returns[0]) + 1)
    mean = np.mean(win_rate, axis=0)
    std = np.std(win_rate, axis=0)
    sem = std / np.sqrt(len(returns))
    return mean, sem
