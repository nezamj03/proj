import matplotlib.pyplot as plt
import numpy as np
import os 

def plot_returns(returns, path):
    
    plt.figure(figsize=(14, 8))

    cum_winrate, error = _prepare_cum_winrate(returns)
    plt.subplot(212)
    plt.plot(cum_winrate, color= 'black')
    wr_bounds = (cum_winrate - error, cum_winrate + error)
    plt.fill_between(range(len(cum_winrate)), *wr_bounds, color='gray', alpha=0.2)
    plt.xlabel('episode number')
    plt.ylabel('cumulative mean reward')
    plt.title('cumulative average reward')
    plt.savefig(os.path.join(path, 'stats.png'))
    plt.close()

def _prepare_cum_returns(returns):
    returns = np.cumsum(returns, axis=1)
    mean = np.mean(returns, axis=0)
    std = np.std(returns, axis=0)
    sem = std / np.sqrt(len(returns))
    return mean, sem

def _prepare_cum_winrate(returns):
    returns = np.cumsum(returns, axis=1)
    wr = returns / np.arange(1, len(returns[0]) + 1)
    mean = np.mean(wr, axis=0)
    std = np.std(wr, axis=0)
    sem = std / np.sqrt(len(returns))
    return mean, sem
