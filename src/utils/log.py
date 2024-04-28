import numpy as np

class SimpleLogger:
    """
    A simple logger class for logging information to the console.
    """
    def info(self, message):
        """
        Logs an informational message to the console.

        Parameters:
        - message (str): The message to log.
        """
        print(f"INFO: {message}")

    def stat(self, message, returns):
        
        print(f"STAT: {message}")
        print(f'Recent 1000: mean={np.mean(returns[-1000:])}, std={np.std(returns[-1000:])}')
        print(f'Recent 2500: mean={np.mean(returns[-2500:])}, std={np.std(returns[-2500:])}')
        print(f'Recent 5000: mean={np.mean(returns[-5000:])}, std={np.std(returns[-5000:])}')
        print(f'Recent 10000: mean={np.mean(returns[-10000:])}, std={np.std(returns[-10000:])}')

