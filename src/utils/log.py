import numpy as np

class SimpleLogger:
    """
    A simple logger class for logging information to the console, specifically designed to handle both
    informational messages and statistical summaries of numerical data.
    """

    def info(self, message: str) -> None:
        """
        Logs an informational message to the console.

        Parameters:
            message (str): The message to log.
        """
        print(f"INFO: {message}")

    def stat(self, message: str, returns: np.ndarray) -> None:
        """
        Logs statistical information about recent returns to the console. Statistics include mean
        and standard deviation for the last 1000, 2500, 5000, and 10000 returns, if available.

        Parameters:
            message (str): A message to precede the statistical data.
            returns (np.ndarray): An array of numerical returns data.

        Notes:
            - Only computes statistics for the data lengths that are available.
        """
        print(f"STAT: {message}")

        if len(returns) > 0:
            self._log_stats(returns, 1000)
            self._log_stats(returns, 2500)
            self._log_stats(returns, 5000)
            self._log_stats(returns, 10000)

    def _log_stats(self, returns: np.ndarray, num_entries: int) -> None:
        """
        Helper method to log mean and standard deviation for a specific number of most recent entries.

        Parameters:
            returns (np.ndarray): The array of return values.
            num_entries (int): The number of entries from the end to consider for statistics.
        """
        if len(returns) >= num_entries:
            recent_returns = returns[-num_entries:]
            mean = np.mean(recent_returns)
            std = np.std(recent_returns)
            print(f'Recent {num_entries}: mean={mean:.2f}, std={std:.2f}')
        else:
            print(f'Not enough data to compute stats for the last {num_entries} entries.')

