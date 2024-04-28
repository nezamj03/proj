import numpy as np

class VsAlways:
    def __init__(self):
        self.num_players = 45
        self.num_rounds = 5
        
        # Initialize the Q-values dictionary
        self.qvalues = {}

        # Start by defining the Q-values for the last round
        self.qvalues[self.generate(self.num_rounds - 1)] = [0, 1/35]

        # Compute Q-values backward from the second last round to the first
        for t in range(self.num_rounds - 2, -1, -1):
            current_state = self.generate(t)
            next_state = self.generate(t + 1)
            remaining = current_state[1]

            # Calculate the Q-value based on the next state's values
            no_offer_value = max(self.qvalues[next_state])
            offer_value = (4 / (remaining - 1)) * (1/5) ** (self.num_rounds - t) + (1 - (4 / (remaining - 1))) * no_offer_value
            self.qvalues[current_state] = [no_offer_value, offer_value]

    def generate(self, t):
        """Generates the state based on the round number t."""
        remaining = max(self.num_players - (t * 4), 2)
        offers = max(1, max(self.num_players - ((t - 1) * 4), 2) - 1)
        return (t, remaining, offers, 0, 1)

    def get(self, **kwargs):
        """Returns the Q-values for a given state or time step."""
        if 'state' in kwargs:
            state = tuple(kwargs['state'])
            return self.qvalues.get(state, "State not found")
        elif 't' in kwargs:
            state = self.generate(kwargs['t'])
            return self.qvalues.get(state, "State not found")

if __name__ == "__main__":
    obj = VsAlways()
    for t in range(5):
        print(f"Q-values at t={t}: {obj.get(t=t)}")
