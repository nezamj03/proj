import numpy as np

class VsAlways:

    def __init__(self):
        self.num_players = 45
        self.num_rounds = 5
        
        self.qvalues = {}
        self.qvalues[self.generate(self.num_rounds - 1)] = [0, 1/35]
        for t in np.arange(3, -1, -1):
            t, remaining, offers, winner, has_winner = self.generate(t)
            print(remaining)
            self.qvalues[self.generate(t)] = [
                max(self.qvalues[self.generate(t + 1)]), 
                (4 / (remaining - 1)) * (1/5) ** (self.num_rounds - t) + (1 - (4 / (remaining - 1))) * max(self.qvalues[self.generate(t + 1)])
            ]

    def generate(self, t):
        if t == 0:
            return (0, self.num_players, 0, 0, 0)
        res = [0, 0, 0, 0, 0]
        res[0] = t
        res[1] = max(self.num_players - (t * 4), 2)
        res[2] = max(1, max(self.num_players - ((t - 1) * 4), 2) - 1)
        res[3] = 0
        res[4] = 1
        return tuple(res)
    
    def get(self, **kwargs):
        if 'state' in kwargs:
            return self.qvalues[tuple(kwargs['state'])]
        elif 't' in kwargs:
            return self.qvalues[self.generate(kwargs['t'])]


if __name__ == "__main__":
    obj = VsAlways()
    print(obj.get(t=0))
    print(obj.get(t=1))
    print(obj.get(t=2))
    print(obj.get(t=3))
    print(obj.get(t=4))