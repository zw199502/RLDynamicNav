import numpy as np

class random_policy:
    def __init__(self):
        self.name = 'random_policy'
        self.phase = None

    def set_phase(self, phase):
        self.phase = phase

    def predict(self, state):
        action = np.random.uniform(-1, 1, 2)
        return action