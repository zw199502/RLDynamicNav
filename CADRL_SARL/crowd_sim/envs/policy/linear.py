import numpy as np
from crowd_sim.envs.policy.policy import Policy
from crowd_sim.envs.utils.action import ActionXY


class Linear(Policy):
    def __init__(self):
        super().__init__()
        self.trainable = False
        self.kinematics = 'holonomic'
        self.multiagent_training = True

    def configure(self, config):
        assert True

    def predict(self, state):
        self_state = state.self_state
        theta = np.arctan2(self_state.gy-self_state.py, self_state.gx-self_state.px)
        vx = np.cos(theta) * self_state.v_pref
        vy = np.sin(theta) * self_state.v_pref
        v = np.array([vx, vy])
        v_noise = np.random.normal(0.0, self_state.v_pref / 4.0, 2)
        v_clip = np.clip(v + v_noise, -1.0, 1.0)
        action = ActionXY(v_clip[0], v_clip[1])

        return action
