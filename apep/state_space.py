
import numpy as np

from apep.tools import global_to_local


class StateSpace:
    def __init__(self):
        self.dim_state = 1

    def pack(self, env):
        return np.array([0], dtype=np.float32)



class StateSpaceGoalReach:
    def __init__(self):
        self.dim_state = 3

    def pack(self, env):
        state = global_to_local(env.goal_pose, env.current_pose)
        return state.astype(np.float32)

