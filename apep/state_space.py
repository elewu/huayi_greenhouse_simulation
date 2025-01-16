
import numpy as np

from apep.tools import global_to_local, state_global_to_local


class StateSpace:
    def __init__(self):
        self.dim_state = 1

    def pack(self, env):
        return np.array([0], dtype=np.float32)



class StateSpaceGoalReach:
    def __init__(self):
        self.dim_state = 9

    def pack(self, env):
        goal_local = global_to_local(env.goal_pose, env.current_pose)
        state_local = state_global_to_local(env.current_state, env.current_state[..., [0,1,2]])
        state = np.concatenate([state_local, goal_local])
        print('state: ', state)
        return state.astype(np.float32)

