
import numpy as np

from apep.tools import global_to_local


class MetricFunc:
    def __init__(self):
        pass

    def compute(self, env, state, action):
        return 0.0



class MetricFuncDistanceToGoal(MetricFunc):
    def __init__(self):
        super().__init__()
        self.last_pose = None

    def compute(self, env, state, action):
        local_goal = global_to_local(env.goal_pose, env.current_pose)
        return {'distance_to_goal': np.linalg.norm(local_goal[..., :2])}


