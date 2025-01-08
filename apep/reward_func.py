
import numpy as np

from apep.tools import global_to_local


class RewardFunc:
    def __init__(self):
        pass

    def compute(self, env, state, action):
        return 0.0



class RewardFuncDistance(RewardFunc):
    def compute(self, env, state, action):

        local_target = global_to_local(env.target_pose, env.current_pose)
        reward_reachability = -np.linalg.norm(local_target)

        local_goal = global_to_local(env.goal_pose, env.current_pose)
        reward_goal = -np.linalg.norm(local_goal)

        reward = reward_reachability + reward_goal *0.1
        return reward


