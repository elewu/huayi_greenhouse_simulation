
import numpy as np

from apep.tools import global_to_local


class RewardFunc:
    def __init__(self):
        pass

    def compute(self, env, state, action):
        return 0.0



class RewardFuncDistance(RewardFunc):
    def __init__(self):
        super().__init__()
        self.last_pose = None

    def compute(self, env, state, action):
        ### pre
        if env.time_step == 1:
            self.last_pose = env.current_pose

        # local_target = global_to_local(env.target_pose, env.current_pose)
        # reward_reachability = -np.linalg.norm(local_target)

        progress = np.linalg.norm((env.goal_pose - self.last_pose)[..., [0,1]]) - np.linalg.norm((env.goal_pose - env.current_pose)[..., [0,1]])
        reward_progress = progress * 50

        local_goal = global_to_local(env.goal_pose, env.current_pose)
        reward_goal = int(np.linalg.norm(local_goal[..., :2]) < 1) * 1

        # reward = reward_reachability + reward_goal *0.1
        reward = reward_progress + reward_goal
        # print('reward progress: ', reward_progress, ' reward goal: ', reward_goal)

        ### post
        self.last_pose = env.current_pose
        return reward


