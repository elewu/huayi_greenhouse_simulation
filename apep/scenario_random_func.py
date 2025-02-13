
import random
import numpy as np

from apep.tools import local_to_global

class ScenarioRandomFunc(object):
    def __init__(self):
        pass

    def reset(self):
        pass



class ScenarioRandomFuncGreenhouse(ScenarioRandomFunc):
    def __init__(self, min_x=-8, min_y=-1.8, max_x=8, max_y=1.8):
        self.min_x = min_x
        self.min_y = min_y
        self.max_x = max_x
        self.max_y = max_y


    def reset(self):
        x = np.random.uniform(self.min_x, self.max_x)
        y = np.random.uniform(self.min_y, self.max_y)
        heading = np.random.uniform(-np.pi, np.pi)
        self.current_pose = np.asarray([x, y, heading])

        radius = 15
        angle = random.uniform(-np.pi/3, np.pi/3) if random.random() < 0.5 else random.uniform(2*np.pi/3, 4*np.pi/3)
        goal_pose = np.asarray([np.cos(angle) * radius, np.sin(angle) * radius, angle])
        self.goal_pose = local_to_global(goal_pose, self.current_pose)
        return
