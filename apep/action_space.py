
import numpy as np


class BoxSpace(object):
    def __init__(self, low, high, shape, dtype):
        self.low = low
        self.high = high
        self.shape = shape
        self.dtype = dtype
        self.max_action = np.ones(self.shape, dtype=self.dtype)
        self.dim_action = self.shape[0]
        self.continuous = True

    def sample(self):
        return np.random.uniform(self.low,self.high, size=self.shape).astype(self.dtype)

    def scale(self, action):
        return action * self.max_action



class ActionSpaceDeltaPose(BoxSpace):
    def __init__(self, max_x=3, max_y=1, max_yaw=np.pi):
        super().__init__(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,), dtype=np.float32)
    
        self.continuous = True
        self.max_action = np.array([max_x, max_y, max_yaw], dtype=np.float32)
        return

