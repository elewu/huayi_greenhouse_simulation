
import numpy as np

import rospy
from geometry_msgs.msg import Twist, Pose, PoseStamped

from apep.tools import local_to_global
from apep.vel_controller import DifferentialRobotVelocityController


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
        return np.clip(action, self.low, self.high) * self.max_action



class ActionSpaceDeltaPose(BoxSpace):
    def __init__(self, max_x=3, max_y=1, max_yaw=np.pi):
        super().__init__(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,), dtype=np.float32)
    
        self.continuous = True
        self.max_action = np.array([max_x, max_y, max_yaw], dtype=np.float32)

        self.decision_frequency = 1
        self.control_frequency = 10
        self.decision_dt = 1 / self.decision_frequency
        self.control_dt = 1 / self.control_frequency
        self.controller = DifferentialRobotVelocityController(rospy, self.control_dt)
        return

    def execute(self, env, action):
        env.target_pose = local_to_global(env.current_pose, action)

        for _ in range(int(self.control_frequency / self.decision_frequency)):
            control = self.controller.run_step(env.current_pose, env.target_pose)
            cmd = Twist()
            cmd.linear.x = control.linear_x
            cmd.linear.y = control.linear_y
            cmd.angular.z = control.angular_z
            env.cmd_vel_pub.publish(cmd)

            env.gazebo_state.unpause(self.control_dt)
            env.gazebo_state.pause()
        return



class ActionSpaceVelocity(BoxSpace):
    def __init__(self, max_vx=5, max_vy=1, max_wz=2):
        super().__init__(low=np.array([-1.0, -1.0, -1.0]), high=np.array([1.0, 1.0, 1.0]), shape=(3,), dtype=np.float32)
    
        self.continuous = True
        self.max_action = np.array([max_vx, max_vy, max_wz], dtype=np.float32)

        self.control_frequency = 10
        self.control_dt = 1 / self.control_frequency
        return

    def execute(self, env, action):
        cmd = Twist()
        cmd.linear.x = action[0]
        cmd.linear.y = action[1]
        cmd.angular.z = action[2]
        env.cmd_vel_pub.publish(cmd)

        env.gazebo_state.unpause(self.control_dt)
        env.gazebo_state.pause()
        return    

