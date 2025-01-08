import rldev

import numpy as np

from apep.tools import global_to_local
from apep.pid import PIDController


class DifferentialRobotVelocityController:
    def __init__(self, logger, dt):
        self.logger = logger
        self.dt = dt
        
        self.linear_pid_x = PIDController(kp=1.0, ki=0.0, kd=0.1)   # For linear X
        self.linear_pid_y = PIDController(kp=1.0, ki=0.0, kd=0.1)   # For linear Y
        self.angular_pid = PIDController(kp=4.0, ki=0.0, kd=0.5)    # For angular Z        

        # Create PID controllers for velocity control
        self.linear_pid_x = PIDController(kp=1.0, ki=0.0, kd=0.1)   # For linear X
        self.linear_pid_y = PIDController(kp=1.0, ki=0.0, kd=0.1)   # For linear Y
        self.angular_pid = PIDController(kp=4.0, ki=0.0, kd=0.5)    # For angular Z

        # Maximum velocity constraints
        self.max_linear_velocity = 1.0    # Max linear velocity (m/s)
        self.max_angular_velocity = np.deg2rad(20)  # Max angular velocity (rad/s)


    def run_step(self, current_pose, target_pose):

        local_pose = global_to_local(target_pose, current_pose)

        delta_x, delta_y, delta_theta = local_pose

        desired_linear_x = self.linear_pid_x.update(delta_x, self.dt)
        desired_linear_y = self.linear_pid_y.update(delta_y, self.dt)
        desired_angular_z = self.angular_pid.update(delta_theta, self.dt)

        linear_velocity = np.sqrt(desired_linear_x**2 + desired_linear_y**2)
        if linear_velocity > self.max_linear_velocity:
            scaling_factor = self.max_linear_velocity / linear_velocity
            desired_linear_x *= scaling_factor
            desired_linear_y *= scaling_factor

        if abs(desired_angular_z) > self.max_angular_velocity:
            desired_angular_z = np.sign(desired_angular_z) * self.max_angular_velocity


        # self.logger.loginfo(f"Error (x, y, theta): [{delta_x:.3f}, {delta_y:.3f}, {np.rad2deg(delta_theta):.3f}], Command (vx, vy, wz): [{desired_linear_x:.3f}, {desired_linear_y:.3f}, {desired_angular_z:.3f}]")
        return rldev.Data(linear_x=desired_linear_x, linear_y=desired_linear_y, angular_z=desired_angular_z)


