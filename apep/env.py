import rldev

import importlib, os
import subprocess
import time
import random

import numpy as np

import rospy
import roslaunch
from geometry_msgs.msg import Twist, Pose, PoseStamped
from gazebo_msgs.msg import ModelStates
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion, quaternion_from_euler

from apep.tools import local_to_global
from apep.gazebo_state import GazeboState
from apep.vel_controller import DifferentialRobotVelocityController





class EnvGazebo:
    def __init__(self, state_space, action_space, reward_func, max_steps=50):
        self.state_space = state_space
        self.action_space = action_space
        self.reward_func = reward_func

        ### counters
        self.step_reset = -1

        # self.decision_frequency = 1
        # self.control_frequency = 10
        # self.decision_dt = 1 / self.decision_frequency
        # self.control_dt = 1 / self.control_frequency

        self.max_steps = max_steps

        ### ros related
        rospy.init_node('env_gazebo')

        self.launch_file = os.path.join(os.path.dirname(importlib.util.find_spec('apep').origin), '../src/mybot_description/launch/empty_world.launch')
        self.start_gazebo()

        self.gazebo_state = GazeboState()
        self.gazebo_state.unpause()

        # self.controller = DifferentialRobotVelocityController(rospy, self.control_dt)


        # Publisher for twist commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscribers for odometry and goal position
        self.current_state = None
        self.current_pose = None
        self.target_pose = None  # (target_x, target_y, target_yaw)

        self.goal_pose_pub = rospy.Publisher('/goal_pose', PoseStamped, queue_size=1, latch=True)

        rospy.Subscriber('/gazebo/model_states', ModelStates, self.model_state_callback)
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        # rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)
        time.sleep(1)

        self.gazebo_state.pause()
        return


    def reset(self):
        self.step_reset += 1
        self.time_step = 0

        ### restart gazebo
        # if self.process is not None:
        #     self.process.kill()
        #     self.process.wait()  # Wait for the process to finish
        if (self.step_reset + 1) % 10 == 0:
            self.gazebo_launch.shutdown()
            self.start_gazebo()

        ### delete model
        self.gazebo_state.pause()
        self.gazebo_state.delete_model('greenbot')

        ### reset simulation
        self.gazebo_state.reset()

        ### spawn model
        robot_description = rospy.get_param('robot_description')
        model_name = 'greenbot'
        initial_pose = Pose()
        initial_pose.position.z = 0.2
        self.gazebo_state.spawn_model(model_name, robot_description, initial_pose)
        self.cmd_vel_pub.publish(Twist())

        self.gazebo_state.unpause()
        self.gazebo_state.pause()

        ### random set goal
        # radius = random.uniform(10, 50)
        radius = 5
        angle = random.uniform(-np.pi/3, np.pi/3) if random.random() < 0.5 else random.uniform(2*np.pi/3, 4*np.pi/3)
        self.goal_pose = np.asarray([np.cos(angle) * radius, np.sin(angle) * radius, angle])
        self.publish_goal_pose(self.goal_pose)
        
        return self.state_space.pack(self)


    def step(self, action_data: rldev.Data):
        self.time_step += 1

        action = self.action_space.scale(action_data.action)
        self.action_space.execute(self, action)
        # action = self.action_space.scale(action_data.action)
        # self.target_pose = local_to_global(self.current_pose, action)
        # # self.target_pose = self.goal_pose  ## delete

        # for _ in range(int(self.control_frequency / self.decision_frequency)):
        #     control = self.controller.run_step(self.current_pose, self.target_pose)
        #     cmd = Twist()
        #     cmd.linear.x = control.linear_x
        #     cmd.linear.y = control.linear_y
        #     cmd.angular.z = control.angular_z
        #     self.cmd_vel_pub.publish(cmd)

        #     self.gazebo_state.unpause(self.control_dt)
        #     self.gazebo_state.pause()
        
        next_state = self.state_space.pack(self)
        reward = self.reward_func.compute(self, None, action)
        done = self.time_step > self.max_steps
        info = {}
        print('action: ', action)
        print('\n')
        return next_state, reward, done, info



    def start_gazebo(self):
        # Start Gazebo using the launch file
        # self.process = subprocess.Popen(["roslaunch", self.launch_file])
        # self.gazebo_launch.start()
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, None)
        roslaunch.configure_logging(uuid)
        self.gazebo_launch = roslaunch.parent.ROSLaunchParent(uuid, [self.launch_file])
        self.gazebo_launch.start()
        time.sleep(10)



    def model_state_callback(self, msg: ModelStates):
        idx = msg.name.index('greenbot')
        x = msg.pose[idx].position.x
        y = msg.pose[idx].position.y
        quat = (msg.pose[idx].orientation.x, 
                msg.pose[idx].orientation.y, 
                msg.pose[idx].orientation.z, 
                msg.pose[idx].orientation.w)
        _, _, heading = euler_from_quaternion(quat)
        vx = msg.twist[idx].linear.x
        vy = msg.twist[idx].linear.y
        wz = msg.twist[idx].angular.z
        self.current_state = np.array([x, y, heading, vx, vy, wz])


    def odom_callback(self, msg: Odometry):

        current_x = msg.pose.pose.position.x
        current_y = msg.pose.pose.position.y
        quat = (msg.pose.pose.orientation.x, 
                msg.pose.pose.orientation.y, 
                msg.pose.pose.orientation.z, 
                msg.pose.pose.orientation.w)

        _, _, current_theta = euler_from_quaternion(quat)
        self.current_pose = np.array([current_x, current_y, current_theta])


    def goal_callback(self, msg):
        # Extract target position from PoseStamped message
        target_yaw = euler_from_quaternion((msg.pose.orientation.x, 
                                             msg.pose.orientation.y, 
                                             msg.pose.orientation.z, 
                                             msg.pose.orientation.w))[2]  # Get yaw
        self.target_pose = np.asarray([msg.pose.position.x, msg.pose.position.y, target_yaw])



    def publish_goal_pose(self, goal_pose):
        goal_msg = PoseStamped()
        goal_msg.header.stamp = rospy.Time(0)
        goal_msg.header.frame_id = 'world'
        goal_msg.pose.position.x = goal_pose[0]
        goal_msg.pose.position.y = goal_pose[1]
        goal_msg.pose.position.z = 0.0  # Assuming 2D navigation, z is 0

        # Convert yaw to quaternion
        quat = quaternion_from_euler(0, 0, goal_pose[2])
        goal_msg.pose.orientation.x = quat[0]
        goal_msg.pose.orientation.y = quat[1]
        goal_msg.pose.orientation.z = quat[2]
        goal_msg.pose.orientation.w = quat[3]

        self.goal_pose_pub.publish(goal_msg)



class StateIndex:
    x = 0
    y = 1
    heading = 2
    vx = 3
    vy = 4
    wz = 5

    dim = 6

