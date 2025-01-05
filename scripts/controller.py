import rospy
import numpy as np  # Import numpy to use deg2rad
from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg import Odometry
from tf.transformations import euler_from_quaternion

from apep.vel_controller import DifferentialRobotVelocityController



class DifferentialRobotController:
    def __init__(self):
        rospy.init_node('differential_robot_controller')

        # Publisher for twist commands
        self.cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)

        # Subscribers for odometry and goal position
        rospy.Subscriber('/odom', Odometry, self.odom_callback)
        rospy.Subscriber('/move_base_simple/goal', PoseStamped, self.goal_callback)

        self.current_pose = None
        self.target_position = None  # (target_x, target_y, target_yaw)

        freq = 10
        self.controller = DifferentialRobotVelocityController(rospy, 1 / freq)

        # self.last_time = rospy.Time.now()
        self.loop_rate = rospy.Rate(freq)  # 10 Hz

    def odom_callback(self, msg):
        self.current_pose = msg.pose.pose

    def goal_callback(self, msg):
        # Extract target position from PoseStamped message
        target_yaw = euler_from_quaternion((msg.pose.orientation.x, 
                                             msg.pose.orientation.y, 
                                             msg.pose.orientation.z, 
                                             msg.pose.orientation.w))[2]  # Get yaw
        self.target_position = (msg.pose.position.x, msg.pose.position.y, target_yaw)


    def run_step(self):
        if self.current_pose is None or self.target_position is None:
            return
        
        # current_time = rospy.Time.now()
        # dt = (current_time - self.last_time).to_sec()
        # self.last_time = current_time

        # Get current position and orientation
        current_x = self.current_pose.position.x
        current_y = self.current_pose.position.y
        quat = (self.current_pose.orientation.x, 
                self.current_pose.orientation.y, 
                self.current_pose.orientation.z, 
                self.current_pose.orientation.w)

        _, _, current_theta = euler_from_quaternion(quat)

        # Unpack the target position and yaw
        target_x, target_y, target_yaw = self.target_position

        # Prepare poses for local transformation
        target_pose = np.array([target_x, target_y, target_yaw])  # Target poses
        current_pose = np.array([current_x, current_y, current_theta])

        control = self.controller.run_step(current_pose, target_pose)

        # Assign desired velocities directly to the command
        cmd = Twist()
        cmd.linear.x = control.linear_x
        cmd.linear.y = control.linear_y
        cmd.angular.z = control.angular_z

        self.cmd_vel_pub.publish(cmd)
        return

    def run(self):
        while not rospy.is_shutdown():
            self.run_step()
            self.loop_rate.sleep()

if __name__ == '__main__':
    controller = DifferentialRobotController()
    controller.run()

