import rospy
import message_filters
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from nav_msgs.msg import Odometry
from std_msgs.msg import Header

class DataCollector:
    def __init__(self, namespace=""):
        # Adjust the namespace for each subscriber
        self.odom_topic = f'{namespace}/odom'
        self.front_camera_topic = f'{namespace}/front_camera/image_raw'
        self.back_camera_topic = f'{namespace}/back_camera/image_raw'
        self.laser1_topic = f'{namespace}/laser1_scan'
        self.laser2_topic = f'{namespace}/laser2_scan'
        self.laser3_topic = f'{namespace}/laser3_scan'

        # Initialize dictionary to store data
        self.reset()

        # Subscribers for collecting data
        self.odom_sub = message_filters.Subscriber(self.odom_topic, Odometry)
        self.front_camera_sub = message_filters.Subscriber(self.front_camera_topic, Image)
        self.back_camera_sub = message_filters.Subscriber(self.back_camera_topic, Image)
        self.laser1_sub = message_filters.Subscriber(self.laser1_topic, LaserScan)
        self.laser2_sub = message_filters.Subscriber(self.laser2_topic, LaserScan)
        self.laser3_sub = message_filters.Subscriber(self.laser3_topic, LaserScan)

        # Synchronize the sensors
        ts = message_filters.ApproximateTimeSynchronizer(
            [self.odom_sub, self.front_camera_sub, self.back_camera_sub, 
             self.laser1_sub, self.laser2_sub, self.laser3_sub], 
            queue_size=10, 
            slop=0.1)
        ts.registerCallback(self.callback)

    def reset(self):
        self.data_dict = {
            'timestamp': None,
            'pose': None,
            'linear_velocity': None,
            'angular_velocity': None,
            'front_camera_image': None,
            'back_camera_image': None,
            'laser1_scan': None,
            'laser2_scan': None,
            'laser3_scan': None
        }


    def callback(self, odom, front_image, back_image, laser1, laser2, laser3):
        # Extract and store the latest timestamp
        self.data_dict['timestamp'] = odom.header.stamp.to_sec()

        # Convert messages and store into the data_dict
        
        # Pose
        self.data_dict['pose'] = np.array([odom.pose.pose.position.x, 
                                           odom.pose.pose.position.y, 
                                           odom.pose.pose.position.z])

        # Linear and Angular Velocities
        self.data_dict['linear_velocity'] = np.array([odom.twist.twist.linear.x, 
                                                      odom.twist.twist.linear.y, 
                                                      odom.twist.twist.linear.z])
        self.data_dict['angular_velocity'] = np.array([odom.twist.twist.angular.x, 
                                                       odom.twist.twist.angular.y, 
                                                       odom.twist.twist.angular.z])

        # Front Camera Image
        front_image_data = np.frombuffer(front_image.data, dtype=np.uint8).reshape(front_image.height, front_image.width, -1)
        self.data_dict['front_camera_image'] = front_image_data

        # Back Camera Image
        back_image_data = np.frombuffer(back_image.data, dtype=np.uint8).reshape(back_image.height, back_image.width, -1)
        self.data_dict['back_camera_image'] = back_image_data

        # Laser Scans
        self.data_dict['laser1_scan'] = np.array(laser1.ranges)
        self.data_dict['laser2_scan'] = np.array(laser2.ranges)
        self.data_dict['laser3_scan'] = np.array(laser3.ranges)

    def start(self):
        rospy.spin()


if __name__ == '__main__':
    rospy.init_node('data_collector')
    data_collector = DataCollector(namespace="")
    data_collector.start()
