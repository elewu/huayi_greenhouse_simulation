
import time

import rospy
from std_srvs.srv import Empty
from gazebo_msgs.srv import SpawnModel, DeleteModel


class GazeboState:
    def __init__(self):
        self.ns = rospy.get_namespace()
        self.reset_proxy = rospy.ServiceProxy(f'{self.ns}gazebo/reset_simulation', Empty)
        self.unpause_proxy = rospy.ServiceProxy(f'{self.ns}gazebo/unpause_physics', Empty)
        self.pause_proxy = rospy.ServiceProxy(f'{self.ns}gazebo/pause_physics', Empty)
    
        self.delete_model_proxy = rospy.ServiceProxy(f'{self.ns}gazebo/delete_model', DeleteModel)
        self.spawn_model_proxy = rospy.ServiceProxy(f'{self.ns}gazebo/spawn_urdf_model', SpawnModel)
        return
    
    def pause(self, sleep=0.1):
        rospy.wait_for_service(f'{self.ns}gazebo/pause_physics')
        try:
            self.pause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/pause_physics service call failed")
        # rospy.loginfo(f'gazebo pause')
        time.sleep(sleep)
        return
    
    def unpause(self, sleep=0.1):
        rospy.wait_for_service('/gazebo/unpause_physics')
        try:
            self.unpause_proxy()
        except (rospy.ServiceException) as e:
            print("gazebo/unpause_physics service call failed")
        # rospy.loginfo(f'gazebo unpause')
        time.sleep(sleep)
        return



    def reset(self, sleep=0.1):
        self.reset_proxy()
        time.sleep(sleep)
        return




    def delete_model(self, model_name):
        rospy.wait_for_service(f'{self.ns}gazebo/delete_model')
        return self.delete_model_proxy(model_name)
    

    def spawn_model(self, model_name, model_xml, pose, reference_frame='world'):
        rospy.wait_for_service(f'{self.ns}gazebo/spawn_urdf_model')
        response = self.spawn_model_proxy(model_name, model_xml, self.ns, pose, reference_frame)
        time.sleep(0.5)
        if response.success:
            rospy.loginfo("Model spawned successfully")
        else:
            rospy.logerr("Failed to spawn model: %s", response.status_message)
        return


