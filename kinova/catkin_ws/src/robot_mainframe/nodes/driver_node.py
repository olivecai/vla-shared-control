
#!/usr/bin/python3
'''
Sep 16 2026

DRIVES actual kinova movement; listens to /driver

This is the ONLY file in the entire repo that is allowed to touch the kinova joints!
'''
import rospy
from sensor_msgs.msg import JointState
from kortex_bringup import KinovaGen3
from std_msgs.msg import Float32
import numpy as np

from const import *

class DriverNode:
    def __init__(self):
        '''
        listens to /driver and will publish position states (TODO this could change to velocity etc)
        '''
        rospy.init_node('driver', anonymous=True)
    
        try:
            self.kinova = KinovaGen3()
            rospy.loginfo(f"Connected to KinovaGen3: {self.kinova}")
        except:
            self.kinova = None
            rospy.loginfo(f"WARNING: Kinova not found.")

        rospy.Subscriber('/driver/joint_state', JointState, self.callback_joint_state) 
        rospy.loginfo(f"Initialized Driver sub on topic /driver/joint_state")
        rospy.Subscriber('/driver/gripper_state', Float32, self.callback_gripper_state)
        rospy.loginfo(f"Initialized Driver sub on topic /driver/gripper_state")

        rospy.spin()

    def callback_joint_state(self, data):
        rospy.loginfo(f"driver_node::callback_joint_state: Received {data.position} from time {data.header.stamp}. Current time: {rospy.Time.now()}")  
        if self.kinova:
            self.kinova.send_joint_angles(data.position)
        else:
            rospy.loginfo(f"WARNING: Kinova not found.")

    def callback_gripper_state(self, data: Float32):
        rospy.loginfo(f"driver_node::callback_gripper_state: Receieved {data.data}")
        if self.kinova:
            # data.data is a percentage (0=open, 100=closed), matching /gripper_state;
            # send_gripper_command expects a fraction (0.0=open, 1.0=closed).
            self.kinova.send_gripper_command(data.data / 100.0)
        else:
            rospy.loginfo(f"WARNING: Kinova not found.")


def main():
    rospy.sleep(1) 
    rospy.loginfo("Starting a Kinova Driver node ...")
    node = DriverNode()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Driver...")


if __name__ == '__main__':
    main()

