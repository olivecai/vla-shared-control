
#!/usr/bin/python3
'''
Sep 16 2026

DRIVES actual kinova movement; listens to /driver

This is the ONLY file in the entire repo that is allowed to touch the kinova joints!
'''
import rospy
from sensor_msgs.msg import JointState
from kortex_bringup import KinovaGen3
from std_msgs.msg import Float32, Empty
from geometry_msgs.msg import Twist
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
        rospy.Subscriber('/driver/cartesian_velocity', Twist, self.callback_cartesian_velocity)
        rospy.loginfo(f"Initialized Driver sub on topic /driver/cartesian_velocity")
        rospy.Subscriber('/driver/home_trigger', Empty, self.callback_home_trigger)
        rospy.loginfo(f"Initialized Driver sub on topic /driver/home_trigger")

        rospy.spin()

    def callback_joint_state(self, data):
        rospy.loginfo(f"driver_subscriber::callback_joint_state: Received {data.position} from time {data.header.stamp}. Current time: {rospy.Time.now()}")
        if self.kinova:
            positions = [np.clip(p, *JOINT_LIMIT[i]) for i, p in enumerate(data.position)]
            self.kinova.send_joint_angles(positions)
        else:
            rospy.loginfo(f"WARNING: Kinova not found.")

    def callback_gripper_state(self, data: Float32):
        rospy.loginfo(f"driver_subscriber::callback_gripper_state: Receieved {data.data}")
        if self.kinova:
            # data.data is a percentage (0=open, 100=closed), matching /gripper_state;
            # send_gripper_command expects a fraction (0.0=open, 1.0=closed).
            fraction = np.clip(data.data / 100.0, *JOINT_LIMIT[7])
            self.kinova.send_gripper_command(fraction)
        else:
            rospy.loginfo(f"WARNING: Kinova not found.")

    def callback_cartesian_velocity(self, data: Twist):
        rospy.loginfo(f"driver_subscriber::callback_cartesian_velocity: Received {data}")
        if self.kinova:
            self.kinova.send_cartesian_velocity([
                data.linear.x, data.linear.y, data.linear.z,
                data.angular.x, data.angular.y, data.angular.z,
            ])
        else:
            rospy.loginfo(f"WARNING: Kinova not found.")

    def callback_home_trigger(self, data: Empty):
        rospy.loginfo(f"driver_subscriber::callback_home_trigger: Received home trigger")
        if self.kinova:
            self.kinova.go_home()
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

