#!/usr/bin/python3
'''
Sep 16 2026

READS kinova joint state 
'''
from const import *
import rospy
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32
from kortex_bringup import KinovaGen3
from kortex_driver.msg import BaseCyclic_Feedback
import numpy as np


class RobotStateNode:
    def __init__(self):
        '''
        inits the robot state publisher node and will auto start publishing the kinova joint states and the gripper on seperate  topics
        '''
        rospy.init_node('robot_state_publisher', anonymous=True)
        self.gripper_position = 0.0  # 0 (open) .. 100 (closed); updated by _gripper_cb
        try:
            kinova = KinovaGen3(read_joint_state=True)
            rospy.loginfo(f"Connected to KinovaGen3: {kinova}")
        except:
            kinova = None
            rospy.loginfo(f"WARNING: Kinova not found.")

        if kinova:
            # Gripper feedback isn't exposed by KinovaGen3 (only joint/cartesian state are), so
            # subscribe to the raw feedback topic directly instead of touching the vendored driver.
            rospy.Subscriber(f"/{kinova.robot_name}/base_feedback", BaseCyclic_Feedback, self._gripper_cb)

        p = rospy.Publisher('/joint_states', JointState, queue_size=10) # what shld queue size be? for training potentially higher...?
        rospy.loginfo(f"Initialized JointState on topic /joint_states")
        g = rospy.Publisher('/gripper_state', Float32, queue_size=5)
        rospy.loginfo(f"Initialized GripperState on topic /gripper_state")
        rate = rospy.Rate(10) #10hz

        j = JointState()
        while not rospy.is_shutdown():
            j.header.stamp = rospy.Time.now()
            j.name = [f"joint{i}" for i in range(7)]
            if kinova:
                j.position = kinova.position
                gripperfloat = self.gripper_position
            else:
                j.position = [0]*KINOVA_DOF
                gripperfloat = 0.0
            p.publish(j)
            g.publish(Float32(data=gripperfloat))

            rate.sleep()

    def _gripper_cb(self, msg):
        '''
        
        BaseCyclic_Feedback carries gripper motor feedback nested under
        interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position (0=open, 100=closed).
        The list is empty if no gripper is attached.
        '''
        feedback = msg.interconnect.oneof_tool_feedback.gripper_feedback
        if feedback:
            self.gripper_position = feedback[0].motor[0].position


def main():
    rospy.sleep(1) 
    rospy.loginfo("Starting a Kinova RobotState node ...")
    node = RobotStateNode()
    
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down JointState...")


if __name__ == '__main__':
    main()

