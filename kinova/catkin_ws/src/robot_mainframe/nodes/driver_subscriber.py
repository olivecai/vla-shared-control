
#!/usr/bin/python3
'''
Sep 16 2026

DRIVES actual kinova movement; listens to /driver

This is the ONLY file in the entire repo that is allowed to touch the kinova joints!
'''
import os
import sys
import rospy
from sensor_msgs.msg import JointState
from kortex_bringup import KinovaGen3
from std_msgs.msg import Float32, Empty, Bool
from geometry_msgs.msg import Twist
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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

        # Lets driver_publisher.py know when an action-based command (currently just homing) is
        # in progress, so it can pause streaming cartesian_velocity -- an active velocity stream
        # conflicts with an in-progress waypoint trajectory action (see callback_home_trigger).
        self.busy_pub = rospy.Publisher('/driver/busy', Bool, queue_size=1, latch=True)
        self.busy_pub.publish(Bool(False))

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
            self.busy_pub.publish(Bool(True))
            try:
                # driver_publisher.py streams cartesian_velocity on every /joy tick (even at rest),
                # so the arm is continuously in velocity-streaming mode. send_joint_angles uses an
                # action-based waypoint trajectory, which conflicts with an active velocity stream --
                # this is what causes the "uninitialized ServerGoalHandle" error and erratic motion.
                # Explicitly zero the velocity stream and give the driver a moment to switch modes
                # before starting the trajectory (kortex_bringup's own kinova_gen3.py flags this same
                # requirement, though its disabled workaround used the buggier service-based call).
                # /driver/busy (published True above, False in the finally block below) additionally
                # tells driver_publisher.py to pause its own velocity streaming for the whole
                # trajectory, not just this initial zeroing -- otherwise the very next /joy tick
                # re-starts the conflict a moment later, mid-trajectory.
                self.kinova.send_cartesian_velocity([0, 0, 0, 0, 0, 0])
                rospy.sleep(0.1)

                # As of Aug 19 2026, verified this is a good home position -- but the raw values
                # here are unsigned 0-360 readings (as Kinova reports joint feedback regardless of
                # whether a joint is continuous), while send_joint_angles/Kinova's trajectory
                # validation expects each limited (non-continuous) joint's signed range. Joints
                # 1, 3, 5 are limited to +-128.9/147.8/120.3 degrees, so their raw values (345, 219,
                # 320) were rejected by Kinova's validation -- wrap to the equivalent signed angle.
                raw_degrees = np.array([11, 345, 170, 219, 5, 320, 80])
                signed_degrees = np.where(raw_degrees > 180, raw_degrees - 360, raw_degrees)
                angles = np.deg2rad(signed_degrees)
                success = self.kinova.send_joint_angles(angles)
                if not success:
                    rospy.logerr("driver_subscriber::callback_home_trigger: send_joint_angles failed, robot NOT sent home")
                print("Kinova sent home:", success, "Joints:", self.kinova.position)
            finally:
                self.busy_pub.publish(Bool(False))
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

