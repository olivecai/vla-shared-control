
#!/usr/bin/python3
'''
Sep 16 2026

DRIVES actual kinova movement; listens to /driver

This is the ONLY file in the entire repo that is allowed to touch the kinova joints!
'''
import os
import sys
import threading
import rospy
from sensor_msgs.msg import JointState
from kortex_bringup import KinovaGen3
from kortex_driver.msg import Finger, GripperMode
from kortex_driver.srv import SendGripperCommandRequest
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
        rospy.Subscriber('/driver/gripper_velocity', Float32, self.callback_gripper_velocity)
        rospy.loginfo(f"Initialized Driver sub on topic /driver/gripper_velocity")
        rospy.Subscriber('/driver/cartesian_velocity', Twist, self.callback_cartesian_velocity)
        rospy.loginfo(f"Initialized Driver sub on topic /driver/cartesian_velocity")
        rospy.Subscriber('/driver/home_trigger', Empty, self.callback_home_trigger)
        rospy.loginfo(f"Initialized Driver sub on topic /driver/home_trigger")

        # Lets driver_publisher.py know when an action-based command (currently just homing) is
        # in progress, so it can pause streaming cartesian_velocity -- an active velocity stream
        # conflicts with an in-progress waypoint trajectory action (see callback_home_trigger).
        self.busy_pub = rospy.Publisher('/driver/busy', Bool, queue_size=1, latch=True)
        self.busy_pub.publish(Bool(False))

        # send_gripper_command() blocks for 0.5s per call (kortex_bringup's own kinova_gen3.py),
        # but driver_publisher.py streams a new gripper target on every /joy tick (~JOY_HZ times a
        # second) while LB/RB is held. Processing each one inline in callback_gripper_state would
        # both stall every other driver callback for that 0.5s AND queue up a large backlog of
        # now-stale targets, which is what actually made the gripper feel jerky/slow. Instead,
        # callback_gripper_state just records the latest requested target, and this background
        # thread keeps sending whatever the FRESHEST target is, dropping the stale ones in between.
        self._gripper_lock = threading.Lock()
        self._gripper_target = None
        threading.Thread(target=self._gripper_worker, daemon=True).start()

        rospy.spin()

    def _send_gripper_position(self, fraction):
        '''
        kortex_bringup's KinovaGen3.send_gripper_command() now uses GRIPPER_SPEED mode (for
        teleop, see callback_gripper_velocity below), so absolute-position control (used by VLA
        inference/rollout via callback_gripper_state) needs its own GRIPPER_POSITION request
        built directly here instead.
        '''
        req = SendGripperCommandRequest()
        finger = Finger()
        finger.finger_identifier = 0
        finger.value = fraction
        req.input.gripper.finger.append(finger)
        req.input.mode = GripperMode.GRIPPER_POSITION
        try:
            self.kinova.send_gripper_command_srv(req)
        except rospy.ServiceException:
            rospy.logerr("driver_subscriber::_send_gripper_position: Failed to call SendGripperCommand")

    def _gripper_worker(self):
        last_sent = None
        while not rospy.is_shutdown():
            with self._gripper_lock:
                target = self._gripper_target
            if target is not None and target != last_sent and self.kinova:
                self._send_gripper_position(target)
                last_sent = target
            else:
                rospy.sleep(0.1)

    def callback_joint_state(self, data):
        rospy.loginfo(f"driver_subscriber::callback_joint_state: Received {data.position} from time {data.header.stamp}. Current time: {rospy.Time.now()}")
        if self.kinova:
            # data.position is in radians (standard JointState convention) and JOINT_LIMIT is
            # radians too, but kinova_gen3.py's send_joint_angles now expects degrees.
            positions_rad = [np.clip(p, *JOINT_LIMIT[i]) for i, p in enumerate(data.position)]
            positions_deg = np.degrees(positions_rad)
            # send_joint_angles is action-based, same as go_home() -- pause driver_publisher.py's
            # cartesian_velocity streaming for the duration, same reasoning as callback_home_trigger.
            self.busy_pub.publish(Bool(True))
            try:
                self.kinova.send_joint_angles(positions_deg)
            finally:
                self.busy_pub.publish(Bool(False))
        else:
            rospy.loginfo(f"WARNING: Kinova not found.")

    def callback_gripper_state(self, data: Float32):
        rospy.loginfo(f"driver_subscriber::callback_gripper_state: Receieved {data.data}")
        if self.kinova:
            # data.data is a percentage (0=open, 100=closed), matching /gripper_state;
            # GRIPPER_POSITION expects a fraction (0.0=open, 1.0=closed). The actual (blocking)
            # hardware call happens in _gripper_worker -- this just records the latest requested
            # target so this callback returns immediately.
            fraction = np.clip(data.data / 100.0, *JOINT_LIMIT[7])
            with self._gripper_lock:
                self._gripper_target = fraction
        else:
            rospy.loginfo(f"WARNING: Kinova not found.")

    def callback_gripper_velocity(self, data: Float32):
        '''
        Continuous gripper speed control for teleop, as an alternative to the
        callback_gripper_state/absolute-position path above. GRIPPER_POSITION requires
        recomputing a "current position + delta" target every tick, which is unstable when
        feedback lags behind an in-flight physical move (the target can undershoot where the
        gripper is already heading, causing it to visibly reverse) -- GRIPPER_SPEED mode instead
        just commands a continuous direction+speed with no target math, so there's nothing to
        regress. kinova_gen3.py's send_gripper_command() now uses GRIPPER_SPEED natively (and
        de-dupes identical repeated values), so this just forwards to it directly.

        data.data: signed speed, -100 (open, full speed) .. 100 (close, full speed). Sign
        convention per Kinova's Kortex API; flip it in driver_publisher.py if it's backwards.
        '''
        if self.kinova:
            self.kinova.send_gripper_command(data.data)
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
                # go_home() uses Kinova's own built-in Home action (a predefined action stored on
                # the robot itself, ID #2) instead of hand-picked joint angles -- no more manual
                # degree/radian or wraparound bookkeeping. It also zeroes joint velocity and sets
                # going_home/movement_blocked internally, guarding send_joint_speeds_command --
                # but that doesn't cover cartesian_velocity, which is what our joystick teleop
                # actually streams, so /driver/busy (published True above, False in the finally
                # block below) still does its job of pausing driver_publisher.py's velocity
                # publishing for the whole trajectory, not just the start of it.
                success = self.kinova.go_home()
                if not success:
                    rospy.logerr("driver_subscriber::callback_home_trigger: go_home failed, robot NOT sent home")
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

