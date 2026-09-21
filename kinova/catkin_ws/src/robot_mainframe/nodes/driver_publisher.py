import os
import sys
import rospy
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Empty, Bool
import numpy as np


'''
Sep 18 2026

This file has all the different robot control methods, like XBOX_TELEOP etc!!!

This node publishes driving commands to the driver_subscriber nodes

DOES NOT ACTUALLY DRIVE KINOVA, that is the job to the driver node subscriber since it recv all cmds

'''

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from const import *
from position_log import pop_last_position, DEFAULT_LOG_PATH

class Driver:
    # consider Driver class as kind of like a Trait in Rust st different types of Drivers need to satisfy the implementation (methods) of the trait Driver
    def __init__(self, driver_name=XBOX_TELEOP):
        self.driver_name : str = driver_name # name can be "XBOX_TELEOP" or "JOYSTICK_TELEOP" or "VLA_ROLLOUT" etc, just a string for logging
        self.driver_object = None #the object instance of the type of driver (see below for all the driver object classes)
        if driver_name == XBOX_TELEOP:
            self.driver_object = XboxController()
        elif driver_name == KEYBOARD_TELEOP:
            self.driver_object = KeyboardController()

    def publish_joint_state(self, joint_state):
        try:
            self.driver_object.publish_joint_state(joint_state)
        except AttributeError:
            rospy.loginfo(f"driver_publisher::publish_joint_state: Driver [name={self.driver_name}, obj={self.driver_object}] has no method publish_joint_state")

    def publish_gripper_state(self, gripper_state):
        try:
            self.driver_object.publish_gripper_state(gripper_state)
        except AttributeError:
            rospy.loginfo(f"driver_publisher::publish_gripper_state: Driver [name={self.driver_name}, obj={self.driver_object}] has no method publish_gripper_state")



class XboxController: #name is XBOX_TELEOP
    '''
    Reads sensor_msgs/Joy from the ros-noetic-joy joy_node and republishes it as
    cartesian velocity / gripper / home commands on the /driver/* topics that
    driver_subscriber.py listens to (that node is the only one allowed to touch
    the kinova object directly).
    '''
    def __init__(self):
        self.cartesian_velocity_pub = rospy.Publisher('/driver/cartesian_velocity', Twist, queue_size=1)
        self.gripper_velocity_pub = rospy.Publisher('/driver/gripper_velocity', Float32, queue_size=1)
        self.home_trigger_pub = rospy.Publisher('/driver/home_trigger', Empty, queue_size=1)
        self.record_toggle_pub = rospy.Publisher('/driver/record_toggle', Empty, queue_size=1)
        self.joint_state_pub = rospy.Publisher('/driver/joint_state', JointState, queue_size=1)
        self.gripper_state_pub = rospy.Publisher('/driver/gripper_state', Float32, queue_size=1)
        self._prev_home_button = 0
        self._prev_mode_button = 0
        self._prev_record_button = 0
        self.mode = 0  # 0 = cartesian velocity (translation), 1 = robot orientation (rotation)

        # UNDO: pop_last_position() returns the same [j0..j6 (deg), gripper (%)] row
        # position_log.py pushes -- convert joints back to radians and forward as an absolute
        # target on /driver/joint_state + /driver/gripper_state, the same topics VLA inference
        # uses. Rate-limited (not edge-triggered) so holding the button keeps undoing.
        self.undo_log_path = rospy.get_param('~undo_log_path', DEFAULT_LOG_PATH)
        self._last_undo_time = rospy.Time(0)

        # driver_subscriber.py sets this True while it's mid-way through a blocking,
        # action-based command (currently just homing) -- an active cartesian_velocity stream
        # conflicts with an in-progress trajectory action, so pause streaming while busy.
        self.busy = False
        rospy.Subscriber('/driver/busy', Bool, self._busy_callback)

        rospy.Subscriber('/joy', Joy, self._joy_callback)
        rospy.loginfo("XboxController: subscribed to /joy")

    def _busy_callback(self, msg: Bool):
        self.busy = msg.data

    def _joy_callback(self, msg: Joy):
        '''
        mode 0 and mode 1 exist but are not implemented - sep 18 2026
        '''
        # left stick -> x/y translation, triggers (axes 2, 5) -> z translation,
        # right stick -> x/y rotation, Y(1)/B(3) -> roll. Ported from the other
        # project's joy_callback (joy_type == 1 branch).
        roll = float(msg.buttons[1] - msg.buttons[3])
        axes_vector = [
            msg.axes[1],
            msg.axes[0],
            (1 / (msg.axes[5] + 1.1) - 1 / (msg.axes[2] + 1.1)) / 10,
            -msg.axes[4] / 2,
            msg.axes[3],
            roll,
        ]
        axes_vector = [v / 2 for v in axes_vector]

        if msg.buttons[0] and not self._prev_mode_button:  # A, rising edge only
            self.mode = 1 - self.mode
            rospy.loginfo(f"XboxController: mode={self.mode} ({'cartesian velocity' if self.mode == 0 else 'orientation'})")
        self._prev_mode_button = msg.buttons[0]

        twist = Twist()
        
        twist.linear.x = axes_vector[0] * MAXV_TX
        twist.linear.y = axes_vector[1] * MAXV_TY
        twist.linear.z = axes_vector[2] * MAXV_TZ
    
        twist.angular.x = axes_vector[3] * MAXV_RX
        twist.angular.y = axes_vector[4] * MAXV_RY
        twist.angular.z = axes_vector[5] * MAXV_RZ
        if not self.busy:
            self.cartesian_velocity_pub.publish(twist)

        # LB/RB command the gripper to move open/closed at GRIPPER_SPEED via Kinova's GRIPPER_SPEED
        # mode (continuous velocity, not a position target -- see
        # driver_subscriber.py::callback_gripper_velocity for why: computing "current position +
        # delta" every tick is unstable when feedback lags an in-flight move, which is what
        # caused the gripper to visibly reverse direction while still being held one way).
        # Sign convention (positive=open, negative=close) matches the reference "other project"
        # implementation this was ported from.
        # Published every tick, like cartesian_velocity, so releasing reliably sends the stop (0).
        if msg.buttons[4]:  # LB -> open gripper
            gripper_speed = GRIPPER_SPEED
        elif msg.buttons[5]:  # RB -> close gripper
            gripper_speed = -GRIPPER_SPEED
        else:
            gripper_speed = 0.0
        self.gripper_velocity_pub.publish(Float32(gripper_speed))

        if msg.buttons[6] and not self._prev_home_button:  
            self.home_trigger_pub.publish(Empty())
        self._prev_home_button = msg.buttons[6]

        if msg.buttons[7] and not self._prev_record_button:
            self.record_toggle_pub.publish(Empty())
        self._prev_record_button = msg.buttons[7]

        if msg.buttons[UNDO_BUTTON_INDEX]:
            now = rospy.Time.now()
            if (now - self._last_undo_time).to_sec() >= UNDO_SECOND_RATE:
                self._trigger_undo()
                self._last_undo_time = now

    def _trigger_undo(self):
        row = pop_last_position(self.undo_log_path)
        if row is None:
            rospy.loginfo("XboxController: undo pressed but position log is empty, nothing to undo")
            return

        joints_deg, gripper_pct = row[:KINOVA_DOF], row[KINOVA_DOF]
        joint_cmd = JointState()
        joint_cmd.header.stamp = rospy.Time.now()
        joint_cmd.position = np.radians(joints_deg).tolist()
        self.joint_state_pub.publish(joint_cmd)
        self.gripper_state_pub.publish(Float32(gripper_pct))
        rospy.loginfo(f"XboxController: undo -> joints(deg)={joints_deg}, gripper={gripper_pct}")
     


class KeyboardController:
    ''' this is to show that any kind of controller can be impl'''
    def __init__(self):
        pass


def main():
    rospy.init_node('driver_publisher', anonymous=True)
    rospy.loginfo("Starting a Driver publisher node ...")
    Driver(XBOX_TELEOP)
    rospy.spin()


if __name__ == '__main__':
    main()
