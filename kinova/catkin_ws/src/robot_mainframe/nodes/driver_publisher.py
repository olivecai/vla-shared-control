import rospy
from sensor_msgs.msg import JointState, Joy
from geometry_msgs.msg import Twist
from std_msgs.msg import Float32, Empty
import numpy as np
from const import *

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
        self.cartesian_velocity_pub = rospy.Publisher('/driver/cartesian_velocity', Twist, queue_size=10)
        self.gripper_pub = rospy.Publisher('/driver/gripper_state', Float32, queue_size=10)
        self.home_trigger_pub = rospy.Publisher('/driver/home_trigger', Empty, queue_size=1)
        self._prev_home_button = 0
        self._prev_mode_button = 0
        self.mode = 0  # 0 = cartesian velocity (translation), 1 = robot orientation (rotation)

        rospy.Subscriber('/joy', Joy, self._joy_callback)
        rospy.loginfo("XboxController: subscribed to /joy")

    def _joy_callback(self, msg: Joy):
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

        # mode 0: translation only (cartesian velocity), mode 1: rotation only (robot orientation)
        twist = Twist()
        if self.mode == 0:
            twist.linear.x = axes_vector[0] * MAXV_TX
            twist.linear.y = axes_vector[1] * MAXV_TY
            twist.linear.z = axes_vector[2] * MAXV_TZ
        else:
            twist.angular.x = axes_vector[3] * MAXV_RX
            twist.angular.y = axes_vector[4] * MAXV_RY
            twist.angular.z = axes_vector[5] * MAXV_RZ
        self.cartesian_velocity_pub.publish(twist)

        if msg.buttons[4]:  # LB -> open gripper
            self.gripper_pub.publish(Float32(0.0))
        elif msg.buttons[5]:  # RB -> close gripper
            self.gripper_pub.publish(Float32(100.0))

        if msg.buttons[6] and not self._prev_home_button:  # Start (rising edge so only press once)
            self.home_trigger_pub.publish(Empty())
        self._prev_home_button = msg.buttons[6]


class KeyboardController:
    def __init__(self):
        pass


def main():
    rospy.init_node('driver_publisher', anonymous=True)
    rospy.loginfo("Starting a Driver publisher node ...")
    Driver(XBOX_TELEOP)
    rospy.spin()


if __name__ == '__main__':
    main()
