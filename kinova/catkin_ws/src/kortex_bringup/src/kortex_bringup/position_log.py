#!/usr/bin/python3
'''
Sep 20 2026

Keep a record of the N last joint positions of the robot; NOt based on time but based on difference in joint position by a certain delta (can be coarse)

so for example, if the robot moved forward, i paused it, for 30 seconds it was idle, then i rotated it and moved it again,
the log would not have 30 seconds worth of one same position.

how to do this: subscribe to joint position, save last position, if || last position - current position || _infinity norm has delta > DELTA_JOINT_POSITION
then append to the txt (save as file because this allows different nodes to access the same file)

the whole gist is an undo button that is similar to a Backwards button in a browser or art app:
suppose you write

1 2 3

then undo 3:

1 2

then write 4:

1 2 4

then suppose you undo 4:

1 2

now if you undo again, you undo 2, since this is like a STACK:

1

Now, this is very intuitive for a web browser, and should be intuitive for a human operating an arm with very little DOF control

the idea is that this heightens the satisfaction of the user because the recovery is HIGH, and while the control of the model itself is low, the ability to correct it is super accessible.
worst case scenario is the user demanding regenerating CaP again and again.

implement as stack:

write to the bottom of the file always, and read at the bottom of the file always and subsequently clear that line


file looks something like this:

j0 j1 j2 j3 j4 j5 j6 g
j0 j1 j2 j3 j4 j5 j6 g
j0 j1 j2 j3 j4 j5 j6 g
j0 j1 j2 j3 j4 j5 j6 g
...
etc



Have this file be read by the driver_publisher whenever the "undo" mode is activated
'''


import os
import sys
import fcntl
import rospy
from sensor_msgs.msg import JointState
from kortex_driver.msg import BaseCyclic_Feedback
import numpy as np


sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from const import *


def _row_to_line(position_deg):
    return " ".join(f"{v:.6f}" for v in position_deg) + "\n"


def _line_to_row(line):
    return np.array([float(v) for v in line.split()])


def push_position(path, position_deg):
    '''Append one row (7 joint degrees + gripper percent) to the bottom of the stack file.'''
    with open(path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.write(_row_to_line(position_deg))
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)


def pop_last_position(path):
    '''
    Pop (read + remove) the bottom row of the stack file -- the "undo" consumer, meant to be
    called from driver_publisher.py when undo mode is triggered. Returns None if the file is
    missing/empty (nothing left to undo to).
    '''
    if not os.path.exists(path):
        return None
    with open(path, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            lines = f.readlines()
            if not lines:
                return None
            last_line = lines.pop()
            f.seek(0)
            f.writelines(lines)
            f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return _line_to_row(last_line)


class PositionLogNode:
    def __init__(self):
        '''
        listens to /my_gen3/joint_states for JointState (arm) and /my_gen3/base_feedback for
        BaseCyclic_Feedback (gripper) -- the same topics control_robot.py / kinova_gen3.py
        already publish and read, so this node logs whatever the arm is actually doing instead
        of a separate /joint_states + /gripper_state pair that nothing publishes.
        '''
        rospy.init_node('robot_joint_position_log', anonymous=True)
        self.log_path = rospy.get_param('~log_path', DEFAULT_LOG_PATH)
        rospy.loginfo(f"PositionLogNode: logging to {self.log_path}")

        # NaN-initialized (not zeros) so "has every field been populated at least once" can be
        # checked with an isnan test, instead of a zero position looking indistinguishable from
        # "not received yet" -- KINOVA_DOF joints (radians from /my_gen3/joint_states) + 1 gripper
        # (percent 0-100 from /my_gen3/base_feedback's gripper_feedback), stored here converted to
        # degrees for the joints so the whole row is in human-readable units matching
        # DELTA_JOINT_POSITION_DEG.
        self.last_position = None  # only set once we have a first full reading to compare against
        self.current_position = np.full(KINOVA_DOF + 1, np.nan)
        self._warned_gripper = False

        rospy.Subscriber('/my_gen3/joint_states', JointState, self.callback_joint_state)
        rospy.loginfo(f"Initialized Log sub on topic /my_gen3/joint_states")
        rospy.Subscriber('/my_gen3/base_feedback', BaseCyclic_Feedback, self.callback_base_feedback)
        rospy.loginfo(f"Initialized Log sub on topic /my_gen3/base_feedback")

    def callback_joint_state(self, data):
        self.current_position[0:KINOVA_DOF] = np.degrees(data.position[:KINOVA_DOF])
        self._maybe_log()

    def callback_base_feedback(self, data: BaseCyclic_Feedback):
        try:
            self.current_position[KINOVA_DOF] = float(
                data.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position)
        except (AttributeError, IndexError):
            if not self._warned_gripper:
                rospy.logwarn("No gripper feedback in base_feedback; position log will never see a full row.")
                self._warned_gripper = True
            return
        self._maybe_log()

    def _maybe_log(self):
        if np.isnan(self.current_position).any():
            return  # still waiting on a first reading for some field

        if self.last_position is None:
            # First fully-populated reading -- nothing to compare against yet, just seed it.
            self.last_position = self.current_position.copy()
            push_position(self.log_path, self.current_position)
            return

        delta = np.linalg.norm(self.current_position - self.last_position, ord=np.inf)
        if delta >= DELTA_JOINT_POSITION_DEG:
            push_position(self.log_path, self.current_position)
            self.last_position = self.current_position.copy()


def main():
    rospy.sleep(1)
    rospy.loginfo("Starting a Kinova Joint State Position Log node ...")
    node = PositionLogNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down PositionLogNode...")


if __name__ == '__main__':
    main()
