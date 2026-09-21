#!/usr/bin/python3
"""
Round 2: send_gripper_command's single-shot GRIPPER_SPEED call only moved the gripper a tiny bit
before it stopped on its own, contradicting Kinova's own documented example (which implies a
single speed command should sustain motion until the limit or a new command). This test checks
whether this gripper/firmware actually needs the speed command continuously refreshed to keep
moving, like the low-level BaseCyclic streaming API does -- bypasses send_gripper_command's
de-dupe (which would otherwise skip identical repeated values) and resends the raw request
directly every 50ms instead of once.

Run inside the container (source setup.bash first):
    python3 /home/user/kinova/catkin_ws/scripts/gripper_isolation_test.py
"""
import rospy
from kortex_bringup import KinovaGen3
from kortex_driver.msg import Finger, GripperMode
from kortex_driver.srv import SendGripperCommandRequest

rospy.init_node('gripper_isolation_test', anonymous=True)
k = KinovaGen3()
rospy.sleep(1)


def send_speed_raw(value):
    req = SendGripperCommandRequest()
    finger = Finger()
    finger.finger_identifier = 0
    finger.value = value
    req.input.gripper.finger.append(finger)
    req.input.mode = GripperMode.GRIPPER_SPEED
    k.send_gripper_command_srv(req)


print("Refreshing close (-0.3) for 3s ...")
end_time = rospy.Time.now() + rospy.Duration(3.0)
while rospy.Time.now() < end_time:
    send_speed_raw(-0.3)
    rospy.sleep(0.05)

print("Sending stop (0.0) ...")
send_speed_raw(0.0)
print("Done. Did it move continuously this time, for the full 3s, instead of a tiny bit?")
