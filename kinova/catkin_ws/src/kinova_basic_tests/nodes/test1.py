#!/usr/bin/env python3

import numpy as np
import rospy

from kortex_bringup import KinovaGen3

#first we initialize a node
rospy.init_node('test1_node', anonymous=False) #only one node so it should not be anon

#then we create a robot node 
gen3 = KinovaGen3()

print(gen3)

angles = np.deg2rad(np.array([11, 345, 170, 219, 5, 320, 80])) #as of Aug 19 2026, verified this is good home position
success = gen3.send_joint_angles(angles)
print("Kinova sent home. Joints:", gen3.position)
'''
#then move the robot by specifying joint velocities
velocities = np.array([1,0,0,0,0,0,0])
success = gen3.send_joint_velocities(velocities)

rospy.sleep(3.)

velocities=np.array([0,0,0,0,0,0,0])
success = gen3.send_joint_velocities(velocities)
print("Kinova after joint velocity changes:", gen3.position)

'''