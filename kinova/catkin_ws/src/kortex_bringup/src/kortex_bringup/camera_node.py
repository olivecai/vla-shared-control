#!/usr/bin/python3

'''
June 23 2025
Updated Sep 17 2026 - images published as RGB for VLA use (previously was publish as grayscale for visual servoing)
Updated Sep 18 2026 - switched from cv2.VideoCapture(/dev/videoN) to pyrealsense2, since
RealSense devices expose multiple /dev/video* nodes per physical camera (depth, color,
2x infrared, etc.), and V4L2 index/path enumeration is unreliable once a RealSense is
attached. pyrealsense2 talks to the camera directly and lets us select a specific stream
(and specific physical device via serial number) unambiguously.
Updated Sep 21 2026 - switched from talking to pyrealsense2 directly to subscribing to the
official realsense-ros driver (the `realsense2_camera` ROS package). That driver already
handles device enumeration/selection by serial, reconnects, and multi-stream config, so
this node's only job now is to republish whatever color image it publishes under this
project's naming convention (/cameras/cam{id}, /cam_id) that record_vla.py / record.py
expect -- the rest of the codebase doesn't need to know which camera driver is running.

Run the RealSense ROS driver separately (one instance per physical camera), then point
this node at its color image topic.

SUBSCRIBES to:
    <_color_topic> (default /camera{cam_id}/color/image_raw)  (sensor_msgs/Image)

PUBLISHES to:
1. topic '/cameras/cam{id}' images as Image message (rgb8)
2. topic '/cam_id' the camera id as Int32 message

Usage:
```
source /home/user/kinova/catkin_ws/devel/setup.bash

# In one terminal, start the RealSense ROS driver for this physical camera:
roslaunch realsense2_camera rs_camera.launch camera:=camera0 serial_no:=<serial number>

# In another terminal, start this adapter node pointed at that driver's topic:
rosrun kortex_bringup camera_node.py _cam_id:=0 _color_topic:=/camera0/color/image_raw

# In a third terminal, view the republished images:
rosrun image_view image_view image:=/cameras/cam0
```

Check the rostopic:
```
# List available topics
rostopic list

# View the raw image topic
rostopic echo /cameras/cam0

# Get info about the topic
rostopic info /cameras/cam0
```
'''

import sys
import os

import rospy
import cv_bridge

from sensor_msgs.msg import Image # ROS Image message
from std_msgs.msg import Int32

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from const import *

# create a camera node that republishes the realsense-ros driver's color stream under
# this project's /cameras/cam{id} naming convention.

class CameraNode:
    '''
    Each CAMERA in use should correspond to a CAMERA NODE.
    Initialize a CameraNode object with argument as index of the camera.
    Subscribes to a realsense2_camera driver's color image topic and republishes it.
    '''
    def __init__(self):
        '''
        Subscribe to the realsense-ros driver's color stream and republish it as
        /cameras/cam{id} (+ /cam_id), matching what record_vla.py / record.py expect.
        '''
        rospy.init_node("camera_node", anonymous=True) #anonymous since there may be multiple cameras

        cam_param = rospy.search_param("cam_id")
        self.cam_id = rospy.get_param(cam_param, None) #this is our identifier for our camera

        if self.cam_id is None:
            rospy.logwarn("Must pass camera index as _cam_id:=<cam_id>")
            exit()

        rospy.delete_param(cam_param) # delete param so its needed for future runs.

        # Topic the realsense2_camera driver is publishing color frames on. Default assumes
        # the driver was launched with camera:=camera{cam_id} (e.g. camera:=camera0 for
        # _cam_id:=0). Override with _color_topic:=... if you launched it under a different
        # name -- e.g. camera:=cam to match record_vla.py's own default image topic of
        # /cam/color/image_raw, or the driver's own default of /camera/color/image_raw.
        topic_param = rospy.search_param("color_topic")
        self.color_topic = rospy.get_param(topic_param, f"/camera{self.cam_id}/color/image_raw")
        rospy.delete_param(topic_param)

        rospy.loginfo(
            f"Camera {self.cam_id}: subscribing to {self.color_topic}, "
            f"republishing on /cameras/cam{self.cam_id}"
        )

        self.bridge = cv_bridge.CvBridge()
        self.cap_pub = rospy.Publisher(f"/cameras/cam{self.cam_id}", Image, queue_size=10)
        self.id_pub = rospy.Publisher("/cam_id", Int32, queue_size=1)

        rospy.loginfo(f"Waiting for first frame on {self.color_topic} ...")
        rospy.wait_for_message(self.color_topic, Image)

        self.image_sub = rospy.Subscriber(self.color_topic, Image, self.image_callback, queue_size=1)

    def image_callback(self, msg):
        '''
        Callback: convert the driver's image to rgb8 and republish under this project's
        naming convention. realsense2_camera normally already publishes rgb8, but going
        through cv_bridge here makes the republished topic robust regardless of how the
        driver (or a different camera driver entirely) is configured.
        '''
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        except cv_bridge.CvBridgeError as e:
            rospy.logwarn(f"Couldn't convert image frame for camera {self.cam_id}: {e}")
            return

        rosImage = self.bridge.cv2_to_imgmsg(frame, encoding="rgb8")
        rosImage.header = msg.header
        rosImage.header.frame_id = f"{self.cam_id}"

        self.id_pub.publish(Int32(data=self.cam_id))
        self.cap_pub.publish(rosImage)

    def shutdown(self):
        self.image_sub.unregister()


def main(args):
    rospy.loginfo("Starting a CameraNode ...")
    node = CameraNode()

    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down Camera...")
    finally:
        node.shutdown()

if __name__ == '__main__':
    main(sys.argv)
