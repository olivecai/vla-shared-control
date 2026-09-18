#!/usr/bin/python3
 
'''
June 23 2025
Updated Sep 17 2026 - images published as RGB for VLA use (previously was publish as grayscale for visual servoing)
Updated Sep 18 2026 - switched from cv2.VideoCapture(/dev/videoN) to pyrealsense2, since
RealSense devices expose multiple /dev/video* nodes per physical camera (depth, color,
2x infrared, etc.), and V4L2 index/path enumeration is unreliable once a RealSense is
attached. pyrealsense2 talks to the camera directly and lets us select a specific stream
(and specific physical device via serial number) unambiguously.
 
Connect to camera
PUBLISH to two topics:
1. topic '/cameras/cam{id}' images as Image message
2. topic '/cam_id' the camera id as Int32 message
 
Usage:
```
source /home/user/kinova/catkin_ws/devel/setup.bash
 
# In one terminal, start the camera node:
rosrun robot_mainframe camera_node.py _cam_id:=0
 
# In another terminal, view the images:
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
import rospy
import numpy as np
import pyrealsense2 as rs
 
from sensor_msgs.msg import Image # ROS Image message
from std_msgs.msg import Int32
 
import os
 
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from const import *
 
# create a camera node that publishes a ros message image every second.
 
class CameraNode:
    '''
    Each CAMERA in use should correspond to a CAMERA NODE.
    Initialize a CameraNode object with argument as index of the camera.
    '''
    def __init__(self):
        '''
        Publish RealSense color stream via pyrealsense2
        '''
        rospy.init_node("camera_node", anonymous=True) #anonymous since there may be multiple cameras
 
        cam_param = rospy.search_param("cam_id")
        self.cam_id=rospy.get_param(cam_param, None) #this is our identifier for our camera
 
        if self.cam_id is None:
            rospy.logwarn("Must pass camera index as _cam_id:=<cam_id>")
            exit()
 
        rospy.delete_param(cam_param) # delete param so its needed for future runs.
        rospy.loginfo(f"Initialized Camera on topic /cameras/cam{self.cam_id}")
 
        # cam_id here selects which physically-attached RealSense to use, by index into
        # the enumerated device list (NOT a /dev/videoN path). If you have only one
        # RealSense plugged in, _cam_id:=0 is fine. With multiple cameras, prefer pinning
        # by serial number (see REALSENSE_SERIALS note below) so the mapping doesn't
        # depend on USB enumeration order.
        ctx = rs.context()
        devices = ctx.query_devices()
        if len(devices) == 0:
            rospy.logerr("Error: No RealSense devices found.")
            exit()
 
        serial = None
        try:
            # Optional: look up a fixed serial number for this cam_id from const.py,
            # e.g. REALSENSE_SERIALS = {0: "123622270XXX", 1: "045322070YYY"}
            serial = REALSENSE_SERIALS.get(int(self.cam_id))
        except NameError:
            pass  # REALSENSE_SERIALS not defined in const.py -- fall back to index-based selection
 
        if serial is None:
            try:
                device = devices[int(self.cam_id)]
            except (ValueError, IndexError):
                rospy.logerr(
                    f"Error: Could not find RealSense device at index {self.cam_id} "
                    f"({len(devices)} device(s) found)."
                )
                exit()
            serial = device.get_info(rs.camera_info.serial_number)
 
        rospy.loginfo(f"Opening RealSense camera {self.cam_id} (serial {serial})")
 
        self.pipeline = rs.pipeline()
        config = rs.config()
        config.enable_device(serial)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.rgb8, 30)
 
        try:
            self.pipeline.start(config)
        except RuntimeError as e:
            rospy.logerr(f"Error: Could not open RealSense camera {self.cam_id} (serial {serial}): {e}")
            exit()
 
        self.cap_pub = rospy.Publisher(f"/cameras/cam{self.cam_id}", Image, queue_size=10)
        self.id_pub = rospy.Publisher("/cam_id", Int32, queue_size=1)
        self.stop_capture = False
 
        #timer to repeatedly publish a capture from the RealSense pipeline
        rospy.Timer(rospy.Duration(0.1), self.publish_capture_callback)
 
    def publish_capture_callback(self, event):
        '''
        Callback: publish a ROS Image Message from the RealSense color stream.
        '''
        if self.stop_capture:
            return
 
        print(f"publishing camera {self.cam_id} video capture...")
        self.id_pub.publish(Int32(data=self.cam_id))
 
        try:
            # short timeout so a stalled camera warns instead of blocking the ROS timer thread
            frames = self.pipeline.wait_for_frames(timeout_ms=1000)
        except RuntimeError:
            rospy.logwarn(f"Couldn't get image frame for camera {self.cam_id} (timeout)")
            return
 
        color_frame = frames.get_color_frame()
        if not color_frame:
            rospy.logwarn(f"Couldn't get image frame for camera {self.cam_id}")
            return
 
        frame = np.asanyarray(color_frame.get_data())  # already RGB, since stream was opened as rs.format.rgb8
 
        # build the ros image message
        rosImage = Image()
        rosImage.height, rosImage.width = frame.shape[:2]
        rosImage.encoding = "rgb8"
        rosImage.step = frame.shape[1] * 3
        rosImage.data = frame.tobytes()
        rosImage.header.frame_id = f"{self.cam_id}"
 
        self.cap_pub.publish(rosImage)
 
    def shutdown(self):
        self.stop_capture = True
        self.pipeline.stop()
 
 
def main(args):
    rospy.sleep(5)
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
 
