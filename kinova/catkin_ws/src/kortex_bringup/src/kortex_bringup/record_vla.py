#!/usr/bin/env python3
"""
Record teleop demonstrations for OpenVLA fine-tuning (https://github.com/openvla/openvla).

only subscribs

reads
    /my_gen3/joint_states      (sensor_msgs/JointState)          arm joint angles
    /my_gen3/base_feedback     (kortex_driver/BaseCyclic_Feedback) gripper position + tool pose
    /cam/color/image_raw       (sensor_msgs/Image)               RGB (override with --image-topics)
    /joy                       (sensor_msgs/Joy)                 buttons[7] starts/stops an episode

Writes one folder per episode under --out-dir:
    episode_<timestamp>/
        instruction.txt         -- language instruction for this episode
        joint_positions.npy     -- (T, 7) float32, radians (as published by joint_states)
        gripper_positions.npy   -- (T,) float32, 0=open, 100=closed (NaN if feedback unavailable)
        ee_pose.npy             -- (T, 6) float32, tool x,y,z (m) + theta x,y,z (deg) from base_feedback
        timestamps.npy          -- (T,) float64, ROS timestamps (joint_states header stamp)
        cam{i}/000000.jpg ...   -- one RGB frame per timestep, per image topic (i = index in --image-topics)

Usage:
    # Already running: kortex bringup, camera (/cam/color/image_raw), joy_node, and xbox_control script (iris_control).
    python3 record_vla.py --out-dir data --hz 5

    # If you instead run the realsense-ros driver under a different camera name and use
    # camera_node.py to republish it as /cameras/cam0 (rgb8):
    #   roslaunch realsense2_camera rs_camera.launch camera:=camera0
    #   rosrun kortex_bringup camera_node.py _cam_id:=0 _color_topic:=/camera0/color/image_raw
    python3 record_vla.py --image-topics /cameras/cam0

Controls: type an instruction in the terminal, then press the controller's button 7
(same button the old recorder used) to START, and again to STOP. Button 6 (home) in your
controller script is untouched.
"""
import argparse
import os
import threading
import time

import cv2
import cv_bridge
import message_filters
import numpy as np
import rospy
from kortex_driver.msg import BaseCyclic_Feedback
from sensor_msgs.msg import Image, JointState, Joy

TOGGLE_BUTTON = 7
TOGGLE_DEBOUNCE_S = 0.5


class Recorder:
    def __init__(self, image_topics, hz):
        rospy.init_node("vla_recorder", anonymous=True)
        self.bridge = cv_bridge.CvBridge()
        self.image_topics = image_topics
        self.period = 1.0 / hz

        self.active = False
        self.frames = []
        self._lock = threading.Lock()
        self._last_saved_t = 0.0

        self.latest_gripper = float("nan")
        self.latest_tool = np.full(6, np.nan, dtype=np.float32)

        self._toggle_event = threading.Event()
        self._prev_button = 0
        self._last_toggle = 0.0
        self._warned_gripper = False

        rospy.loginfo("Waiting for joint states, base feedback, and camera image(s) ...")
        rospy.wait_for_message("/my_gen3/joint_states", JointState)
        rospy.wait_for_message("/my_gen3/base_feedback", BaseCyclic_Feedback)
        for topic in image_topics:
            rospy.wait_for_message(topic, Image)

        # base_feedback has no header, so cache it instead of putting it in the synchronizer
        rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self._feedback_cb)
        rospy.Subscriber("/joy", Joy, self._joy_cb)

        joints_sub = message_filters.Subscriber("/my_gen3/joint_states", JointState)
        img_subs = [message_filters.Subscriber(t, Image) for t in image_topics]
        ts = message_filters.ApproximateTimeSynchronizer([joints_sub] + img_subs, 100, slop=0.1)
        ts.registerCallback(self._sync_cb)

    # ---- callbacks -------------------------------------------------------
    def _feedback_cb(self, msg):
        self.latest_tool = np.array([
            msg.base.tool_pose_x, msg.base.tool_pose_y, msg.base.tool_pose_z,
            msg.base.tool_pose_theta_x, msg.base.tool_pose_theta_y, msg.base.tool_pose_theta_z,
        ], dtype=np.float32)
        try:
            self.latest_gripper = float(
                msg.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position)
        except (AttributeError, IndexError):
            if not self._warned_gripper:
                rospy.logwarn("No gripper feedback in base_feedback; recording NaN for gripper.")
                self._warned_gripper = True

    def _joy_cb(self, msg):
        pressed = msg.buttons[TOGGLE_BUTTON] if len(msg.buttons) > TOGGLE_BUTTON else 0
        now = time.time()
        if pressed and not self._prev_button and now - self._last_toggle > TOGGLE_DEBOUNCE_S:
            self._last_toggle = now
            self._toggle_event.set()
        self._prev_button = pressed

    def _sync_cb(self, joint_msg, *img_msgs):
        if not self.active:
            return
        t = joint_msg.header.stamp.to_sec()
        if t - self._last_saved_t < self.period * 0.95:  # downsample to --hz
            return
        try:
            images = [self.bridge.imgmsg_to_cv2(m, desired_encoding="rgb8") for m in img_msgs]
        except Exception as e:
            rospy.logerr("image conversion failed: %s", e)
            return
        frame = {
            "t": t,
            "joints": np.array(joint_msg.position[:7], dtype=np.float32),
            "gripper": self.latest_gripper,
            "ee": self.latest_tool.copy(),
            "images": images,
        }
        with self._lock:
            if self.active:
                self.frames.append(frame)
                self._last_saved_t = t

    # ---- episode control -------------------------------------------------
    def wait_for_toggle(self):
        """Blocks until the controller button is pressed (returns False on ROS shutdown)."""
        while not rospy.is_shutdown():
            if self._toggle_event.wait(timeout=0.2):
                self._toggle_event.clear()
                return True
        return False

    def start_episode(self):
        with self._lock:
            self.frames = []
            self._last_saved_t = 0.0
            self.active = True

    def stop_episode(self):
        with self._lock:
            self.active = False
            return self.frames


def save_episode(frames, instruction, out_dir):
    ep_dir = os.path.join(out_dir, f"episode_{int(time.time())}")
    os.makedirs(ep_dir, exist_ok=True)

    with open(os.path.join(ep_dir, "instruction.txt"), "w") as f:
        f.write(instruction)

    np.save(os.path.join(ep_dir, "joint_positions.npy"), np.stack([f["joints"] for f in frames]))
    np.save(os.path.join(ep_dir, "gripper_positions.npy"),
            np.array([f["gripper"] for f in frames], dtype=np.float32))
    np.save(os.path.join(ep_dir, "ee_pose.npy"), np.stack([f["ee"] for f in frames]))
    np.save(os.path.join(ep_dir, "timestamps.npy"), np.array([f["t"] for f in frames], dtype=np.float64))

    for i in range(len(frames[0]["images"])):
        cam_dir = os.path.join(ep_dir, f"cam{i}")
        os.makedirs(cam_dir, exist_ok=True)
        for t, frame in enumerate(frames):
            bgr = cv2.cvtColor(frame["images"][i], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(cam_dir, f"{t:06d}.jpg"), bgr)

    print(f"Saved {len(frames)} steps to {ep_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data")
    parser.add_argument("--image-topics", nargs="+", default=["/cam/color/image_raw"])
    parser.add_argument("--hz", type=float, default=5.0)
    args, _ = parser.parse_known_args()  # ignore rosrun's __name/__log remap args

    os.makedirs(args.out_dir, exist_ok=True)
    recorder = Recorder(args.image_topics, args.hz)

    instruction = ""
    while not rospy.is_shutdown():
        typed = input(f"\nInstruction [{instruction or 'none yet'}] (Enter to reuse, 'quit' to exit): ").strip()
        if typed.lower() == "quit":
            break
        if typed:
            instruction = typed
        if not instruction:
            print("Need an instruction before recording the first episode.")
            continue

        print(f"Press button {TOGGLE_BUTTON} on the controller to START recording ...")
        if not recorder.wait_for_toggle():
            break
        recorder.start_episode()
        print(f"Recording ... press button {TOGGLE_BUTTON} again to STOP this episode.")
        if not recorder.wait_for_toggle():
            break
        frames = recorder.stop_episode()

        if len(frames) < 2:
            print("Episode too short (need at least 2 steps to form an action), discarding.")
            continue
        save_episode(frames, instruction, args.out_dir)


if __name__ == "__main__":
    main()