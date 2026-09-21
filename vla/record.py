#!/usr/bin/env python3
"""
Record teleop demonstrations for OpenVLA fine-tuning (https://github.com/openvla/openvla).

Reads:
    /cameras/cam{id}        (sensor_msgs/Image)
    /joint_states           (sensor_msgs/JointState)
    /gripper_state          (std_msgs/Float32)
    /driver/record_toggle   (std_msgs/Empty) -- xbox controller's Back button starts/stops an episode

Writes one folder per episode under --out-dir:
    episode_<timestamp>/
        instruction.txt         -- language instruction for this episode
        joint_positions.npy     -- (T, 7) float32 array of recorded joint angles
        gripper_positions.npy   -- (T,) float32 array of recorded gripper positions (0=open, 100=closed)
        timestamps.npy          -- (T,) float64 array of ROS timestamps
        cam{id}/000000.jpg ...  -- one RGB frame per timestep, per camera

USAGE::
    run the ./start_robot_session.sh
    # seperate terminal:
    docker exec -it vla_shared_control bash 
    cd ../vla
    python3 record.py --out-dir vla/data --cam-ids 0 --hz 5

    OR


    cd kinova 

    docker compose build
    docker compose up -d 

    
    docker exec -it vla_shared_control bash
    roslaunch kortex_bringup kortex_bringup.launch ip_address:=192.168.1.127

    docker exec -it vla_shared_control bash
    rosrun robot_mainframe camera_node.py _cam_id:=0     # in another terminal

    docker exec -it vla_shared_control bash
    rosrun robot_mainframe robot_state_node.py           # in another terminal

    docker exec -it vla_shared_control bash
    rosrun robot_mainframe driver_subscriber.py          # in another terminal, drives the arm

    docker exec -it vla_shared_control bash
    rosrun joy joy_node _dev:=/dev/input/js0             # in another terminal, reads the xbox controller

    docker exec -it vla_shared_control bash
    rosrun robot_mainframe driver_publisher.py           # in another terminal, joystick -> /driver/*

    #now for the record script

    docker exec -it vla_shared_control bash 
    cd ../vla
    python3 record.py --out-dir vla/data --cam-ids 0 --hz 5
"""
import argparse
import os
import threading
import time

import cv2
import cv_bridge
import numpy as np
import rospy


from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Float32, Empty


class Recorder:
    def __init__(self, cam_ids, hz):
        rospy.init_node("vla_recorder", anonymous=True)
        self.bridge = cv_bridge.CvBridge()
        self.cam_ids = cam_ids
        self.latest_images = {cam_id: None for cam_id in cam_ids}
        self.latest_joints = None
        self.latest_gripper = None
        self.active = False
        self.frames = []  # list of {"t": float, "joints": np.ndarray, "gripper": float, "images": {cam_id: np.ndarray}}
        self._toggle_event = threading.Event()

        for cam_id in cam_ids:
            rospy.Subscriber(f"/cameras/cam{cam_id}", Image, self._image_cb, callback_args=cam_id)
        rospy.Subscriber("/joint_states", JointState, self._joint_cb)
        rospy.Subscriber("/gripper_state", Float32, self._gripper_cb)
        rospy.Subscriber("/driver/record_toggle", Empty, self._toggle_cb)

        rospy.loginfo("Waiting for first camera image(s), joint state, and gripper state ...")
        for cam_id in cam_ids:
            rospy.wait_for_message(f"/cameras/cam{cam_id}", Image)
        rospy.wait_for_message("/joint_states", JointState)
        rospy.wait_for_message("/gripper_state", Float32)

        # Snapshot the latest cached image(s)/joint state at a fixed rate while `active` is set.
        rospy.Timer(rospy.Duration(1.0 / hz), self._timer_cb)

    def _image_cb(self, msg, cam_id):
        frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")
        self.latest_images[cam_id] = frame # frame is already RGB. if later change fmt then need a CvtColor func here

    def _joint_cb(self, msg):
        self.latest_joints = np.array(msg.position, dtype=np.float32)

    def _gripper_cb(self, msg):
        self.latest_gripper = msg.data

    def _toggle_cb(self, msg):
        self._toggle_event.set()

    def wait_for_toggle(self):
        """Blocks until the xbox controller's Back button (start/stop) is pressed."""
        self._toggle_event.wait()
        self._toggle_event.clear()

    def _timer_cb(self, event):
        if not self.active:
            return
        self.frames.append({
            "t": rospy.Time.now().to_sec(),
            "joints": self.latest_joints.copy(),
            "gripper": self.latest_gripper,
            "images": {cam_id: img.copy() for cam_id, img in self.latest_images.items()},
        })

    def record_episode(self):
        self.frames = []
        self.active = True

    def stop_episode(self):
        self.active = False
        return self.frames


def save_episode(frames, instruction, out_dir):
    ep_dir = os.path.join(out_dir, f"episode_{int(time.time())}")
    os.makedirs(ep_dir, exist_ok=True)

    with open(os.path.join(ep_dir, "instruction.txt"), "w") as f:
        f.write(instruction)

    joints = np.stack([f["joints"] for f in frames])
    gripper = np.array([f["gripper"] for f in frames], dtype=np.float32)
    timestamps = np.array([f["t"] for f in frames])
    np.save(os.path.join(ep_dir, "joint_positions.npy"), joints)
    np.save(os.path.join(ep_dir, "gripper_positions.npy"), gripper)
    np.save(os.path.join(ep_dir, "timestamps.npy"), timestamps)

    for cam_id in frames[0]["images"]:
        cam_dir = os.path.join(ep_dir, f"cam{cam_id}")
        os.makedirs(cam_dir, exist_ok=True)
        for t, frame in enumerate(frames):
            bgr = cv2.cvtColor(frame["images"][cam_id], cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(cam_dir, f"{t:06d}.jpg"), bgr)

    print(f"Saved {len(frames)} steps to {ep_dir}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="vla/data")
    parser.add_argument("--cam-ids", type=int, nargs="+", default=[0])
    parser.add_argument("--hz", type=float, default=5.0)
    args, _ = parser.parse_known_args()  # ignore rosrun's __name/__log remap args

    os.makedirs(args.out_dir, exist_ok=True)
    recorder = Recorder(args.cam_ids, args.hz)

    
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

        print("Press Back on the controller to START recording (move the arm once recording starts) ...")
        recorder.wait_for_toggle()
        recorder.record_episode()
        print("Recording ... press Back again to STOP this episode.")
        recorder.wait_for_toggle()
        frames = recorder.stop_episode()

        if len(frames) < 2:
            print("Episode too short (need at least 2 steps to form an action), discarding.")
            continue
        save_episode(frames, instruction, args.out_dir)


if __name__ == "__main__":
    main()
