#!/usr/bin/env python3
"""
Run a fine-tuned OpenVLA checkpoint (see train.py) in closed loop on the real Kinova arm.

Reads:
    /cameras/cam{id}     (sensor_msgs/Image)      -- scene camera
    /joint_states        (sensor_msgs/JointState) -- current joint positions
    /gripper_state       (std_msgs/Float32)       -- current gripper position (0=open, 100=closed)
                                                      Both are used to turn the model's predicted
                                                      DELTA action into absolute targets.
Writes:
    /driver/joint_state  (sensor_msgs/JointState) -- absolute joint target
    /driver/gripper_state (std_msgs/Float32)      -- absolute gripper target (0=open, 100=closed)
                                                      Both consumed by
                                                      robot_mainframe/nodes/driver_subscriber.py
                                                      (the only node allowed to move the arm).

Usage:
    rosrun robot_mainframe camera_node.py _cam_id:=0    # in another terminal
    rosrun robot_mainframe robot_state_node.py          # in another terminal
    rosrun robot_mainframe driver_subscriber.py               # in another terminal
    python3 inference.py --checkpoint vla/checkpoints/kinova-lora --instruction "pick up the cup"
"""
import argparse

import cv_bridge
import numpy as np
import rospy
import torch
from PIL import Image as PILImage
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Float32
from transformers import AutoModelForVision2Seq, AutoProcessor

ACTION_DIM = 8  # 7 joints + 1 gripper
UNNORM_KEY = "kinova"  # must match --unnorm-key used in train.py


class VLAInferenceNode:
    def __init__(self, checkpoint, instruction, cam_id, rate_hz):
        rospy.init_node("vla_inference", anonymous=True)
        self.instruction = instruction
        self.bridge = cv_bridge.CvBridge()
        self.latest_image = None
        self.latest_joints = None
        self.latest_gripper = None

        rospy.loginfo(f"Loading OpenVLA checkpoint from {checkpoint} ...")
        self.processor = AutoProcessor.from_pretrained(checkpoint, trust_remote_code=True)
        self.model = AutoModelForVision2Seq.from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, low_cpu_mem_usage=True, trust_remote_code=True,
        ).to("cuda").eval()

        rospy.Subscriber(f"/cameras/cam{cam_id}", Image, self._image_cb)
        rospy.Subscriber("/joint_states", JointState, self._joint_cb)
        rospy.Subscriber("/gripper_state", Float32, self._gripper_cb)
        self.joint_pub = rospy.Publisher("/driver/joint_state", JointState, queue_size=1)
        self.gripper_pub = rospy.Publisher("/driver/gripper_state", Float32, queue_size=1)

        rospy.loginfo("Waiting for first camera image, joint state, and gripper state ...")
        rospy.wait_for_message(f"/cameras/cam{cam_id}", Image)
        rospy.wait_for_message("/joint_states", JointState)
        rospy.wait_for_message("/gripper_state", Float32)

        rospy.Timer(rospy.Duration(1.0 / rate_hz), self._step)
        rospy.loginfo(f"Running VLA inference at {rate_hz} Hz with instruction: '{instruction}'")

    def _image_cb(self, msg):
        # camera_node.py already publishes rgb8.
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _joint_cb(self, msg):
        self.latest_joints = np.array(msg.position)

    def _gripper_cb(self, msg):
        self.latest_gripper = msg.data

    def _step(self, event):
        if self.latest_image is None or self.latest_joints is None or self.latest_gripper is None:
            return

        image = PILImage.fromarray(self.latest_image)
        prompt = f"In: What action should the robot take to {self.instruction}?\nOut:"
        inputs = self.processor(prompt, image, return_tensors="pt").to("cuda", dtype=torch.bfloat16)

        with torch.no_grad():
            # predict_action() decodes the model's output tokens back into a continuous action
            # and unnormalizes it using dataset_statistics.json[UNNORM_KEY] saved by train.py.
            delta_action = np.asarray(self.model.predict_action(**inputs, unnorm_key=UNNORM_KEY, do_sample=False))

        joint_target = self.latest_joints[:7] + delta_action[:7]
        gripper_target = float(np.clip(self.latest_gripper + delta_action[7], 0.0, 100.0))

        joint_cmd = JointState()
        joint_cmd.header.stamp = rospy.Time.now()
        joint_cmd.position = joint_target.tolist()
        self.joint_pub.publish(joint_cmd)
        self.gripper_pub.publish(Float32(data=gripper_target))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="vla/checkpoints/kinova-lora")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--cam-id", type=int, default=0)
    parser.add_argument("--rate", type=float, default=5.0)
    args, _ = parser.parse_known_args()  # ignore rosrun's __name/__log remap args

    VLAInferenceNode(args.checkpoint, args.instruction, args.cam_id, args.rate)
    rospy.spin()


if __name__ == "__main__":
    main()
