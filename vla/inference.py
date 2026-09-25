#!/usr/bin/env python3
"""
Run a fine-tuned OpenVLA checkpoint (see train.py) in closed loop on the real Kinova arm.

Reads (same topics/units as kortex_bringup/record_vla.py recorded the training data with):
    /camera0/color/image_raw (sensor_msgs/Image)             -- scene camera (override with --image-topic)
    /my_gen3/joint_states    (sensor_msgs/JointState)        -- current joint positions, radians
    /my_gen3/base_feedback   (kortex_driver/BaseCyclic_Feedback) -- current gripper position (0=open, 100=closed)
                                                      Both state readings are used to turn the model's
                                                      predicted DELTA action into an absolute target.
Writes:
    /my_gen3/inference/target (sensor_msgs/JointState)       -- absolute target: position[0:7] = joints
                                                      (radians), position[7] = gripper (0-100).
                                                      Consumed by kortex_bringup/control_robot.py, which
                                                      moves the arm toward it ONLY while the deadman
                                                      button (left-stick click, INFERENCE_ENABLE_BUTTON in
                                                      const.py) is held, and stops when it is released.

Usage (arm, controller, camera and control_robot.py up first -- ./start_robot_session.sh does all of it):
    python3 vla/inference.py --checkpoint vla/checkpoints/kinova-lora --instruction "pick up the cup"
    # then HOLD the left-stick click on the controller to let the model move the arm.
"""
import argparse

import cv_bridge
import numpy as np
import rospy
import torch
from PIL import Image as PILImage
from kortex_driver.msg import BaseCyclic_Feedback
from sensor_msgs.msg import Image, JointState
from transformers import AutoModelForVision2Seq, AutoProcessor

ACTION_DIM = 8  # 7 joints + 1 gripper
UNNORM_KEY = "kinova"  # must match --unnorm-key used in train.py
JOINT_STATES_TOPIC = "/my_gen3/joint_states"
FEEDBACK_TOPIC = "/my_gen3/base_feedback"
TARGET_TOPIC = "/my_gen3/inference/target"  # keep in sync with INFERENCE_TARGET_TOPIC in kortex_bringup/const.py


class VLAInferenceNode:
    def __init__(self, checkpoint, instruction, image_topic, rate_hz):
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

        rospy.Subscriber(image_topic, Image, self._image_cb)
        rospy.Subscriber(JOINT_STATES_TOPIC, JointState, self._joint_cb)
        rospy.Subscriber(FEEDBACK_TOPIC, BaseCyclic_Feedback, self._feedback_cb)
        self.target_pub = rospy.Publisher(TARGET_TOPIC, JointState, queue_size=1)

        rospy.loginfo("Waiting for first camera image, joint state, and gripper feedback ...")
        rospy.wait_for_message(image_topic, Image)
        rospy.wait_for_message(JOINT_STATES_TOPIC, JointState)
        rospy.wait_for_message(FEEDBACK_TOPIC, BaseCyclic_Feedback)

        rospy.Timer(rospy.Duration(1.0 / rate_hz), self._step)
        rospy.loginfo(f"Running VLA inference at {rate_hz} Hz with instruction: '{instruction}'")

    def _image_cb(self, msg):
        # realsense2_camera publishes rgb8.
        self.latest_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="rgb8")

    def _joint_cb(self, msg):
        self.latest_joints = np.array(msg.position)

    def _feedback_cb(self, msg):
        try:
            self.latest_gripper = float(msg.interconnect.oneof_tool_feedback.gripper_feedback[0].motor[0].position)
        except (AttributeError, IndexError):
            rospy.logwarn_throttle(5, "No gripper feedback in base_feedback; not publishing targets")

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

        target = JointState()
        target.header.stamp = rospy.Time.now()
        target.name = [f"joint_{i + 1}" for i in range(7)] + ["gripper"]
        target.position = joint_target.tolist() + [gripper_target]
        self.target_pub.publish(target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="vla/checkpoints/kinova-lora")
    parser.add_argument("--instruction", required=True)
    parser.add_argument("--image-topic", default="/camera0/color/image_raw")
    parser.add_argument("--rate", type=float, default=5.0)
    args, _ = parser.parse_known_args()  # ignore rosrun's __name/__log remap args

    VLAInferenceNode(args.checkpoint, args.instruction, args.image_topic, args.rate)
    rospy.spin()


if __name__ == "__main__":
    main()
