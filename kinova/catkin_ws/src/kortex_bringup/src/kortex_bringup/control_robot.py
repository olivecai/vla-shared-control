#!/usr/bin/python3

import rospy
import numpy as np
from kortex_driver.srv import *
from kortex_driver.msg import *
from sensor_msgs.msg import Joy, JointState
from control_utils.ik_utils import png_control, cartesian_control, joint_control, xbox_control
from control_utils.kinova_gen3 import RGBDVision, JOINT_LIMIT

from std_msgs.msg import Int32MultiArray, Int16, Bool


from const import *
from position_log import peek_last_position, pop_last_position

class CustomCommand():
    def __init__(self, ax, mode, trans_gain, rot_gain, wrist_gain):
        self.ax = ax
        self.mode = mode
        self.trans_gain = trans_gain
        self.rot_gain = rot_gain
        self.wrist_gain = wrist_gain

def gen_iris(base):
    class IrisRecord(base):
        def __init__(self):
            super(IrisRecord, self).__init__(None)
            self.mode = 0 # modes for control
            self.automatic = 0 # mode for whether it approaches automatically
            self.prev_button_2 = 0 # prev button 2 to prevent double clicks (joystick mode only)
            # Same param name/default as position_log.py's ~log_path, so both nodes agree on
            # the stack file location unless overridden identically on both.
            self.undo_log_path = rospy.get_param('~log_path', DEFAULT_LOG_PATH)
            self.undo_held = False # button 2 currently down (set by joy_callback, acted on in step)
            self.undoing = False # an undo is in progress (arm being driven back through the stack)
            self.undo_target = None # row peeked off the stack that the arm is currently heading to
            self.undo_active_pub = rospy.Publisher(UNDO_ACTIVE_TOPIC, Bool, queue_size=1, latch=True)
            self.undo_active_pub.publish(Bool(False))
            self.prev_gripper_cmd = 0.0 # prev gripper cmd
            self.gripper_cmd = 0.0 # gripper cmd
            self.axes_vector = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0] # joystick cmd
            # self.home_array = np.array([30, 14, 148, -80, 57, 3, -140]) # home array in deg
            self.home_array = np.array([11, 345, 170, 219, 5, 320, 80]) # home array in deg
            self.send_joint_angles(self.home_array) #sends robot home
            self.window_center = (424, 240)
            self.grip_center = None
            self.stage = 1

            self.custom_commands = []
            print(f"Current mode = {self.mode}\a", end="\r")

            self.tool_sub = rospy.Subscriber("/my_gen3/base_feedback", BaseCyclic_Feedback, self.tool_callback)
            self.joy_sub = rospy.Subscriber("/joy", Joy, self.joy_callback)

            self.inference_held = False # deadman button down (set by joy_callback, acted on in step)
            self.inferring = False # currently following model targets
            self.inference_target = None # latest [7 joints rad, gripper 0-100] from vla/inference.py
            self.inference_target_time = None
            self.last_inference_gripper = None
            self.inference_sub = rospy.Subscriber(INFERENCE_TARGET_TOPIC, JointState, self.inference_target_callback, queue_size=1)

            self.stage_pub = rospy.Publisher("/my_gen3/inference/stage", Int16, queue_size=10)

        def mode_switch(self):
            self.mode = (self.mode + 1) % 2
            print(f"Current mode = {self.mode}\a", end="\r")
            return

        def tool_callback(self, msg):
            self.tooldata = [msg.base.tool_pose_x, msg.base.tool_pose_y, msg.base.tool_pose_z, msg.base.tool_pose_theta_x, msg.base.tool_pose_theta_y, msg.base.tool_pose_theta_z]
            # print("TOOL DATA", self.tooldata)

        def joy_callback(self, msg):
            self.joy_type = 1 #0 for joystick 1 for xbox controller
            self.buttons = msg.buttons

            # check for gripper commands
            if self.joy_type == 0:
                self.axes_vector = msg.axes
                MAXV_GR = 0.3

                if msg.buttons[0]: # trigger button - close gripper
                    self.gripper_cmd = -1 * MAXV_GR
                elif msg.buttons[1]: # button by thumb - open gripper
                    self.gripper_cmd = MAXV_GR
                else: # both buttons 0 and 1 are zero
                    self.gripper_cmd = 0.0

                if msg.buttons[2] == 0 and self.prev_button_2 == 1:
                    self.mode_switch()
                    
                self.prev_button_2 = msg.buttons[2]
                    
                if msg.buttons[3]:
                    self.automatic = not self.automatic

                if msg.buttons[4]:
                    pass

                if msg.buttons[5]:
                    pass

                if msg.buttons[6]:
                    pass

                if msg.buttons[7]:
                    pass
                
                if msg.buttons[8]:
                    # button 8 pressed, send robot home
                    self.run = False
                    self.send_joint_speeds_command(np.zeros(7))
                    self.send_joint_angles(self.home_array)
                    rospy.loginfo("Button 8 pressed: sending robot to starting position")
                    self.run = True

                if msg.buttons[9]:
                    pass

                if msg.buttons[10]:
                    pass
                    
                if msg.buttons[11]:
                    pass

            elif self.joy_type == 1:
                MAXV_GR = 0.3
                roll = msg.buttons[1] - msg.buttons[3]
                self.axes_vector = [msg.axes[1], msg.axes[0], (1/(msg.axes[5]+1.1) - 1/(msg.axes[2]+1.1))/10, -msg.axes[4]/2, msg.axes[3], roll]
                self.axes_vector = [self.axes_vector[i]/2 for i in range(len(self.axes_vector))]
                # self.axes_vector = [msg.axes[1], msg.axes[0], msg.axes[2], msg.axes[4], msg.axes[3], msg.axes[5]]

                if msg.buttons[0]:
                    pass

                if msg.buttons[1]:
                    pass

                # Only record the button state here. The motion itself runs from step() so it
                # never blocks this callback (a blocked callback lets /joy messages pile up).
                self.undo_held = bool(msg.buttons[2])
                self.inference_held = len(msg.buttons) > INFERENCE_ENABLE_BUTTON and bool(msg.buttons[INFERENCE_ENABLE_BUTTON])

                if msg.buttons[3]:
                    pass

                if msg.buttons[4]: # LB - open gripper
                    self.gripper_cmd = MAXV_GR

                elif msg.buttons[5]: # RB - close gripper
                    self.gripper_cmd = -1 * MAXV_GR

                else: # both buttons 0 and 1 are zero
                    self.gripper_cmd = 0.0

                if msg.buttons[6]:
                    # Start button pressed, send robot home
                    self.run = False
                    self.send_joint_speeds_command(np.zeros(7))
                    self.send_joint_angles(self.home_array)
                    rospy.loginfo("Start button pressed: sending robot to starting position")
                    self.run = True


                if msg.buttons[7]:
                    pass
                
                if msg.buttons[8]:
                    pass

                if msg.buttons[9]:
                    pass

                if msg.buttons[10]:
                    pass

        def undo_step(self):
            '''
            One control tick of undo, called while button 2 is held: drive the joints toward the
            bottom row of position_log.py's stack with a joint-speed command, and pop that row once
            reached so the next tick heads to the one below it. Never pushes to the stack.
            '''
            if self.position is None:
                return
            if not self.undoing:
                self.undoing = True
                self.undo_active_pub.publish(Bool(True))

            if self.undo_target is None:
                row = peek_last_position(self.undo_log_path)
                if row is None:
                    self.send_joint_speeds_command(np.zeros(KINOVA_DOF))
                    rospy.loginfo_throttle(2, "XboxController: undo held but position log is empty, nothing to undo")
                    return
                self.undo_target = row
                self.send_gripper_position(row[KINOVA_DOF] / 100.0)

            target = self.undo_target[:KINOVA_DOF]
     
            if np.max(np.abs(target - np.degrees(self.position[:KINOVA_DOF]))) >= UNDO_TOLERANCE_DEG:
                self.send_joint_angles(target, wait_timeout=UNDO_HOP_TIMEOUT_S)
            pop_last_position(self.undo_log_path)
            self.undo_target = None


        def inference_target_callback(self, msg):
            if len(msg.position) < KINOVA_DOF + 1:
                rospy.logwarn_throttle(2, f"Inference: target needs {KINOVA_DOF + 1} values, got {len(msg.position)}; ignoring")
                return
            self.inference_target = np.array(msg.position[:KINOVA_DOF + 1], dtype=np.float64)
            self.inference_target_time = rospy.Time.now() # receive time, so clock differences don't matter

        def inference_step(self):
            '''
            One control tick while the deadman button is held: drive the joints toward the model's
            latest absolute target with a capped joint-speed command (non-blocking, refreshed every
            tick), and move the gripper to its target. Anything stale, malformed or far from the
            arm's current pose makes the arm hold still instead.
            '''
            if not self.inferring:
                self.inferring = True
                rospy.loginfo("Inference: deadman held, following model targets")
            if self.position is None:
                return

            zeros = np.zeros(KINOVA_DOF)
            fresh = (self.inference_target is not None and
                     (rospy.Time.now() - self.inference_target_time).to_sec() < INFERENCE_TARGET_TIMEOUT_S)
            if not fresh:
                self.send_joint_speeds_command(zeros)
                rospy.loginfo_throttle(2, "Inference: no fresh target from the model, holding still")
                return

            target = self.inference_target.copy()
            for j in (1, 3, 5): # only the physically limited joints; the others can report angles past +-pi
                target[j] = np.clip(target[j], JOINT_LIMIT[j][0], JOINT_LIMIT[j][1])

            err = np.degrees(target[:KINOVA_DOF] - self.position[:KINOVA_DOF])
            worst = np.max(np.abs(err))
            if worst > INFERENCE_MAX_STEP_DEG:
                self.send_joint_speeds_command(zeros)
                rospy.logwarn_throttle(1, f"Inference: target is {worst:.1f} deg from the arm (> {INFERENCE_MAX_STEP_DEG}), ignoring it")
                return

            if worst < INFERENCE_DEADBAND_DEG:
                vel = zeros
            else:
                vel = INFERENCE_GAIN * err
                peak = np.max(np.abs(vel))
                if peak > INFERENCE_MAX_JOINT_SPEED_DEG_S:
                    vel = vel * (INFERENCE_MAX_JOINT_SPEED_DEG_S / peak)
            self.send_joint_speeds_command(vel)

            grip = float(np.clip(target[KINOVA_DOF], 0.0, 100.0))
            if self.last_inference_gripper is None or abs(grip - self.last_inference_gripper) >= INFERENCE_GRIPPER_DEADBAND:
                self.send_gripper_position(grip / 100.0)
                self.last_inference_gripper = grip

        def inference_end(self):
            self.send_joint_speeds_command(np.zeros(KINOVA_DOF))
            self.inferring = False
            self.last_inference_gripper = None
            self.prev_cv_cmd = None # so teleop's dedupe doesn't swallow its first command after inference
            rospy.loginfo("Inference: deadman released, stopped")

        def undo_end(self):
            self.send_joint_speeds_command(np.zeros(KINOVA_DOF))
            self.undoing = False
            self.undo_target = None
            self.prev_cv_cmd = None # so teleop's dedupe doesn't swallow its first command after undo
            self.undo_active_pub.publish(Bool(False))

        def step(self):
            if self.run:
                # Priority: undo > model (deadman held) > joystick teleop.
                if self.undo_held:
                    if self.inferring:
                        self.inference_end()
                    self.undo_step()
                    return
                if self.undoing:
                    self.undo_end()

                if self.inference_held:
                    self.inference_step()
                    return
                if self.inferring:
                    self.inference_end()

                self.custom_commands = []
                
                # step according to rospy rate
                # super().step(self.axes_vector, self.mode, self.custom_commands)
                super().step(self.axes_vector, self.custom_commands)
                if self.gripper_cmd != self.prev_gripper_cmd:
                    success = self.send_gripper_command(self.gripper_cmd)
                    self.prev_gripper_cmd = self.gripper_cmd

    return IrisRecord()

def main():
    controller = xbox_control # can replace with cartesian_control or joint_control or png_control or xbox_control
    robot = gen_iris(controller)
    
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        robot.step()
        rate.sleep()


if __name__ == "__main__":
    try:
        rospy.init_node('iris_control', anonymous=True)
        main()
    except rospy.ROSInterruptException:
        print("ROSInterruptException")


