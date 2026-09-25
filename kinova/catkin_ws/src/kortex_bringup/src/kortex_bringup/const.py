DELTA_JOINT_POSITION_DEG = 3
KINOVA_DOF = 7
DEFAULT_LOG_PATH = "/tmp/kinova_position_log.txt"

# Undo (hold button 2): control_robot.py drives the arm back through position_log.py's stack with
# joint-speed commands for exactly as long as the button is held, and stops on release.
UNDO_JOINT_SPEED_DEG_S = 15.0  # speed toward each logged position, deg/s (max-norm across joints)
UNDO_HOP_TIMEOUT_S = 2.0       # give up waiting on one hop's ACTION_END after this long
UNDO_TOLERANCE_DEG = 0.5      # a logged position counts as reached (and is popped) within this
# Latched Bool published by control_robot.py so position_log.py doesn't log the undo motion itself.
UNDO_ACTIVE_TOPIC = "/my_gen3/undo_active"

# VLA inference: vla/inference.py publishes an absolute target (JointState: position[0:7] = joints in
# radians, position[7] = gripper 0-100, same units as record_vla.py's data). control_robot.py follows
# it with joint-speed commands ONLY while the deadman button is held, and stops the moment it's released.
INFERENCE_TARGET_TOPIC = "/my_gen3/inference/target"
INFERENCE_ENABLE_BUTTON = 9              # hold (left-stick click) to let the model move the arm
INFERENCE_GAIN = 3.0                     # commanded deg/s per degree of error
INFERENCE_MAX_JOINT_SPEED_DEG_S = 10.0   # per-joint speed cap (direction preserved)
INFERENCE_MAX_STEP_DEG = 20.0            # a target farther than this from the arm is treated as a glitch and ignored
INFERENCE_DEADBAND_DEG = 0.2             # closer than this counts as arrived
INFERENCE_GRIPPER_DEADBAND = 3.0         # re-command the gripper only when its target moves by more than this (percent)
INFERENCE_TARGET_TIMEOUT_S = 1.0         # no fresh target for this long -> hold still
