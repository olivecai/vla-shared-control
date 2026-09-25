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
