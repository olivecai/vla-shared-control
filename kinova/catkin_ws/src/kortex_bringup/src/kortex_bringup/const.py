DELTA_JOINT_POSITION_DEG = 10
KINOVA_DOF = 7
DEFAULT_LOG_PATH = "/tmp/kinova_position_log.txt"

# While the undo button is held, pop one position off position_log.py's stack at most once
# every this many seconds (repeats for as long as it's held).
UNDO_RATE = 1.0