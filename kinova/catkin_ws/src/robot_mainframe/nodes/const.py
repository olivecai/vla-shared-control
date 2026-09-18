KINOVA_DOF=7


XBOX_TELEOP = "XBOX_TELEOP"
KEYBOARD_TELEOP = "KEYBOARD_TELEOP"

# Per-axis max cartesian velocity for xbox teleop (m/s for linear, rad/s for angular).
# TX/RZ shld be ok TODO finetune others
MAXV_TX = 0.05
MAXV_TY = 0.05
MAXV_TZ = 0.05
MAXV_RX = 0.5
MAXV_RY = 0.5
MAXV_RZ = 0.5

# Per-joint position limits (radians), index 7 is the gripper (0=open, 1=closed).
JOINT_LIMIT = {
    0: [-3.141592653589793, 3.141592653589793],
    1: [-2.2497294058206907, 2.2497294058206907],
    2: [-3.141592653589793, 3.141592653589793],
    3: [-2.5795966344476193, 2.5795966344476193],
    4: [-3.141592653589793, 3.141592653589793],
    5: [-2.0996310901491784, 2.0996310901491784],
    6: [-3.141592653589793, 3.141592653589793],
    7: [0, 1],
}