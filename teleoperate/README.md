xbox controller button map:

impl in `kinova/catkin_ws/src/robot_mainframe/nodes/driver_publisher.py` (`XboxController`).

## buttons always in effect

| Input        | Action                                    |
|--------------|--------------------------------------------|
| A            | Toggle mode (0 <-> 1)   |
| LB           | Open gripper                              |
| RB           | Close gripper                             |
| Start        | Send arm home       |

## mode 0 cartesian velocity (translation)

| Input             | Action                          |
|-------------------|----------------------------------|
| Left stick Y      | Translate x                     |
| Left stick X      | Translate y                     |
| LT / RT           | Translate z (RT - LT)           |
| Right stick, Y/B  | No effect (rotation is zeroed)  |

## mode 1 robot orientation (rotation)

| Input             | Action                          |
|-------------------|----------------------------------|
| Right stick Y     | Rotate about x                  |
| Right stick X     | Rotate about y                  |
| Y / B             | Roll (rotate about z): Y = +, B = -  |
| Left stick, LT/RT | No effect (translation is zeroed) |

Unused: X, Back, stick-click buttons.
