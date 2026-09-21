# kortex_teleop (extracted from kinova-diffusion)

This is a trimmed copy of the files needed to joystick-teleoperate the Kinova
Gen3 arm, pulled out of the `kinova-diffusion` repo. No source file content
was modified — everything here is a byte-for-byte copy of the original.

## Layout

```
catkin_ws/
  CMakeLists.txt                      # copy of catkin's generated toplevel.cmake
  src/
    CMakeLists.txt                    # same file, also present at src/ (symlink target in original ws)
    kortex_bringup/                   # trimmed package: only the arm-control path
      CMakeLists.txt
      package.xml
      setup.py
      msg/Float32MultiArrayStamped.msg   # kept only so CMakeLists.txt's
                                          # add_message_files() still works unedited;
                                          # not used by record_main.py
      launch/kortex_bringup.launch    # starts the kortex_driver node (talks to the arm)
      src/kortex_bringup/
        __init__.py
        kinova_gen3.py                # only kept because __init__.py imports it;
                                       # NOT used by the teleop control path itself
        record_main.py                # *** the actual teleop controller ***
        control_utils/
          kinova_gen3.py              # KinovaGen3 + RGBDVision — real motion commands
          ik_utils.py                 # xbox_control / cartesian_control / joint_control /
                                       # png_control — joystick axes -> robot commands
    ros_kortex/                       # Kinova's vendor driver, copied whole (unedited)
      kortex_driver/                  # kortex_arm_driver node + kortex_driver.msg/.srv
      kortex_description/             # URDF/xacro + joint_limits.yaml
      kortex_api/                     # empty placeholder — vendor SDK is fetched by
                                       # kortex_driver/conanfile.py during the build
```

## What was deliberately left out

- `record.py`, `inference.py`, `inference_main.py`, `segment.py`,
  `pointcloud_processing.py`, `kinova_util.py` — episode recording / diffusion
  inference / segmentation, not arm control.
- `camera_node/`, `realsense-ros/` — camera pipeline.
- `kortex_control`, `kortex_gazebo`, `kortex_move_it_config`, `kortex_examples`,
  `ros_kortex/third_party/` — simulation / MoveIt / example code, not needed to
  drive the physical arm with a joystick.
- The root-level `src/kortex_bringup/ik_utils.py` — dead code, not imported
  anywhere in the original repo.

## External dependencies (not part of this repo, install separately)

- ROS Noetic + this package built in a catkin workspace
- `ros-noetic-joy` (provides `joy_node`, publishes `/joy`)
- `ros-noetic-joint-state-publisher`, `ros-noetic-robot-state-publisher`
- `xacro`
- Conan (for `kortex_driver`'s vendor SDK fetch — see `kortex_driver/conanfile.py`)

## Running it

```bash
# terminal 1
roslaunch kortex_bringup kortex_bringup.launch ip_address:=<arm ip>

# terminal 2
rosrun joy joy_node

# terminal 3
rosrun kortex_bringup record_main.py
```
