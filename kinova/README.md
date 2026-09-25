# Set up Kinova Docker and catkin packages...

1. clone repository https://github.com/olivecai/vla-shared-control if you haven't already: `git clone git@github.com:olivecai/vla-shared-control.git`

2. cd into the repository vla-shared-control

3. `cd kinova/ ; docker compose up -d --build`

4. `docker exec -it vla_shared_control bash` to enter the Docker container.

Now you should be in the container; 
> example output:
> root@oliveoil-ubuntu:/home/user/kinova# 


6. In the container, run the following. It will take a few minutes:
```
echo $ROS_DISTRO          # should print noetic 
rosdep update             #
cd /home/user/kinova/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

Now if you would like to run the actual robot and ROS nodes:

7. Turn on the Kinova by pressing the ON button on the base until the blue LED turns on. Let go of the button. The blue LED will turn off and after ~30 seconds, the blue LED and a yellow LED will turn on. If the yellow LED turns green, the Kinova is ready; if else it turns red, turn off the Kinova by holding the power button, readjust its position, and try again. Ensure the RJ45 ethernet LED is green.

8. Ensure you are connected to the RobotVision wifi, otherwise the connection to the robot will not work.

9. Launch the robot driver `roslaunch kortex_bringup kortex_bringup.launch ip_address:=192.168.1.127`
> example output:
    [INFO] [1787172238.776892559]: State changed from INITIALIZING to IDLE

    [INFO] [1787172238.782742997]: -------------------------------------------------
    [INFO] [1787172238.782775029]: Initializing Kortex Driver's services...
    [INFO] [1787172240.872538225]: Kortex Driver's services initialized correctly.
    [INFO] [1787172240.872582046]: -------------------------------------------------
    [INFO] [1787172241.074417166]: The Kortex driver has been initialized correctly!

9. Find out the camera serial number:

Run `roslaunch realsense2_camera rs_camera.launch camera:=camera0`. It will error. Then read the `Device with serial number` line in the log. Go into `start_robot_session.sh` and edit the SERIAL_NO at the top of the file. For instance, the line is currently `SERIAL_NO="${SERIAL_NO:-017322072808}"`

10. In a seperate container terminal: run your script using rosrun `rosrun kinova_basic_tests test1.py` (see the vla/record.py script for )
