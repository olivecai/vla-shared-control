# Set up Kinova Docker and catkin packages...

1. clone repository https://github.com/olivecai/vla-shared-control if you haven't already: `git clone git@github.com:olivecai/vla-shared-control.git`

2. cd into the repository vla-shared-control

3. `git submodule init; git submodule update`

3. `cd kinova`

4. `docker exec -it vla_shared_control bash` to enter the Docker container.

Now you should be in the container; 
> example output:
> root@oliveoil-ubuntu:/home/user/kinova# 


6. In the container, run the following. It will take a few minutes:
```
cd /home/user/kinova/catkin_ws/src
git clone -b noetic-devel https://github.com/Kinovarobotics/ros_kortex.git
git clone -b master https://github.com/Kinovarobotics/ros_kortex_vision.git
git clone https://github.com/cjiang2/kortex_bringup.git
cd /home/user/kinova/catkin_ws
rosdep install --from-paths src --ignore-src -r -y
catkin_make
source devel/setup.bash
```

Now if you would like to run the actual robot and ROS nodes:
7. Turn on the Kinova.
8. Launch the robot driver `roslaunch kortex_bringup kortex_bringup.launch ip_address:=192.128.1.127`
> example output:
    [INFO] [1787172238.776892559]: State changed from INITIALIZING to IDLE

    [INFO] [1787172238.782742997]: -------------------------------------------------
    [INFO] [1787172238.782775029]: Initializing Kortex Driver's services...
    [INFO] [1787172240.872538225]: Kortex Driver's services initialized correctly.
    [INFO] [1787172240.872582046]: -------------------------------------------------
    [INFO] [1787172241.074417166]: The Kortex driver has been initialized correctly!

9. In a seperate container terminal: run your script using rosrun `rosrun kinova_basic_tests test1.py` 