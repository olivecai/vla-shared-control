# Set up Kinova Docker and catkin packages...

1. clone repository https://github.com/olivecai/vla-shared-control if you haven't already: `git clone git@github.com:olivecai/vla-shared-control.git`

2. cd into the repository vla-shared-control

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

Then you can launch the ros nodes:
`roslaunch kortex_bringup kortex_bringup.launch`