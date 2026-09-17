rosrun joy joy_node _dev:=/dev/input/js0     # publishes /joy from xbox
rosrun robot_mainframe driver_subscriber.py  # the only node touching the arm
rosrun robot_mainframe driver_publisher.py   # reads /joy, publishes /driver/*
