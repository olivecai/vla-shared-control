# vla-shared-control

This repository contains code to train and deploy human control, VLA control, and shared control on the Kinova arm. 

For tractability, there exists four seperate directories:
- vla
- human_control
- shared_control
- kinova

The code in directories 'vla' and 'human_control' output ONLY VLA actions/confidence/metadata and human control actions/confidence/metadata (aka teleop) respectively.

The code in directory 'shared_control' takes input as vla output and human_control output, and outputs the shared control actions/confidence/metadata.

The code in 'kinova' reads from the ROS control output nodes and outputs real robot actions.

This structure ensures that computations and dataflow is not too messy!!!

