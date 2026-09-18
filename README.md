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




python3 scripts/convert_unified_to_lerobot_v2.py kinova-diffusion/data/my-robot-dataset <out_dir> --repo-id my-username/my-robot-dataset
 



 SEP 17 

 pip install torch==2.9.0 torchvision==0.24.0
pip install flash-attn==2.8.3 --no-build-isolation
pip install -e .
