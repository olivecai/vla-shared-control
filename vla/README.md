# VLA computations

Record teleop demos, LoRA fine-tune [OpenVLA](https://github.com/openvla/openvla) on them, and
run the fine-tuned model in closed loop on the Kinova arm. `train.py`/`inference.py` fine-tune
directly via `transformers` + `peft` (LoRA) on plain per-episode folders

Action space is the 7 Kinova joint angles + 1 gripper position. The model predicts a normalized
DELTA between consecutive recorded steps; inference adds that delta to the current
`/joint_states` + `/gripper_state` reading to get absolute targets for `/driver/joint_state` and
`/driver/gripper_state`.

## Recording

```
rosrun robot_mainframe camera_node.py _cam_id:=0
rosrun robot_mainframe robot_state_node.py
python3 vla/record.py --out-dir vla/data --cam-ids 0 --hz 5
```

Move the arm by hand (or via teleop) between the "START" and "STOP" prompts; each run is saved
as `vla/data/episode_<timestamp>/` (instruction, per-camera jpgs, joint positions, gripper
positions, timestamps).

## Training

```
pip install -r vla/requirements.txt
python3 vla/train.py --data-dir vla/data --output-dir vla/checkpoints/kinova-lora
```

Needs a CUDA GPU (bf16 LoRA fine-tuning of the 7B model fits on ~24GB). Saves a merged
(LoRA-baked-in) checkpoint plus `dataset_statistics.json` for action unnormalization.

## Inference

```
rosrun robot_mainframe camera_node.py _cam_id:=0
rosrun robot_mainframe robot_state_node.py
rosrun robot_mainframe driver_node.py
python3 vla/inference.py --checkpoint vla/checkpoints/kinova-lora --instruction "pick up the cup"
```

inputs:
- vision (read over ROS node /cameras/cam{id})
- robot state (read over ROS nodes /joint_states and /gripper_state)
- language (fixed instruction passed via `--instruction`)

outputs:
- vla_action (write over ROS nodes to /driver/joint_state and /driver/gripper_state)