# VLA computations

Record teleop demos, LoRA fine-tune [OpenVLA](https://github.com/openvla/openvla) on them, and
run the fine-tuned model in closed loop on the Kinova arm. `train.py`/`inference.py` fine-tune
directly via `transformers` + `peft` (LoRA) on plain per-episode folders

Action space is the 7 Kinova joint angles + 1 gripper position. The model predicts a normalized
DELTA between consecutive recorded steps; inference adds that delta to the current
`/my_gen3/joint_states` + gripper reading (from `/my_gen3/base_feedback`) to get an absolute target,
published on `/my_gen3/inference/target` for `control_robot.py` to follow.

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

1. Bring up the arm, controller, camera and `control_robot.py`:

   ```
   ./start_robot_session.sh
   ```

   (`position_log.py` runs in its own window too, so anything the model does can be undone with
   the X button.) The `record` window can stay idle.

2. Start the model (needs a CUDA GPU, torch/transformers from `vla/requirements.txt`, and the ROS
   workspace sourced so `kortex_driver` messages import -- same environment as `record_vla.py`):

   ```
   python3 vla/inference.py --checkpoint vla/checkpoints/kinova-lora --instruction "pick up the cup"
   ```

   Options: `--image-topic` (default `/camera0/color/image_raw`), `--rate` (Hz, default 5, match
   the `--hz` you recorded with).

3. **Hold the left-stick click (button 9) on the controller to let the model move the arm.** The
   arm follows the model only while the button is held and stops the moment it's released. X
   (undo) takes priority over the model, and joystick teleop is ignored while it's held. Keep a
   hand on the e-stop.

`inference.py` only publishes targets; it never moves the arm itself. `control_robot.py` is what
moves it, and ignores a target that is stale (>1 s old) or more than 20 degrees from the arm's
current pose. Speed and these limits are the `INFERENCE_*` values in
`kinova/catkin_ws/src/kortex_bringup/src/kortex_bringup/const.py`.

inputs:
- vision (`/camera0/color/image_raw`, from `realsense2_camera`)
- robot state (`/my_gen3/joint_states` for joints in radians, `/my_gen3/base_feedback` for the gripper, 0-100)
- language (fixed instruction passed via `--instruction`)

outputs:
- vla_action (`/my_gen3/inference/target`, a `JointState`: 7 joint angles in radians + gripper 0-100)
