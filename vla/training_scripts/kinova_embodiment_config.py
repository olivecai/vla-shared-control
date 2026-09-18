# Kinova Gen3 embodiment/modality config for NVIDIA Isaac GR00T fine-tuning
# (https://github.com/NVIDIA/Isaac-GR00T). Mirrors the structure of Isaac-GR00T's own
# examples/SO100/so100_config.py, the reference pattern for registering a custom
# ("new embodiment") robot's modality config.
#
# The modality_keys below MUST match the keys emitted in meta/modality.json by
# vla/offline_scripts/convert_to_lerobot_groot.py:
#   state/action: "arm" (7 joints, indices 0:7) + "gripper" (1 value, index 7:8)
#   video:        "cam0" by default -- add "cam1", "cam2", ... here if you record.py
#                 with more than one --cam-ids (each becomes its own video key)
#   annotation:   "human.task_description" (record.py's instruction text)
#
# Usage: passed to launch_finetune.py via --modality-config-path (see train_groot.py);
# it dynamically imports this file, which registers the config as a side effect of the
# module-level register_modality_config() call below.
from gr00t.configs.data.embodiment_configs import register_modality_config
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.data.types import (
    ActionConfig,
    ActionFormat,
    ActionRepresentation,
    ActionType,
    ModalityConfig,
)

kinova_config = {
    # Video: current frame only. Add more camera keys here if you record multiple cams.
    "video": ModalityConfig(
        delta_indices=[0],
        modality_keys=["cam0"],
    ),
    # State: current proprioceptive reading (7 joint angles + gripper position).
    "state": ModalityConfig(
        delta_indices=[0],
        modality_keys=["arm", "gripper"],
    ),
    # Action: 16-step prediction horizon, one ActionConfig per modality key, same order
    # as modality_keys above.
    "action": ModalityConfig(
        delta_indices=list(range(0, 16)),
        modality_keys=["arm", "gripper"],
        action_configs=[
            # arm: RELATIVE -- GR00T computes the delta from the current state itself
            # (better generalization than absolute joint targets), matching the SO100
            # reference config. convert_to_lerobot_groot.py stores the raw next observed
            # state as `action`, which is exactly what GR00T expects to compute this from.
            ActionConfig(
                rep=ActionRepresentation.RELATIVE,
                type=ActionType.NON_EEF,  # joint-space, not end-effector
                format=ActionFormat.DEFAULT,
            ),
            # gripper: ABSOLUTE -- an open/close target works better as an absolute
            # position than as a delta.
            ActionConfig(
                rep=ActionRepresentation.ABSOLUTE,
                type=ActionType.NON_EEF,
                format=ActionFormat.DEFAULT,
            ),
        ],
    ),
    # Language: task instruction, from record.py's instruction.txt via the "human.task_description"
    # annotation that convert_to_lerobot_groot.py writes into meta/modality.json.
    "language": ModalityConfig(
        delta_indices=[0],
        modality_keys=["annotation.human.task_description"],
    ),
}

register_modality_config(kinova_config, embodiment_tag=EmbodimentTag.NEW_EMBODIMENT)
