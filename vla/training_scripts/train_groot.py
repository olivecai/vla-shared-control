#!/usr/bin/env python3
"""
Fine-tune NVIDIA Isaac GR00T (https://github.com/NVIDIA/Isaac-GR00T) on the Kinova
dataset produced by vla/offline_scripts/convert_to_lerobot_groot.py.

This is a thin wrapper around Isaac-GR00T's own finetuning entrypoint
(gr00t/experiment/launch_finetune.py, a tyro.cli(FinetuneConfig) script) -- it does not
reimplement GR00T's training loop, it just supplies Kinova-specific defaults (dataset
path, embodiment tag, modality config) and forwards everything else straight through.

Setup (once) -- Isaac-GR00T must be installed separately, it's not a pip package:
    git clone https://github.com/NVIDIA/Isaac-GR00T.git
    cd Isaac-GR00T && pip install -e .


Usage:
    python3 vla/training_scripts/train_groot.py \
        --dataset-path vla/lerobot_data \
        --output-dir vla/checkpoints/kinova-groot

Any other FinetuneConfig field (learning rate, max steps, wandb, LoRA-equivalent tune_*
flags, etc. -- see Isaac-GR00T's gr00t/configs/finetune_config.py for the full list) can
be set by passing it straight through, since unrecognized flags are forwarded verbatim:
    python3 vla/training_scripts/train_groot.py --dataset-path vla/lerobot_data \
        --output-dir vla/checkpoints/kinova-groot --max-steps 5000 --use-wandb
"""
import argparse
import os
import sys

DEFAULT_BASE_MODEL = "nvidia/GR00T-N1.7-3B"
MODALITY_CONFIG = os.path.join(os.path.dirname(os.path.abspath(__file__)), "kinova_embodiment_config.py")


def find_launch_finetune():
    try:
        import gr00t
    except ImportError as exc:
        raise SystemExit(
            "Isaac-GR00T is not installed. Clone it and pip install -e it:\n"
            "  git clone https://github.com/NVIDIA/Isaac-GR00T.git\n"
            "  cd Isaac-GR00T && pip install -e ."
        ) from exc

    gr00t_pkg_dir = os.path.dirname(os.path.abspath(gr00t.__file__))
    launch_script = os.path.join(gr00t_pkg_dir, "experiment", "launch_finetune.py")
    if not os.path.isfile(launch_script):
        raise SystemExit(
            f"Could not find launch_finetune.py next to the installed gr00t package "
            f"(looked in {launch_script}). Isaac-GR00T's layout may have changed."
        )
    return launch_script


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-path", default="vla/lerobot_data",
                         help="Output of vla/offline_scripts/convert_to_lerobot_groot.py")
    parser.add_argument("--output-dir", default="vla/checkpoints/kinova-groot")
    parser.add_argument("--base-model-path", default=DEFAULT_BASE_MODEL)
    parser.add_argument("--modality-config-path", default=MODALITY_CONFIG)
    parser.add_argument("--num-gpus", type=int, default=1)
    args, extra = parser.parse_known_args()

    launch_script = find_launch_finetune()

    # launch_finetune.py itself (underscore flags, matching Isaac-GR00T's own
    # examples/finetune.sh wrapper -- tyro's CLI also accepts hyphenated forms, but
    # underscores are what the maintainers' own script uses).
    cmd = [
        sys.executable, launch_script,
        "--base_model_path", args.base_model_path,
        "--dataset_path", args.dataset_path,
        "--embodiment_tag", "new_embodiment",
        "--modality_config_path", args.modality_config_path,
        "--num_gpus", str(args.num_gpus),
        "--output_dir", args.output_dir,
    ] + extra

    env = os.environ.copy()
    if args.num_gpus == 1:
        # Single GPU: pin CUDA_VISIBLE_DEVICES so HF Trainer doesn't wrap the model in
        # DataParallel (crashes with a StopIteration in the model's device property) --
        # same guard Isaac-GR00T's own examples/finetune.sh applies.
        env.setdefault("CUDA_VISIBLE_DEVICES", "0")
        print("Running:", " ".join(cmd))
        os.execvpe(cmd[0], cmd, env)
    else:
        cmd = ["torchrun", f"--nproc_per_node={args.num_gpus}"] + cmd[1:]
        print("Running:", " ".join(cmd))
        os.execvpe(cmd[0], cmd, env)


if __name__ == "__main__":
    main()
