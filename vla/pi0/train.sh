pip install "lerobot[pi]@git+https://github.com/huggingface/lerobot.git"

# preview
python3 train_lerobot_pi0.py /path/to/lerobot_dataset --dry-run

# actual
python3 train_lerobot_pi0.py /path/to/lerobot_dataset \
    --repo-id my_kinova_dataset \
    --exp-name my_run \
    --steps 30000 \
    --batch-size 8 \
    --train-expert-only