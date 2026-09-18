#!/usr/bin/env python3
"""
Push recorded episode data (from record.py) or a trained checkpoint (from train.py) to the
Hugging Face Hub.

Requires being logged in first: `huggingface-cli login`, or set the HF_TOKEN env var.

Usage:
    # Push everything under vla/data to a dataset repo
    python3 push_to_hub.py dataset --local-dir vla/data --repo-id <user-or-org>/kinova-teleop

    # Push a trained checkpoint to a model repo
    python3 push_to_hub.py model --local-dir vla/checkpoints/kinova-lora --repo-id <user-or-org>/kinova-openvla-lora

    # Make either repo private
    python3 push_to_hub.py dataset --local-dir vla/data --repo-id <user-or-org>/kinova-teleop --private
"""
import argparse
import os

from huggingface_hub import HfApi


def push(local_dir, repo_id, repo_type, private, commit_message, token):
    if not os.path.isdir(local_dir):
        raise FileNotFoundError(f"{local_dir} does not exist")

    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type=repo_type, private=private, exist_ok=True)

    print(f"Uploading {local_dir} -> {repo_id} ({repo_type}) ...")
    url = api.upload_folder(
        folder_path=local_dir,
        repo_id=repo_id,
        repo_type=repo_type,
        commit_message=commit_message,
    )
    print(f"Done: {url}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    dataset_parser = subparsers.add_parser("dataset", help="Push a recorded dataset directory (e.g. vla/data)")
    dataset_parser.add_argument("--local-dir", default="vla/data")
    dataset_parser.add_argument("--commit-message", default="Add/update recorded episodes")

    model_parser = subparsers.add_parser("model", help="Push a trained checkpoint directory (e.g. vla/checkpoints/kinova-lora)")
    model_parser.add_argument("--local-dir", default="vla/checkpoints/kinova-lora")
    model_parser.add_argument("--commit-message", default="Add/update trained checkpoint")

    for p in (dataset_parser, model_parser):
        p.add_argument("--repo-id", required=True, help="e.g. your-username/kinova-teleop")
        p.add_argument("--private", action="store_true", help="Create the repo as private if it doesn't exist yet")
        p.add_argument("--token", default=None, help="HF token; defaults to HF_TOKEN env var or `huggingface-cli login` cache")

    args = parser.parse_args()
    repo_type = "dataset" if args.kind == "dataset" else "model"
    push(args.local_dir, args.repo_id, repo_type, args.private, args.commit_message, args.token)


if __name__ == "__main__":
    main()
