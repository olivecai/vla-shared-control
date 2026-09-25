#!/usr/bin/env python3
"""
Pull a dataset or trained checkpoint from the Hugging Face Hub down to a local directory.

Requires being logged in first: `huggingface-cli login`, or set the HF_TOKEN env var
(only needed for private repos).

Usage:
    # Pull a dataset repo into vla/data
    python3 pull_from_hub.py dataset --repo-id <user-or-org>/kinova-teleop --local-dir vla/data

    # Pull a model repo into vla/checkpoints/kinova-lora
    python3 pull_from_hub.py model --repo-id <user-or-org>/kinova-openvla-lora --local-dir vla/checkpoints/kinova-lora

    # Pull a specific revision (branch, tag, or commit sha)
    python3 pull_from_hub.py dataset --repo-id <user-or-org>/kinova-teleop --local-dir vla/data --revision main
"""
import argparse

from huggingface_hub import snapshot_download


def pull(repo_id, local_dir, repo_type, revision, token):
    print(f"Downloading {repo_id} ({repo_type}) -> {local_dir} ...")
    path = snapshot_download(
        repo_id=repo_id,
        repo_type=repo_type,
        revision=revision,
        local_dir=local_dir,
        token=token,
    )
    print(f"Done: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    subparsers = parser.add_subparsers(dest="kind", required=True)

    dataset_parser = subparsers.add_parser("dataset", help="Pull a dataset repo into a local directory (e.g. vla/data)")
    dataset_parser.add_argument("--local-dir", default="vla/data")

    model_parser = subparsers.add_parser("model", help="Pull a model/checkpoint repo into a local directory (e.g. vla/checkpoints/kinova-lora)")
    model_parser.add_argument("--local-dir", default="vla/checkpoints/kinova-lora")

    for p in (dataset_parser, model_parser):
        p.add_argument("--repo-id", required=True, help="e.g. your-username/kinova-teleop")
        p.add_argument("--revision", default=None, help="Branch, tag, or commit sha to pull (defaults to the repo's default branch)")
        p.add_argument("--token", default=None, help="HF token; defaults to HF_TOKEN env var or `huggingface-cli login` cache")

    args = parser.parse_args()
    repo_type = "dataset" if args.kind == "dataset" else "model"
    pull(args.repo_id, args.local_dir, repo_type, args.revision, args.token)


if __name__ == "__main__":
    main()
