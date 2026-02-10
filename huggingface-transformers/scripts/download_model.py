#!/usr/bin/env python3
"""Download a HuggingFace model or dataset to a local directory.

Handles authentication, revision selection, and file filtering.
Useful as a pre-download step in init containers or build stages.

Usage:
    python download_model.py meta-llama/Llama-3.1-8B-Instruct
    python download_model.py meta-llama/Llama-3.1-8B-Instruct --local-dir /models/llama-8b
    python download_model.py meta-llama/Llama-3.1-8B-Instruct --exclude "*.bin" --revision main
    python download_model.py --dataset tatsu-lab/alpaca --local-dir /data/alpaca

Requires: huggingface_hub, hf_xet (optional, for faster downloads)
"""

import argparse
import os
import sys

from huggingface_hub import snapshot_download


def main():
    parser = argparse.ArgumentParser(description="Download HuggingFace model/dataset")
    parser.add_argument("repo_id", nargs="?", help="Model or dataset repo ID")
    parser.add_argument("--dataset", help="Download a dataset instead of a model")
    parser.add_argument("--local-dir", help="Local download directory")
    parser.add_argument("--revision", default="main", help="Branch/tag/commit")
    parser.add_argument(
        "--exclude",
        nargs="*",
        help="Glob patterns to exclude (e.g., '*.bin' '*.ot')",
    )
    parser.add_argument(
        "--include",
        nargs="*",
        help="Glob patterns to include (e.g., '*.safetensors')",
    )
    parser.add_argument(
        "--token",
        default=os.environ.get("HF_TOKEN"),
        help="HF token (default: $HF_TOKEN)",
    )
    args = parser.parse_args()

    repo_id = args.dataset or args.repo_id
    if not repo_id:
        parser.error("Provide a repo_id or --dataset")

    repo_type = "dataset" if args.dataset else "model"
    local_dir = args.local_dir or os.path.join(
        os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface")),
        repo_type + "s",
        repo_id.replace("/", "--"),
    )

    print(f"Downloading {repo_type}: {repo_id}")
    print(f"  Revision: {args.revision}")
    print(f"  Destination: {local_dir}")
    if args.include:
        print(f"  Include: {args.include}")
    if args.exclude:
        print(f"  Exclude: {args.exclude}")
    print()

    try:
        path = snapshot_download(
            repo_id=repo_id,
            repo_type=repo_type,
            revision=args.revision,
            local_dir=local_dir,
            allow_patterns=args.include,
            ignore_patterns=args.exclude,
            token=args.token,
        )
        print(f"\nDownloaded to: {path}")

        # Print size
        total_size = 0
        for dirpath, _, filenames in os.walk(path):
            for f in filenames:
                fp = os.path.join(dirpath, f)
                if not os.path.islink(fp):
                    total_size += os.path.getsize(fp)
        print(f"Total size: {total_size / 1e9:.2f} GB")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
