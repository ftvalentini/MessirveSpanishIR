"""...
"""

import argparse
import os
import re
from pathlib import Path

from huggingface_hub import HfApi

from utils import set_logger


logger = set_logger()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset repo name in HF Hub")
    parser.add_argument("--outdir", type=str, required=True, help="Output dir to store jsonl.gz files")
    parser.add_argument("--pattern", type=str, required=True, help="Pattern to filter files")
    args = parser.parse_args()

    hf_api = HfApi()

    hf_access_token = os.getenv("HF_READ_TOKEN")
    all_files = hf_api.list_repo_files(
        args.dataset, token=hf_access_token, repo_type="dataset")

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # filter files:
    # pattern = re.compile(args.pattern)
    files = [file for file in all_files if args.pattern in file]
    for file in files:
        if file.endswith(".tsv"):
            logger.info(f"Downloading {file}...")
            hf_api.hf_hub_download(
                repo_id=args.dataset, filename=file, repo_type="dataset",
                local_dir=args.outdir, token=hf_access_token
            )
    
    logger.info("Done!")


if __name__ == "__main__":
    main()
