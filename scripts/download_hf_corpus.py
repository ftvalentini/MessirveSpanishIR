"""Download corpus (collection of docs) from Hugging Face Datasets and save 
it as jsonl.gz files to be used with pyserini.
"""

import argparse
import gzip
import json
import os
from pathlib import Path

from datasets import load_dataset

from utils import set_logger


logger = set_logger()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name in HF Hub")
    parser.add_argument("--outdir", type=str, required=True, help="Output dir to store jsonl.gz files")
    parser.add_argument("--max_docs_per_file", type=int, default=500_000, help="Max number of docs per jsonl file")
    args = parser.parse_args()

    logger.info(f"Downloading dataset: {args.dataset}")
    hf_access_token = os.getenv("HF_READ_TOKEN") # None if not set
    dataset = load_dataset(args.dataset, cache_dir=None, token=hf_access_token)
    n_docs = len(dataset["corpus"])
    logger.info(f"Loaded {n_docs} docs")

    logger.info(f"Writing jsonl.gz files to {args.outdir}")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    batch_size = args.max_docs_per_file
    end_start_indices = [(i, i + batch_size) for i in range(0, n_docs, batch_size)]

    for file_index, (start, end) in enumerate(end_start_indices):
        end_ = min(end, n_docs)
        docs = dataset["corpus"].select(range(start, end_))
        write_batch_to_jsonl_gz(docs, outdir, file_index)
        logger.info(f"Saved docs {start} to {end_} ({len(docs)} docs)")

    logger.info("Done!")


def write_batch_to_jsonl_gz(docs, outdir, file_index):
    out_file = outdir / f"docs-{file_index:02d}.jsonl.gz"
    with gzip.open(out_file, "wt", encoding="utf-8") as f_out:
        for doc in docs:
            f_out.write(json.dumps(doc, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
