"""Prepare collection of documents to fine-tune dense retrieval model.

See:
* https://github.com/microsoft/unilm/blob/b42b637ac77f6c043e262290df956ae6286de9cb/simlm/misc/prepare_msmarco_data.py
"""

import json
import gzip
from argparse import ArgumentParser
from pathlib import Path

from utils import set_logger


logger = set_logger()


def main():
    # args
    parser = ArgumentParser()
    parser.add_argument("--corpus_dir",
                        type=str, default="data/eswiki_20240401_corpus", help="Path to jsonl.gz files with collection of docs")
    parser.add_argument("--out_dir",
                        type=str, default="runs/messirve_training", help="Output dir to store processed corpus and mapping of old to new docids")
    args = parser.parse_args()
    corpus_dir = args.corpus_dir
    out_dir = args.out_dir

    # main
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    out_file = f'{out_dir}/passages.jsonl.gz'

    logger.info(f"Reading corpus from {corpus_dir} and saving to {out_dir}")
    corpus_docids = save_corpus(corpus_dir, out_file)

    logger.info(f"Saving dict from old 2 docids to {out_dir}/docid_map.json")
    with open(f'{out_dir}/docid_map.json', 'w') as f:
        json.dump(corpus_docids, f)

    logger.info("Done!")


def save_corpus(input_dir: str, output_file: str) -> dict:
    """Converts docids to stringed ints (required by training script),
    saves docs with correct format and returns a mapping from old docids to new docids.
    """
    old2newids = {}
    input_files = sorted(list(Path(input_dir).glob('docs-*.jsonl.gz')))
    n_done = 0
    with gzip.open(output_file, 'wt', encoding='utf-8') as out_f:
        for path in input_files:
            with gzip.open(path) as in_f:
                for i, line in enumerate(in_f):
                    if i % 100_000 == 0:
                        logger.info(f"Processed {n_done} docs")
                    data = json.loads(line)
                    docid = data['docid'] # are like "106#1"
                    new_docid = str(n_done)
                    data_to_write = {'id': new_docid, 'contents': data['text'], 'title': data['title']}
                    old2newids[docid] = new_docid
                    out_f.write(json.dumps(data_to_write, ensure_ascii=False) + '\n')
                    n_done += 1
    return old2newids


if __name__ == "__main__":
    main()
