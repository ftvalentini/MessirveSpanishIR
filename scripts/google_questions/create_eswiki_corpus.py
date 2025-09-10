"""Read extracted wikipedia JSON files and create a MIRACL-like corpus.
We follow MIRACL's corpus structure:
 
* https://github.com/project-miracl/miracl?tab=readme-ov-file#-corpora

Each article consists of a dict like:
  {"id": "", "revid": "", "url": "", "title": "", "text": "..."}
For each article, we need multiple dicts like:
  [ 
      {"docid": "5163778#0", "title": "Yaxcopoil (Yaxkukul)", "text": "Yaxcopoil, es una localidad del estado de Yuca."},
      {"docid": "5163778#1", "title": "Yaxcopoil (Yaxkukul)", "text": "Yaxcopoil es un toponímico que en idioma maya .."} 
  ]
We need to make sure that
 * passages don't start with title
 * passages don't have newlines
 * split passages by paragraph
 * Use article "id" to build the docid
 * Use article "title" to build the title

NOTE we remove chunks with 3 words or less (this is arbitrary)
"""

import argparse
import gzip
import html
import json
import re
from pathlib import Path

from tqdm import tqdm

from helpers import set_logger


logger = set_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Path to extracted Wikipedia JSON files")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="Output dir to store processed jsonl files")
    parser.add_argument("--cirrus", action="store_true",
                        help="Use cirrussearch input format")
    parser.add_argument("--max_docs_per_file", type=int, default=500_000, 
                        help="Max number of docs per jsonl file")
    args = parser.parse_args()

    # Iterate over files in input_dir of the form input_dir/*/wiki_*:
    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    max_docs_per_file = args.max_docs_per_file

    input_files = list(input_dir.glob("*/wiki_*"))
    logger.info(f"Found {len(input_files)} files in {input_dir}")

    total_size = 0
    for input_file in input_files:
        total_size += Path(input_file).stat().st_size
    logger.info(f"Total size of input files: {total_size / 1e9:.2f} GB")

    process_extracted_dump(output_dir, max_docs_per_file, input_files)


def process_extracted_dump(output_dir, max_docs_per_file, input_files):
    file_index = 0
    docs = []
    for input_file in tqdm(input_files, desc="Processing files", unit="file"):
        n_lines = count_lines(input_file)
        with open(input_file, encoding='utf-8') as f:
            for line in tqdm(f, total=n_lines, desc="Processing articles", unit="article"):
                article = json.loads(line)
                article_id = article["id"]
                article_title = article["title"]
                article_text = article["text"]
                # If article title starts with "Portal:" or "Anexo:", insert space after the prefix.
                # This is important because docs will be represented as title + doc, 
                # so title = "Portal: Mexico" will be tokenized better than "Portal:Mexico"
                allowed_namespace_prefixes = ["Portal:", "Anexo:"]
                if article_title.startswith(tuple(allowed_namespace_prefixes)):
                    article_title = article_title.replace(":", ": ", 1)
                # Split article into passages:
                passages = article_text.split("\n")
                # doc: dict with docid, title and text:
                passage_index = 0
                for passage in passages:
                    title = article_title
                    text = passage
                    # clean passage and title:
                    text = clean_passage(text)
                    title = html.unescape(title).strip()
                    # skip chunks with 3 words or less
                    if len(text.split()) <= 3:
                        continue
                    docid = f"{article_id}#{passage_index}"
                    passage_index += 1
                    docs.append({"docid": docid, "title": title, "text": text})
                    # Write to jsonl.gz file if we have enough docs:
                    if len(docs) >= max_docs_per_file:
                        output_file = output_dir / f"docs-{file_index:02d}.jsonl.gz"
                        with gzip.open(output_file, "wt", encoding="utf-8") as f:
                            for doc in docs:
                                f.write(json.dumps(doc, ensure_ascii=False) + "\n")
                        file_index += 1
                        docs = []

    # Write the remaining docs to the output file if there are any:
    if docs:
        output_file = output_dir / f"docs-{file_index:02d}.jsonl.gz"
        with gzip.open(output_file, "wt", encoding="utf-8") as f:
            for doc in docs:
                f.write(json.dumps(doc, ensure_ascii=False) + "\n")


def clean_passage(passage: str) -> str:
    # remove all "#REDIRECCIÓN" from the text:
    passage = passage.replace("#REDIRECCIÓN", " ")
    # replace all multiple whitespace with a single space:
    passage = re.sub("\s+", " ", passage)
    # # TODO we should have used here:
    # # Replace special html characters e.g. "&amp;" --> "&"
    # passage = html.unescape(passage)
    # remove leading and trailing whitespace:
    passage = passage.strip()
    return passage


def count_lines(file: str) -> int:
    with open(file) as f:
        n_lines = sum(1 for _ in f)
    return n_lines


if __name__ == "__main__":
    main()
