"""Encode documents from a HF dataset with OpenAI embeddings and save them as txt file. 

Approx space required for 3072-dim embeddings float32, 14M docs:
(32 * 3072 * 14 047 759) * bits = 172.6 GB
"""

import argparse
import os
from pathlib import Path
import random
import time
from typing import List

import numpy as np
import openai
import tiktoken
from tqdm import tqdm
from datasets import load_dataset, Value, Dataset

from utils import set_logger
from helpers import filter_ids


logger = set_logger()

API_KEY = os.getenv("OPENAI_SPANISH_IR_PERSONAL_KEY", None)
ORG_ID = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset (local dir or repo name in HF Hub)")
    parser.add_argument("--subset", type=str, default=None, help="Subset of the dataset to encode, if any")
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset to encode")
    parser.add_argument("--model_name", type=str, required=True, help="OpenAI model name for encoding text")
    parser.add_argument("--outdir", type=str, required=True, help="Output dir to store index files")
    parser.add_argument("--batch_size", type=int, default=2048, help="Batch size for API requests")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix to add to documents")
    parser.add_argument("--fields", type=str, nargs="+", default=["title", "text"], help="Fields to encode (will be concatenated)")
    parser.add_argument("--id_field", type=str, default="docid", help="Field name for document IDs")
    parser.add_argument("--drop_duplicates", action="store_true", help="Drop duplicate documents (by ID)")    
    args = parser.parse_args()

    logger.info(f"Loading dataset: {args.dataset}")
    hf_access_token = os.getenv("HF_READ_TOKEN") # None if not set
    dataset = load_dataset(args.dataset, args.subset, token=hf_access_token)
    dataset = dataset[args.split]

    logger.info(f"Casting ID field to string")
    dataset = dataset.cast_column(args.id_field, Value("string"))

    if args.drop_duplicates:
        logger.info(f"Dropping duplicate documents by ID")
        dataset = dataset.to_pandas().drop_duplicates(subset=args.id_field).reset_index(drop=True)
        dataset = Dataset.from_pandas(dataset)

    # if outdir exists, read ids and remove them from dataset:
    vectors_dir = f"{args.outdir}/vectors"
    ids_dir = f"{args.outdir}/ids"
    last_batch = 0
    if Path(args.outdir).is_dir():
        ids_files = sorted(Path(ids_dir).glob("batch_*.txt"))
        if ids_files:
            encoded_ids = []
            for file in ids_files:
                with open(file, "r") as f:
                    encoded_ids.extend(f.read().splitlines())
            encoded_ids = set(encoded_ids)
            if encoded_ids:
                logger.info(f"Removing {len(encoded_ids)} already encoded documents")
                dataset = filter_ids(dataset, encoded_ids, args.id_field)
        vectors_files = sorted(Path(vectors_dir).glob("batch_*.npy"))
        if vectors_files:
            last_file = vectors_files[-1]
            last_batch = int(last_file.stem.split("_")[-1])
            logger.info(f"Found existing vectors. Starting from batch {last_batch + 1}")
        if ids_files and vectors_files:
            # Check that both dirs have the same number of files:
            assert len(vectors_files) == len(ids_files), "Mismatch between vectors and ids files"
            # Check that the last number is the same in both:
            assert last_batch == int(ids_files[-1].stem.split("_")[-1]), "Mismatch between last batch in vectors and ids"

    print("First dataset example:")
    print(dataset[0])
    print("Last dataset example:")
    print(dataset[-1])

    def prepare_documents(examples, fields: list, model_name: str, prefix=None):
        """Concatenate fields + add prefix, and truncate to max tokens 
        admitted by the model.
        """
        texts = []
        for element in zip(*[examples[field] for field in fields]):
            concatenated = ". ".join(element)
            if prefix:
                concatenated = f"{prefix} {concatenated}"
            texts.append(concatenated)
        texts = [truncate_text_tokens(text, model_name) for text in texts]
        return {"text": texts}
    
    logger.info(f"Preparing documents")
    cols_to_remove = [col for col in dataset.column_names if col not in args.fields + [args.id_field]]
    dataset = dataset.remove_columns(cols_to_remove)
    dataset = dataset.map(
        lambda examples: prepare_documents(
            examples, fields=args.fields, model_name=args.model_name, prefix=args.prefix
        ),
        batched=True, num_proc=8, remove_columns=args.fields
    )

    logger.info(f"Initializing OpenAI client")
    api_client = openai.OpenAI(api_key=API_KEY, organization=ORG_ID)

    logger.info(f"Encoding documents with model: {args.model_name} (batch size: {args.batch_size})")
    Path(vectors_dir).mkdir(parents=True, exist_ok=True)
    Path(ids_dir).mkdir(parents=True, exist_ok=True)
    n_samples = len(dataset)
    pbar = tqdm(total=n_samples, desc="Encoded", unit="docs")
    dataset_iterator = dataset.iter(batch_size=args.batch_size)

    for i, batch in enumerate(dataset_iterator, start=last_batch+1):
        ids = batch[args.id_field]
        embeddings = get_embeddings(api_client, batch["text"], args.model_name)
        # print(embeddings.shape)
        # print(embeddings.dtype)
        vectors_file = f"{args.outdir}/vectors/batch_{i:04d}.npy"
        ids_file = f"{args.outdir}/ids/batch_{i:04d}.txt"
        if not Path(vectors_file).exists():
            np.save(vectors_file, embeddings)
            with open(ids_file, "w") as f:
                f.writelines(id_ + "\n" for id_ in ids)
        else:
            raise FileExistsError(f"File {vectors_file} already exists")            
        pbar.update(len(ids))

    logger.info(f"Done!")


def truncate_text_tokens(text, model_name: str):
    """Truncate a string to have `max_tokens` according to the given encoding.
    Source: https://cookbook.openai.com/examples/embedding_long_inputs
    """
    if model_name in ["text-embedding-3-large", "text-embedding-3-small"]:
        encoding_name = 'cl100k_base'
        max_tokens = 8191
    else:
        raise ValueError(f"Model {model_name} not supported")
    encoding = tiktoken.get_encoding(encoding_name)
    return encoding.encode(text)[:max_tokens]


def retry_with_exponential_backoff(
    func,
    initial_delay: float = 1,
    exponential_base: float = 2,
    jitter: bool = True,
    max_retries: int = 5,
    errors: tuple = (openai.RateLimitError,),
):
    """Retry a function with exponential backoff.
    Source: https://platform.openai.com/docs/guides/rate-limits/error-mitigation
    """
    def wrapper(*args, **kwargs):
        # Initialize variables
        num_retries = 0
        delay = initial_delay
        # Loop until a successful response or max_retries is hit or an exception is raised
        while True:
            try:
                return func(*args, **kwargs)
            # Retry on specific errors
            except errors as e:
                # Increment retries
                num_retries += 1
                # Check if max retries has been reached
                if num_retries > max_retries:
                    raise Exception(
                        f"Maximum number of retries ({max_retries}) exceeded."
                    )
                # Increment the delay
                delay *= exponential_base * (1 + jitter * random.random())
                # Sleep for the delay
                time.sleep(delay)
            # Raise exceptions for any errors not specified
            except Exception as e:
                raise e
    return wrapper


@retry_with_exponential_backoff
def get_embeddings(client, texts: List[str], model_name: str) -> np.ndarray:
    res = client.embeddings.create(input=texts, model=model_name)
    embeddings = [item.embedding for item in res.data]
    return np.array(embeddings, dtype=np.float32)


if __name__ == "__main__":
    main()


# # See:
# # https://huggingface.co/docs/datasets/en/stream#stream
# # https://github.com/castorini/pyserini/blob/b7e1da305dd31b195244d49321087505996260c6/pyserini/encode/__main__.py
# # https://github.com/huggingface/blog/blob/main/hf-bitsandbytes-integration.md
# # https://blog.eleuther.ai/transformer-math/
