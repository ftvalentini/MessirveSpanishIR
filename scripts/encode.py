"""Encode texts from a HF dataset (documents or queries) and save them as 
compressed txt file (wout using Pyserini)

NOTE we dont normalize embeddings here, it's best to save them as they are
and normalize them when saving the index or when running retrieval.
"""

import argparse
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from datasets import load_dataset, Value, Dataset
from transformers import AutoTokenizer, DataCollatorWithPadding

from utils import set_logger, get_unused_gpu_ids
from helpers import TextEncoder, BatchShufflingSampler, filter_ids


logger = set_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True, help="Dataset (local dir or repo name in HF Hub)")
    parser.add_argument("--subset", type=str, default=None, help="Subset of the dataset to encode, if any")
    parser.add_argument("--split", type=str, default="train", help="Split of the dataset to encode")
    parser.add_argument("--outdir", type=str, required=True, help="Output dir to store encoded docs")
    parser.add_argument("--encoder", type=str, required=True, help="HF model name for encoding text")
    parser.add_argument("--pooling", type=str, required=True, help="Pooling strategy for encoding")
    parser.add_argument("--prefix", type=str, default=None, help="Prefix to add to documents before encoding")
    parser.add_argument("--fields", type=str, nargs="+", default=["title", "text"], help="Fields to encode (will be concatenated)")
    parser.add_argument("--id_field", type=str, default="docid", help="Field name for document IDs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for encoding")
    parser.add_argument("--tf32", action="store_true", help="Enable TF32 for CUDA matmul")
    parser.add_argument("--fp16", action="store_true", help="Enable FP16 for model")
    parser.add_argument("--max_gpus", type=int, default=2, help="Max number of GPUs to use")
    parser.add_argument("--drop_duplicates", action="store_true", help="Drop duplicate documents (by ID)")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device_ids = get_unused_gpu_ids()
        device_ids = device_ids[:args.max_gpus]
        # NOTE temporary solution:
        if len(device_ids) in [0,1]:
            device_ids = [0, 1]
        device = torch.device(f'cuda:{device_ids[0]}')
        logger.info(f"Using free GPUs: {device_ids}")
    else:
        device = torch.device('cpu')
        device_ids = [0]
        logger.info(f"Using {device}")

    if args.tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    logger.info(f"Loading dataset: {args.dataset}")
    hf_access_token = os.getenv("HF_READ_TOKEN") # None if not set
    if args.dataset.endswith(".csv"):
        dataset = load_dataset("csv", data_files=args.dataset)
    else:
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

    logger.info(f"Loading tokenizer: {args.encoder}")
    tokenizer = AutoTokenizer.from_pretrained(args.encoder)

    def prepare_documents(examples, fields: list, prefix=None):
        """Concatenate fields + add prefix + tokenize, and compute number of
        tokens per document.
        NOTE we don't pad here, we'll do it later in the DataLoader
        """
        texts = []
        for element in zip(*[examples[field] for field in fields]):
            concatenated = ". ".join(element)
            if prefix:
                concatenated = f"{prefix} {concatenated}"
            texts.append(concatenated)
        inputs = tokenizer(
            texts, max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="np")
            # NOTE "np" should be faster than "pt" (and "pt" requires padding here)
        inputs["n_tokens"] = [mask.sum() for mask in inputs["attention_mask"]]
        return inputs

    logger.info(f"Preparing and tokenizing documents")
    # 1st remove any field that is not needed:
    cols_to_remove = [col for col in dataset.column_names if col not in args.fields + [args.id_field]]
    dataset = dataset.remove_columns(cols_to_remove)
    dataset = dataset.map(
        lambda examples: prepare_documents(examples, fields=args.fields, prefix=args.prefix),
        batched=True, num_proc=8, remove_columns=args.fields) # we only need IDs

    logger.info(f"Sorting docs by number of tokens")
    dataset = dataset.sort("n_tokens", reverse=True)
    dataset = dataset.remove_columns(["n_tokens"])

    logger.info(f"Loading model: {args.encoder}")
    doc_encoder = TextEncoder(
        args.encoder, device=device, pooling=args.pooling, fp16=args.fp16
    )
    doc_encoder.load_model(device_ids=device_ids)

    dataset = dataset.with_format("torch")
    data_collator = DataCollatorWithPadding(
        tokenizer=tokenizer, padding="longest", max_length=tokenizer.model_max_length)

    def collate(examples):
        # NOTE we can't use data_collator directly because it expects only tensors
        # and we need to keep the ID to write the embeddings
        ids = [example[args.id_field] for example in examples]
        data = [{k: v for k, v in example.items() if k != args.id_field} for example in examples]
        padded_data = data_collator(data)
        return {"id": ids, **padded_data}

    logger.info(f"Encoding documents with model: {args.encoder} (per device batch size: {args.batch_size})")
    total_batch_size = args.batch_size * len(device_ids)
    batch_sampler = BatchShufflingSampler(dataset, batch_size=total_batch_size)
    dataloader = DataLoader(
        dataset, collate_fn=collate, batch_sampler=batch_sampler, pin_memory=False)
    # NOTE we currently write to files because we it seems faiss requires
    # to load the whole index in memory to add new vectors...

    Path(vectors_dir).mkdir(parents=True, exist_ok=True)
    Path(ids_dir).mkdir(parents=True, exist_ok=True)
    n_samples = len(dataset)
    pbar = tqdm(total=n_samples, desc="Encoded", unit="docs")

    for i, batch in enumerate(dataloader, start=last_batch + 1):
        ids = batch["id"]
        batch_data = {k: v.to(device) for k, v in batch.items() if k != "id"}
        with torch.inference_mode():
            embeddings = doc_encoder.encode(batch_data)
            embeddings = embeddings.cpu().numpy() # float16 if using fp16
        # print(batch_data["input_ids"].shape)
        # print(f"pad % = {100 * (batch_data['attention_mask'] == 0).sum() / batch_data['attention_mask'].numel():.2f}")
        # print(embeddings.shape)
        # print(embeddings.dtype)
        vectors_file = f"{args.outdir}/vectors/batch_{i:05d}.npy"
        ids_file = f"{args.outdir}/ids/batch_{i:05d}.txt"
        if not Path(vectors_file).exists():
            np.save(vectors_file, embeddings)
            with open(ids_file, "w") as f:
                f.writelines(id_ + "\n" for id_ in ids)
        else:
            raise FileExistsError(f"File {vectors_file} already exists")
        pbar.update(len(ids))

    logger.info(f"Done!")


if __name__ == "__main__":
    main()


# # See:
# # https://huggingface.co/docs/datasets/en/stream#stream
# # https://github.com/castorini/pyserini/blob/b7e1da305dd31b195244d49321087505996260c6/pyserini/encode/__main__.py
# # https://github.com/huggingface/blog/blob/main/hf-bitsandbytes-integration.md
# # https://blog.eleuther.ai/transformer-math/
