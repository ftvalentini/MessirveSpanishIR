"""Retrieve K closest document embeddings to query embeddings using pytorch. 
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from utils import set_logger, get_unused_gpu_ids


logger = set_logger()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--doc_embeddings", type=str, required=True, help="Directory with document embeddings")
    parser.add_argument("--query_embeddings", type=str, required=True, help="Directory with query embeddings")
    parser.add_argument("--out_file", type=str, required=True, help="File to save results")
    parser.add_argument("--normalize_l2", action="store_true", help="Normalize embeddings to unit L2 norm")
    parser.add_argument("--hits", type=int, default=100, help="Number of closest embeddings to retrieve")
    args = parser.parse_args()

    if torch.cuda.is_available():
        device_ids = get_unused_gpu_ids()
        # NOTE temporary solution:
        if len(device_ids) == 0:
            device_ids = [0, 1]
        device = torch.device(f'cuda:{device_ids[0]}')
    else:
        device = torch.device('cpu')
    logger.info(f"Using {device}")

    if not Path(args.query_embeddings).is_dir():
        raise FileNotFoundError(f"Directory {args.query_embeddings} not found")
    
    if not Path(args.doc_embeddings).is_dir():
        raise FileNotFoundError(f"Directory {args.doc_embeddings} not found")

    queries_dir = f"{args.query_embeddings}/vectors"
    query_ids_dir = f"{args.query_embeddings}/ids"
    
    docs_dir = f"{args.doc_embeddings}/vectors"
    doc_ids_dir = f"{args.doc_embeddings}/ids"

    logger.info("Loading query embeddings and IDs...")
    queries = load_vectors(queries_dir, args.normalize_l2, device)
    query_ids = load_ids(query_ids_dir)
    # print(queries.shape, queries.dtype)

    logger.info("Loading doc IDs...")
    doc_ids = load_ids(doc_ids_dir)

    logger.info("Finding closest document embeddings (for all queries at once, iterating over batches of docs)...")
    n_docs = len(doc_ids)
    top_k_indices, top_k_similarities = find_k_closest(
        queries, docs_dir, args.hits, normalize_l2=args.normalize_l2, total_docs=n_docs)

    logger.info("Making and saving results...")
    top_k_indices = top_k_indices.cpu().numpy()
    top_k_similarities = top_k_similarities.cpu().numpy()
    save_results(top_k_indices, top_k_similarities, query_ids, doc_ids, args.out_file)

    logger.info("Done!")


def find_k_closest(
    queries: torch.Tensor, docs_dir: str, k: int, normalize_l2: bool, total_docs: int = None
) -> tuple[torch.Tensor, torch.Tensor]:

    n_queries = queries.size(0)
    device = queries.device
    dtype = queries.dtype

    # Initialize top K results:
    top_k_indices = torch.zeros((n_queries, k), dtype=torch.long, device=device)
    top_k_similarities = torch.full((n_queries, k), float('-inf'), dtype=dtype, device=device)

    pbar = tqdm(total=total_docs, unit="docs", desc="Processing batches of document embeddings")
    current_index = 0
    docs_files = sorted(Path(docs_dir).glob("batch_*.npy"))

    for file in docs_files:
        batch = np.load(file) # shape (batch_size, emb_dim)
        batch = torch.tensor(batch, device=device)
        if normalize_l2:
            batch = F.normalize(batch, p=2, dim=1)
        batch_size = batch.size(0)

        similarities = torch.matmul(queries, batch.T) # shape (n_queries, batch_size)

        # Add current similarities and indices to the top K so far
        combined_similarities = torch.cat((top_k_similarities, similarities), dim=1)
        batch_indices = torch.arange(
            current_index, current_index + batch_size, device=queries.device).expand(n_queries, -1)
        # NOTE Expanding a tensor does not allocate new memory
        combined_indices = torch.cat((top_k_indices, batch_indices), dim=1)
        # Update top K
        top_k_similarities, top_k_indices_in_combined = torch.topk(combined_similarities, k, dim=1)
        top_k_indices = torch.gather(combined_indices, 1, top_k_indices_in_combined)

        # free memory by deleting variables:
        del batch_indices, combined_similarities, combined_indices
        torch.cuda.empty_cache()

        current_index += batch_size
        pbar.update(batch_size)

    pbar.close()

    return top_k_indices, top_k_similarities


def save_results(
    top_k_indices: np.ndarray, top_k_similarities: np.ndarray,
    query_ids: np.ndarray, doc_ids: np.ndarray, out_file: str
) -> None:
    """Save results using TREC format as in:
    https://github.com/castorini/pyserini/blob/49d8c43eebcc6a634e12f61382f17d1ae0729c0f/pyserini/output_writer.py#L83
    """
    # Indices in long format:
    df_indices = pd.DataFrame(top_k_indices)
    df_indices["query_id"] = query_ids
    df_indices = df_indices.melt(id_vars="query_id", value_name="doc_idx", var_name="rank")
    # Similarities in long format:
    df_sim = pd.DataFrame(top_k_similarities)
    df_sim["query_id"] = query_ids
    df_sim = df_sim.melt(id_vars="query_id", value_name="sim", var_name="rank")
    # Merge in one dataframe:
    df = pd.merge(df_indices, df_sim, on=["query_id", "rank"])
    df["rank"] = df["rank"] + 1
    # Replace each "doc_idx" with the actual docid according to the order in the embeddings dir:
    df["doc_id"] = doc_ids[df["doc_idx"].values]
    df = df.drop(columns=["doc_idx"])
    # Make final expected format:
    df["Q0"] = "Q0"
    df["sim"] = df["sim"] #.round(6) NOTE it crashes with float16
    df["tag"] = "FV"
    df = df[["query_id", "Q0", "doc_id", "rank", "sim", "tag"]].sort_values(["query_id", "rank"])
    # Save results:
    df.to_csv(out_file, index=False, header=False, sep=" ")


def load_vectors(vectors_dir: str, normalize_l2: bool, device: torch.device) -> torch.Tensor:
    files = sorted(Path(vectors_dir).glob("batch_*.npy"))
    if not files:
        raise FileNotFoundError(f"No files found in {vectors_dir}")
    vectors = []
    for file in files:
        vectors.append(np.load(file))
    vectors = np.concatenate(vectors)
    vectors = torch.tensor(vectors, device=device) # keeps the dtype of the numpy array
    if normalize_l2:
        return F.normalize(vectors, p=2, dim=1)
    return vectors


def load_ids(ids_dir: str) -> np.ndarray:
    files = sorted(Path(ids_dir).glob("batch_*.txt"))
    if not files:
        raise FileNotFoundError(f"No files found in {ids_dir}")
    ids = []
    for file in files:
        with open(file) as f:
            ids.extend(f.read().splitlines())
    return np.array(ids)


if __name__ == "__main__":
    main()
