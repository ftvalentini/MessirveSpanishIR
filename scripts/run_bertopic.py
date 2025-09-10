"""Run BERTopic on a dataset of queries
"""

import argparse
from pathlib import Path

import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
import unidecode
from bertopic import BERTopic
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from bertopic.vectorizers import ClassTfidfTransformer
from hdbscan import HDBSCAN # Slow
from nltk.corpus import stopwords

from utils import set_logger


logger = set_logger()


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, required=True,
                        help="File with input texts")
    parser.add_argument("--embeddings", type=str, required=True,
                        help="Path with previously computed embeddings")
    parser.add_argument("--outdir", type=str, required=True,
                        help="Output dir to store model")
    parser.add_argument("--cluster_method", type=str, default="kmeans",
                        help="kmeans or hdbscan")
    parser.add_argument("--n", type=int, default=50,
                        help="Number of topics to extract (for KMeans) or min cluster size (for HDBSCAN)")
    args = parser.parse_args()

    logger.info("Loading previosuly encoded dataset")
    ids = load_ids(f"{args.embeddings}/ids")
    vectors = load_vectors(
        f"{args.embeddings}/vectors", normalize_l2=True, device="cpu" # NOTE bertopic requires numpy
    )

    # # Keep a random sample to debug:
    # idx = np.random.choice(len(ids), size=1000, replace=False)
    # ids = ids[idx]
    # vectors = vectors[idx]

    logger.info("Loading dataset")
    df_queries = pd.read_csv(args.dataset, low_memory=False)
    # sort df according to order in ids list:
    df_queries["id"] = df_queries["id"].astype(str)
    df_queries = df_queries.set_index("id").loc[ids].reset_index()
    logger.info(f"{df_queries.shape=}")

    vectors = vectors.numpy()
    queries = df_queries["query"].tolist()
    dim_model = PCA(n_components=30)

    if args.cluster_method == "hdbscan":
        cluster_model = HDBSCAN(
            min_cluster_size=args.n, metric='euclidean', cluster_selection_method='eom', min_samples=10)
    elif args.cluster_method == "kmeans":
        cluster_model = KMeans(n_clusters=args.n, random_state=33)
    else:
        raise ValueError(f"Invalid cluster_method: {args.cluster_method}")
    stop_words = stopwords.words("spanish")
    stop_words = [unidecode.unidecode(w) for w in stop_words]
    vectorizer_model = CountVectorizer(stop_words=stop_words, ngram_range=(1, 1))
    ctfidf_model = ClassTfidfTransformer(reduce_frequent_words=True)

    topic_model = BERTopic(
        embedding_model=None,
        umap_model=dim_model,
        hdbscan_model=cluster_model,
        vectorizer_model=vectorizer_model,
        ctfidf_model=ctfidf_model,
        top_n_words=10,
        verbose=True,
        calculate_probabilities=False,
    )

    logger.info("Running BERTopic...")
    topics, probs = topic_model.fit_transform(documents=queries, embeddings=vectors) # list, np.ndarray
    # print(topic_model.representative_docs_[2])

    logger.info("Saving...")
    Path(f"{args.outdir}/model").mkdir(parents=True, exist_ok=True)
    Path(f"{args.outdir}/topics").mkdir(parents=True, exist_ok=True)
    topic_model.save(f"{args.outdir}/model", serialization="safetensors", save_ctfidf=True, save_embedding_model=None)
    np.save(f"{args.outdir}/topics/topics.npy", topics)
    if probs is not None:
        np.save(f"{args.outdir}/topics/probs.npy", probs)

    logger.info("Done!")


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
        raise FileNotFoundError(f"No txt files found in {ids_dir}")
    ids = []
    for file in files:
        with open(file) as f:
            ids.extend(f.read().splitlines())
    return np.array(ids)    


if __name__ == "__main__":
    main()
