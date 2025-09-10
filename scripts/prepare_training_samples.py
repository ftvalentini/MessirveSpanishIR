"""Prepare files to fine-tune dense retrieval model.

See:
* https://github.com/microsoft/unilm/blob/b42b637ac77f6c043e262290df956ae6286de9cb/simlm/misc/prepare_msmarco_data.py
"""

import json
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils import set_logger


logger = set_logger()


def main():
    # args
    parser = ArgumentParser()
    parser.add_argument("--ids_file",
                        type=str, default="data/messirve_training/docid_map.json", help="Path to dict from old to new docids")
    parser.add_argument("--trec_dir",
                        type=str, default="data/messirve-v1.0", help="Path to TREC topics and qrels")
    parser.add_argument("--runs_dir",
                        type=str, default="runs/retrieved", help="Path to BM25 retrieved results")
    parser.add_argument("--out_dir",
                        type=str, default="runs/messirve_training", help="Output dir to store processed corpus and negative mined passages")
    parser.add_argument("--n_negatives",
                        type=int, default=200, help="Number of negative passages to get for each query")
    parser.add_argument("--dev_frac",
                        type=float, default=0.15, help="APPROX. fraction of queries to use as dev set")
    parser.add_argument("--seed",
                        type=int, default=33, help="Random seed")
    
    args = parser.parse_args()
    ids_file = args.ids_file
    out_dir = args.out_dir
    trec_dir = args.trec_dir
    runs_dir = args.runs_dir
    n_negatives = args.n_negatives
    dev_frac = args.dev_frac
    seed = args.seed

    # main
    logger.info(f"Reading docids from {ids_file}")
    with open(ids_file) as f:
        old2newids = json.load(f)

    SUBSETS = [
        "mx", 
        "full",
        "us", "co", "es", "ar", "pe", "ve", "cl", "gt", "ec", "cu", "bo",
        "do", "hn", "sv", "py", "ni", "cr", "pa", "pr", "uy", "no_country",
    ]

    for subset in SUBSETS:
        
        logger.info(f"Reading data for '{subset}'")
        df_qrels, df_run = load_data(trec_dir, runs_dir, subset)

        logger.info(f"Splitting data in train/dev for '{subset}'")
        df_run_train, df_run_dev, df_qrels_train, df_qrels_dev = \
            train_dev_split(df_qrels, df_run, dev_frac, seed)
        del df_qrels, df_run

        logger.info(f"Converting docids for '{subset}'")
        df_qrels_train['docid'] = df_qrels_train['docid'].map(old2newids)
        df_qrels_dev['docid'] = df_qrels_dev['docid'].map(old2newids)
        df_run_train['docid'] = df_run_train['docid'].map(old2newids)
        df_run_dev['docid'] = df_run_dev['docid'].map(old2newids)

        Path(out_dir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Saving training data for '{subset}'")
        new_docids = list(old2newids.values())
        out_file_ = f"{out_dir}/train_neg{n_negatives}_{subset}.jsonl"
        save_training_data(df_qrels_train, df_run_train, out_file_, new_docids, n_negatives, seed)
        
        logger.info(f"Saving dev data for '{subset}'")
        out_file_ = f"{out_dir}/dev_neg{n_negatives}_{subset}.jsonl"
        save_training_data(df_qrels_dev, df_run_dev, out_file_, new_docids, n_negatives, seed)

    logger.info("Done!")


def load_data(trec_dir, runs_dir, subset):
    df_topics = pd.read_csv(
            f"{trec_dir}/topics.messirve-v1.0-{subset}-train.tsv", 
            sep='\t', names=['qid', 'query'], dtype={'qid': int, 'query': str}
        )
    df_qrels = pd.read_csv(
            f"{trec_dir}/qrels.messirve-v1.0-{subset}-train.tsv",
            sep='\t', names=['qid', 'Q0', 'docid', 'rel'],
            usecols=[0, 2, 3],
            dtype={'qid': int, 'docid': str, 'rel': int},
        )
    # merge qrels with topics:
    df_qrels = df_qrels.merge(df_topics, on='qid')
    del df_topics
    # print("Reading run")
    df_run = pd.read_csv(
        f"{runs_dir}/run.eswiki_20240401.bm25.messirve-v1.0-{subset}-train.tsv",
        sep=" ",
        names=["qid", "Q0", "docid", "rank", "score", "run"],
        usecols=[0, 2, 3],
        dtype={"qid": int, "docid": str, "rank": int},
    )
    return df_qrels, df_run


def train_dev_split(df_qrels, df_run, dev_frac, seed=33) -> tuple:
    """Split qrels and run into train and dev sets so that no question and no relevant
    article is in both sets.
    """
    # extract article_ids from docids (e.g. "156168#12" -> "156168")
    df_qrels['article_id'] = df_qrels['docid'].str.split('#').str[0]
    # Split articles into train and dev:
    aids = df_qrels['article_id'].unique()
    aids_train, aids_dev = train_test_split(aids, test_size=dev_frac, random_state=seed)
    df_qrels_train = df_qrels[df_qrels['article_id'].isin(aids_train)]
    df_qrels_dev = df_qrels[df_qrels['article_id'].isin(aids_dev)]
    # Check overlap between train and dev qids, aids:
    qids_train = set(df_qrels_train['qid'])
    qids_dev = set(df_qrels_dev['qid'])
    overlap_qids = qids_train & qids_dev
    while len(overlap_qids) > 0:
        print(f"Overlap between train and dev queries: moving {len(overlap_qids)} queries from dev to train")
        df_qrels_train = pd.concat([df_qrels_train, df_qrels_dev[df_qrels_dev['qid'].isin(overlap_qids)]])
        df_qrels_dev = df_qrels_dev[~df_qrels_dev['qid'].isin(overlap_qids)]
        overlap_aids = set(df_qrels_train['article_id']) & set(df_qrels_dev['article_id'])
        if len(overlap_aids) > 0:
            print(f"Overlap between train and dev articles: moving {len(overlap_aids)} articles from dev to train")
            df_qrels_train = pd.concat([df_qrels_train, df_qrels_dev[df_qrels_dev['article_id'].isin(overlap_aids)]])
            df_qrels_dev = df_qrels_dev[~df_qrels_dev['article_id'].isin(overlap_aids)]
            overlap_qids = set(df_qrels_train['qid']) & set(df_qrels_dev['qid'])
        else:
            overlap_qids = set()
    # Assert no qid overlap and no article overlap:
    assert len(set(df_qrels_train['qid']) & set(df_qrels_dev['qid'])) == 0, "Qid overlap"
    assert len(set(df_qrels_train['article_id']) & set(df_qrels_dev['article_id'])) == 0, "Article overlap"
    qids_train = df_qrels_train['qid'].unique()
    qids_dev = df_qrels_dev['qid'].unique()
    df_qrels_train = df_qrels_train.drop(columns=['article_id']).reset_index(drop=True)
    df_qrels_dev = df_qrels_dev.drop(columns=['article_id']).reset_index(drop=True)
    # Print final train stats (N and %):
    n_train = len(qids_train)
    n_dev = len(qids_dev)
    n_total = n_train + n_dev
    print(f"Train queries: {n_train} ({n_train / n_total:.2%})")
    print(f"Dev queries: {n_dev} ({n_dev / n_total:.2%})")
    # Split run into train and dev:
    df_run_train = df_run[df_run['qid'].isin(qids_train)].reset_index(drop=True)
    df_run_dev = df_run[df_run['qid'].isin(qids_dev)].reset_index(drop=True)
    return df_run_train, df_run_dev, df_qrels_train, df_qrels_dev


def save_training_data(
        df_qrels: pd.DataFrame, df_run: pd.DataFrame, out_file: str, doc_ids: list[str],
        num_negatives: int = 200, seed: int = 33
    ):
    """Converts qrels and run to samples for train.jsonl. 
    If there are less than num_negatives in the run, use random negatives 
    from doc_ids.
    """
    np.random.seed(seed)
    # df_qrels = df_qrels.sort_values(by=['qid'], ascending=[True, False])
    # Keep only first num_negatives for each query:
    qid2query = df_qrels.drop_duplicates(subset='qid').set_index('qid')['query'].to_dict()
    df_qrels = df_qrels.query("rel > 0").drop(columns=['rel', 'query']) # we only have positives but still
    # NOTE we have to keep more than num_negatives if case the retriever found them:
    n_keep = df_qrels['qid'].value_counts().max() + num_negatives
    df_run = df_run.query("rank <= @n_keep").copy()
    df_run["score"] = round(1.0 / df_run["rank"], 6)
    df_run = df_run.drop(columns=['rank'])
    df_full = df_qrels.merge(df_run, on=['qid'], how='left', suffixes=('_qrels', '_run'))
    del df_qrels, df_run
    df_full["docid_run"] = df_full["docid_run"].fillna("")
    df_full = df_full.sort_values(by=['score'], ascending=[False])
    rng = np.random.default_rng(seed)
    with open(out_file, 'w', encoding='utf-8') as writer:
        n_qids = df_full['qid'].nunique()
        pbar = tqdm(total=n_qids, desc=f"Writing {out_file}", unit="qids")
        for qid, df_qid in df_full.groupby('qid', sort=False):
            # if qid in [38963, 5333399]:
            #     print("DEBUG")
            positive_docids = df_qid['docid_qrels'].unique().tolist()
            # # This should never happen in our case:
            # if len(positive_ids) == 0:
            #     logger.info(f"Skipping {qid} because there are no positives")
            #     continue
            retrieved_docids = df_qid['docid_run'].tolist()
            # get hard negatives (NOTE we are assuming docs are sorted by score):
            negative_ids = [docid for docid in retrieved_docids if docid not in positive_docids and docid != ""]
            if len(negative_ids) < num_negatives:
                # add random negatives:
                while len(negative_ids) < num_negatives:
                    logger.info(f"{qid} has {len(negative_ids)} hard negatives, adding random")
                    random_negatives = rng.choice(doc_ids, size=num_negatives-len(negative_ids), replace=False)
                    random_negatives = set(random_negatives) - set(positive_docids) - set(negative_ids)
                    negative_ids.extend(random_negatives)
            negative_ids = negative_ids[:num_negatives]
            np.random.shuffle(negative_ids)
            doc2score = dict(zip(df_qid['docid_run'], df_qid['score'])) # NOTE think these scores arent used in the end
            negative_scores = [doc2score.get(docid, -1.0) for docid in negative_ids]
            positive_scores = [-1.0] * len(positive_docids)
            sample = {
                'query_id': qid,
                'query': qid2query[qid],
                'positives': {'doc_id': positive_docids, 'score': positive_scores},
                'negatives': {'doc_id': negative_ids, 'score': negative_scores}
            }
            writer.write(json.dumps(sample, ensure_ascii=False, separators=(',', ':')) + '\n')
            pbar.update(1)


if __name__ == "__main__":
    main()
