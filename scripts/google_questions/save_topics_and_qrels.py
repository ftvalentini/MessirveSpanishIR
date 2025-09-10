
import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import HfApi

from helpers import set_logger, connect_to_db


logger = set_logger()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, required=True, help="Input csv file (qrels.csv.gz)")
    parser.add_argument("--outdir", type=str, required=True, help="Path to save all tsv output files")
    parser.add_argument("--dataset_name", type=str, help="ID to use in filenames and HF repo (if saving to HF)")
    parser.add_argument("--hf_dir", type=str, help="HF Hub dir to save files")
    # parser.add_argument("--topics_outfile", type=str, required=True, help="Path to tsv output file")
    # parser.add_argument("--score_threshold", type=float, default=0.5,
    #                     help="Threshold for matching score between Google answer and corpus document")
    # parser.add_argument("--min_version", type=int, default=1,
    #                     help="Min. version of the matched answers to use TODO revisar si es necesario")
    args = parser.parse_args()

    # logger.info("Connecting to DB...")
    # conn, cur = connect_to_db()

    # logger.info("Fetching matched answers data...")
    # df_questions = fetch_questions_answers(cur, min_version=args.min_version, min_match_score=args.score_threshold)

    logger.info("Reading input file...")
    df_questions = pd.read_csv(args.input_file)

    logger.info("Saving qrels and topic files...")
    splits = df_questions["split"].unique()
    countries = df_questions["country"].unique()
    for split in splits:
        for country in countries:
            logger.info(f"Saving files for {country} ({split})...")
            df_i = df_questions[(df_questions["split"] == split) & (df_questions["country"] == country)].copy()
            # qrels:
            df_qrels = create_qrels_df(df_i)
            out_file = Path(args.outdir) / f"qrels.{args.dataset_name}-{country}-{split}.tsv"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            df_qrels.to_csv(out_file, sep="\t", index=False, header=False)
            # topics:
            df_topics = create_topics(df_i)
            out_file = Path(args.outdir) / f"topics.{args.dataset_name}-{country}-{split}.tsv"
            out_file.parent.mkdir(parents=True, exist_ok=True)
            df_topics.to_csv(out_file, sep="\t", index=False, header=False)

    if args.hf_dir is not None:
        hf_api = HfApi()

        # repo_dir = f"{args.hf_hub}/{args.dataset_name}"
        logger.info(f"Creating HF repo '{args.hf_dir}'...")
        
        hf_api.create_repo(args.hf_dir, private=True, repo_type="dataset", exist_ok=True)

        logger.info(f"Uploading files to HF repo '{args.hf_dir}'...")
        hf_api.upload_folder(
            folder_path=args.outdir,
            # path_in_repo="ar/qrels/qrels.miracl-v1.0-ar-dev.tsv",
            repo_id=args.hf_dir,
            repo_type="dataset"
        )

    # logger.info("Closing DB connection...")
    # cur.close()
    # conn.close()

    logger.info("Done!")


def fetch_questions_answers(
        cur, min_version: int = 1, min_match_score: float = 0.5
    ) -> pd.DataFrame: 
    cur.execute(f"""
        SELECT e.id, q.question, e.corpus_docid
        FROM extractions AS e
            INNER JOIN queries AS q ON e.id = q.id
        WHERE corpus_docid IS NOT NULL
            AND e.match_v >= {min_version}
            AND e.match_score >= {min_match_score}
        ;
    """)
    res = cur.fetchall()
    df = pd.DataFrame(res, columns=["id", "question", "docid"])
    return df


def create_qrels_df(df_questions: pd.DataFrame) -> pd.DataFrame:
    """Create qrels df with columns: ["id", "q0", "docid", "rel"] to comply
    with pyserini format.
    """
    df_qrels = df_questions.copy()
    df_qrels["q0"] = "Q0"
    df_qrels["rel"] = 1
    df_qrels = df_qrels[["id", "q0", "docid", "rel"]]
    return df_qrels


def create_topics(df_questions: pd.DataFrame) -> pd.DataFrame:
    """Create topics df with columns: ["id", "desc"] to comply with pyserini
    format.
    """
    df_topics = df_questions.copy()
    df_topics["desc"] = df_topics["query"]
    df_topics = df_topics[["id", "desc"]].drop_duplicates().reset_index(drop=True)
    return df_topics


if __name__ == "__main__":
    main()
