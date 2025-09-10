"""Create a CSV of queries from all Spanish general-domain IR datasets
(mmarco, sqac, miracl, ours)
"""

import os
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import datasets
import unidecode

from utils import set_logger


logger = set_logger()


def main():
    # args:
    parser = ArgumentParser()
    parser.add_argument("--out_file", type=str, required=True)
    args = parser.parse_args()

    logger.info("Loading queries from MIRACL")
    df_miracl = get_miracl_queries()

    logger.info("Loading queries from SQAC")
    df_sqac = get_sqac_queries()

    logger.info("Loading queries from MMARCO")
    df_mmarco = get_mmarco_queries()

    logger.info("Loading queries from MESSIRVE")
    df_messirve = get_messirve_queries()

    logger.info("Concatenating datasets")
    df_queries = pd.concat(
        [df_miracl, df_sqac, df_mmarco, df_messirve]
    )
    df_queries = df_queries.reset_index(drop=True)

    logger.info("Cleaning queries")
    df_queries = clean_queries_df(df_queries)

    logger.info(f"Saving to {args.out_file}")
    df_queries.to_csv(args.out_file, index=False)

    logger.info("Done!")


def get_miracl_queries() -> pd.DataFrame:
    """
    Requires previously downloading the whole dataset:
    ```bash
    # Requires Access Token authentication:
    mkdir -p data &&
    cd data && 
    git clone https://huggingface.co/datasets/miracl/miracl &&
    cd -
    ```
    NOTE this downloads all subsets -- it might not be the best or only way to do this.
    But this is because the regular load_dataset() fails.
    """
    df = pd.DataFrame()
    for split in ["train", "dev", "test"]:
        df_tmp = read_miracl_queries(split)
        df_tmp["split"] = split
        df = pd.concat([df, df_tmp])
    df["dataset"] = "miracl"
    return df


def get_sqac_queries() -> pd.DataFrame:
    df = pd.DataFrame()
    for split in ["train", "dev", "test"]:
        df_tmp = get_sqac(split)
        df_tmp["split"] = split
        df = pd.concat([df, df_tmp])
    df["dataset"] = "sqac"
    df = df[["id", "query", "split", "dataset"]].drop_duplicates()
    return df


def get_mmarco_queries() -> pd.DataFrame:
    """
    """
    mmarco_queries = datasets.load_dataset('unicamp-dl/mmarco', 'queries-spanish', trust_remote_code=True)
    df = pd.DataFrame()
    for split in ["train", "dev"]:
        df_tmp = mmarco_queries[split].to_pandas().rename(columns={"text": "query"})
        df_tmp["split"] = split
        df = pd.concat([df, df_tmp])
    df["dataset"] = "mmarco"
    df = df[["id", "query", "split", "dataset"]].drop_duplicates()
    return df


def get_messirve_queries() -> pd.DataFrame:
    hf_access_token = os.getenv("HF_READ_TOKEN") # None if not set
    df = pd.DataFrame()
    for split in ["train", "test"]:
        df_tmp = datasets.load_dataset("spanish-ir/messirve", "full", token=hf_access_token)[
            split].to_pandas()
        df_tmp = df_tmp[['id', 'query']]
        df_tmp["split"] = split
        df = pd.concat([df, df_tmp])
    df["dataset"] = "messirve"
    df = df[["id", "query", "split", "dataset"]].drop_duplicates()
    return df


def clean_queries_df(df: pd.DataFrame) -> pd.DataFrame:
    """Remove punctuation, lowercase and remove accents.
    Drop duplicates. Add ID for each query. Remove where query is empty (only one)
    """
    df = df.rename(columns={"query": "original_query"})
    df["query"] = (
        df["original_query"]
        .str.lower()
        .str.replace(r'[^\w\s]', '', regex=True)
        .apply(remove_accents_but_keep_enie)
        # .apply(unidecode.unidecode) # we need to keep the Ñ
    )
    df = df.rename(columns={"id": "original_id"})
    df["id"] = np.arange(len(df))
    df = df[(df["query"].notnull()) & (df["query"] != "")]
    return df


def read_miracl_queries(split: str):
    split_ = f"test-b" if split == "test" else split
    df = pd.read_csv(
        f"data/miracl/miracl-v1.0-es/topics/topics.miracl-v1.0-es-{split_}.tsv",
        sep="\t", header=None, names=["id", "query"],
    )
    return df


def sqac_examples(sqac_data):
    """This function returns the examples in the raw (text) form.
    We replicate the loading script frmo HuggingFace"""
    for article in sqac_data:
        title = article.get("title", "").strip()
        for paragraph in article["paragraphs"]:
            context = paragraph["context"].strip()
            for qa in paragraph["qas"]:
                question = qa["question"].strip()
                id_ = qa["id"]

                answer_starts = [answer["answer_start"] for answer in qa["answers"]]
                answers = [answer["text"].strip() for answer in qa["answers"]]

                # Features currently used are "context", "question", and "answers".
                # Others are extracted here for the ease of future expansions.
                yield id_, {
                    "title": title,
                    "context": context,
                    "question": question,
                    "id": id_,
                    "answers": {
                        "answer_start": answer_starts,
                        "text": answers,
                    },
                }


def get_sqac(split):
    sqac = pd.read_json(f"https://huggingface.co/datasets/PlanTL-GOB-ES/SQAC/resolve/main/{split}.json")
    sqac_data = sqac["data"].tolist()
    examples = list(sqac_examples(sqac_data))
    formatted_data = [
        {
            'id': item[0],
            'title': item[1]['title'],
            'text': item[1]['context'],
            'query': item[1]['question'],
        }
        for item in examples
    ]
    df_sqac = pd.json_normalize(formatted_data)
    return df_sqac


def remove_accents_but_keep_enie(text):
    # Temporarily replace Ñ and ñ with placeholders
    text = text.replace('Ñ', 'TEMP_ENE_UPPER')
    text = text.replace('ñ', 'TEMP_ENE_LOWER')
    text = unidecode.unidecode(text)
    # Replace placeholders back with Ñ and ñ
    text = text.replace('TEMP_ENE_UPPER', 'Ñ')
    text = text.replace('TEMP_ENE_LOWER', 'ñ')
    return text


if __name__ == "__main__":
    main()
