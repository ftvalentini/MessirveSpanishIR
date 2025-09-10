"""Match the extracted answers with documents from the corpus. We currently
keep the best match for each answer and save a match_score.
"""

import argparse
import multiprocessing
import re
import unicodedata
import urllib
from typing import Tuple

import pandas as pd
import datasets
import pylcs
from tqdm import tqdm
from psycopg2 import extras

from helpers import connect_to_db, set_logger


logger = set_logger()


def main(args):

    logger.info("Fetching full list of IDs to process...")
    ids = fetch_ids_to_process(args.version)
    n_ids = len(ids)
    batches = [ids[i:i+args.batch_size] for i in range(0, n_ids, args.batch_size)]
    n_batches = len(batches)

    logger.info(f"Processing {n_ids} IDs in {n_batches} batches of {args.batch_size}...")
    with multiprocessing.Pool(processes=args.max_workers) as pool, tqdm(total=n_batches) as pbar:
        for _ in pool.imap_unordered(process_batch, batches):
            pbar.update(1)

    logger.info("Done!")


def fetch_ids_to_process(version: int) -> list:
    conn, cur = connect_to_db()
    cur.execute(f"""
        SELECT e.id
        FROM extractions AS e
        WHERE answer IS NOT NULL
            AND answer_url LIKE '%es.wikipedia.org%'
            AND answer_type IN ('feat_snip', 'rich_snip', 'rich_set', 'rich_list', 'descript', 'knowledge')
            AND (e.match_v < {version} OR e.match_v IS NULL)
            AND corpus_docid IS NULL
            --AND (e.match_v < 6 OR e.match_v IS NULL)
        ;
    """)
    res = cur.fetchall()
    ids = [row[0] for row in res]
    cur.close()
    conn.close()
    return ids


def process_batch(ids: list) -> None:
    conn, cur = connect_to_db()
    df_answers = fetch_answers_to_match(cur, ids)
    
    # for each answer find matching doc in the corpus:
    results = {}  # id: (docid, score)
    for _, row in df_answers.iterrows():
        docid, score = find_docid(row["id"], row["answer_url"], row["answer"], DF_DOCS)
        results[row["id"]] = (docid, score)
    
    # Add results to the df:
    df_answers["corpus_docid"] = df_answers["id"].apply(lambda x: results[x][0])
    df_answers["match_score"] = df_answers["id"].apply(lambda x: results[x][1])
    
    # Build df to update the extractions table:
    df_matches = df_answers[
        df_answers["corpus_docid"].notnull()
        ].reset_index(drop=True).copy()[["id", "corpus_docid", "match_score"]]
    df_matches["version"] = args.version
    
    # Update the extractions table: 
    update_table(cur, df_matches)
    conn.commit()
    cur.close()
    conn.close()


def fetch_answers_to_match(cur, ids: list) -> pd.DataFrame:
    ids_str = ", ".join([str(id) for id in ids])
    cur.execute(f"""
        SELECT e.id, answer, answer_url, q.question
        FROM extractions AS e
            LEFT JOIN queries AS q ON e.id = q.id
        WHERE answer IS NOT NULL AND e.id IN ({ids_str})
        ;
    """)
    rows = cur.fetchall()
    df_answers = pd.DataFrame(rows, columns=["id", "answer", "answer_url", "question"])
    df_answers = df_answers.sort_values("id").reset_index(drop=True)
    return df_answers


def find_docid(
        question_id: int, answer_url: str, answer: str, df_docs: pd.DataFrame
    ) -> Tuple[str, float]:
    """Find the docid of the document that contains the answer.
    Returns (docid, match_score)
    """
    article_title = extract_article_title(answer_url)
    # NOTE If it is "Portal", we consider all articles _starting with_ the title
    # e.g. "Portal: Oaxaca" matches all "Portal: Oaxaca", "Portal: Oaxaca/Destacado", "Portal: Oaxaca/Municipio del mes", etc.
    if article_title.startswith("Portal:"):
        df_article = df_docs[df_docs["title"].str.startswith(article_title)].copy()
    else:
        df_article = df_docs[df_docs["title"] == article_title].copy()
    
    if len(df_article) == 0:
        logger.warning(f"id={question_id}: article '{article_title}' not found ({answer_url})")
        return None, None
    
    # Find the passage in the article that contains the answer
    clean_answer = clean_string(answer)
    df_article["clean_text"] = df_article["text"].apply(clean_string)
    df_article["match"] = df_article["clean_text"].str.contains(clean_answer)
    df_article["similarity"] = 1.0 # default value
    
    if df_article["match"].sum() > 1:
        logger.debug(f"id={question_id}: multiple exact matches in article '{article_title}': keeping the first one")
    
    elif df_article["match"].sum() == 0:
        logger.debug(f"id={question_id}: no exact match in article '{article_title}': computing similarity")
        df_article["similarity"] = df_article["clean_text"].apply(
            lambda x: passage_similarity(clean_answer, x))
        df_article = df_article.sort_values("similarity", ascending=False)
        df_article["match"] = True # to consider the best match later
        similar_docs = df_article["similarity"] > 0.5 # we'll consider a match even if below threshold
        
        if similar_docs.sum() > 1:
            logger.debug(f"id={question_id}: multiple similar passages above threshold in article '{article_title}': keeping the best one")
        
        elif similar_docs.sum() == 0:
            logger.debug(f"id={question_id}: no passage above threshold in article '{article_title}': keeping the best one")
    
    # Keep the only match, first match or best match:
    docid, match_score = df_article[df_article["match"]].iloc[0][["docid", "similarity"]] 
    return docid, match_score


def update_table(cursor, df: pd.DataFrame) -> None:
    query = f"""
        UPDATE extractions   
        SET corpus_docid = data.docid,
            match_score = data.match_score,
            match_v = data.version
        FROM (VALUES %s) AS data(id, docid, match_score, version)
        WHERE extractions.id = data.id
        ;
    """
    values = [(row['id'], row['corpus_docid'], row['match_score'], row['version']) for _, row in df.iterrows()]
    extras.execute_values(cursor, query, values)
    # NOTE use cur.executemany(query, values) if this doesn't work


def extract_article_title(url: str) -> str:
    """NOTE urls are like:
    https://es.wikipedia.org/wiki/Grupo_sangu%C3%ADneo#:~:text=El%20grupo%20O...
    https://es.wikipedia.org/wiki/VIH/sida
    https://es.wikipedia.org/wiki/VIH/sida#:~:text=El%20tiempo%20promedio%20...
    --> must extract text between "wiki/" and optional "#"
    e.g. "Grupo sanguíneo", "VIH/sida", "VIH/sida"
    We also handle "sections" of Portales as if they were from the same page
    e.g. "Portal:OTAN/Estados_miembros" --> "Portal: OTAN" 
    """
    extracted_name = url.split("wiki/")[-1].split("#")[0]
    article_name = urllib.parse.unquote(extracted_name).replace("_", " ")
    # If it starts with "Portal/Anexo", insert space after "Portal:" 
    namespace_prefixes = ["Portal:", "Anexo:"]
    if article_name.startswith(tuple(namespace_prefixes)):
        article_name = article_name.replace(":", ": ", 1)    
    # For portal, remove everything after first slash:
    if article_name.startswith("Portal:"):
        article_name = article_name.split("/")[0]
    return article_name


def clean_string(s: str) -> str:
    """To improve passage matching"""
    s = s.lower()
    s = unicodedata.normalize("NFKD", s)
    s = re.sub(r'[^a-z0-9 ]', '', s)
    s = re.sub(r'\s+', ' ', s)
    return s


def passage_similarity(answer: str, passage: str) -> float:
    return pylcs.lcs_string_length(answer, passage) / len(answer)
    # return 1 - distance(answer, passage, weights=(0, 1, 1)) / len(answer)


def get_corpus_df(path: str) -> pd.DataFrame:
    corpus = datasets.load_dataset(path)
    df_docs = corpus["train"].to_pandas()
    return df_docs


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus_dir", type=str, default="runs/corpora/eswiki-20240401-corpus")
    parser.add_argument("--version", type=int, default=1,
                        help="Increment version to go through answers we are uncertain about again")
    parser.add_argument("--max_workers", type=int, default=12)
    parser.add_argument("--batch_size", type=int, default=80)
    args = parser.parse_args()
    
    DF_DOCS = get_corpus_df(args.corpus_dir)

    main(args)
