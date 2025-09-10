"""
NOTE: we remove questions ending with " en wikipedia", " la wikipedia", " de wikipedia", 
and those with 3 or less words followed by "wikipedia" because it's not clear
if they are _about_ wikipedia or _searching_ in wikipedia.
"""


import logging
import os

import pandas as pd
from psycopg2 import OperationalError, extras
from unidecode import unidecode

from helpers import QUERY_PATTERNS, add_accented_patterns, connect_to_db, set_logger, build_search_request


DB_NAME = "retrieval"
QUESTIONS_DIR = "runs/google_questions"


logger = set_logger()

def main():
    logging.info("Creating database...")
    create_database()
    
    logging.info("Creating tables if they don't exist...")
    create_extractions_table()
    create_htmls_table()
    create_queries_table()
    
    logging.info("Populating queries table...")
    populate_queries_table()
    
    logging.info("Done!")


def create_database():
    connection = None
    try:
        # Connect to the default PostgreSQL database (postgres) to create a new database
        connection, cursor = connect_to_db(dbname=None)
        connection.autocommit = True
        # Create a new database if it doesn't exist
        cursor.execute(f"""
            SELECT FROM pg_database WHERE datname = '{DB_NAME}'
        """)
        if cursor.fetchone() is None:
            cursor.execute(f"CREATE DATABASE {DB_NAME}")
            print(f"Database {DB_NAME} created successfully")
        else:
            print(f"Database {DB_NAME} already exists")
    except OperationalError as e:
        print(f"Error: {e}")
    finally:
        if connection:
            cursor.close()
            connection.close()


def create_queries_table():
    connection = None
    try:
        connection, cursor = connect_to_db(dbname=DB_NAME)
        # Create a new table if it doesn't exist
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS queries (
                id SERIAL PRIMARY KEY,
                question TEXT,
                country TEXT,
                date DATE,
                url TEXT,
                relevance INT,
                relevance_rank INT
            );
            """
        )
        connection.commit()
        print(f"Table queries created successfully")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    finally:
        # Close the connection
        if connection:
            cursor.close()
            connection.close()


def create_extractions_table():
    # With fields id (foreign key?), short_answer, answer, answer_url, answer_type, extract_v:
    connection = None
    try:
        connection, cursor = connect_to_db(dbname=DB_NAME)
        # Create a new table if it doesn't exist
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS extractions (
                id SERIAL PRIMARY KEY,
                short_answer TEXT,
                answer TEXT,
                answer_url TEXT,
                answer_type TEXT,
                extract_v INT,
                expanded_search BOOLEAN, 
                corpus_docid TEXT,
                match_score FLOAT,
                match_v INT
            );
            """
        )
        connection.commit()
        print("Table extractions created successfully")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    finally:
        if connection:
            cursor.close()
            connection.close()


def create_htmls_table():
    connection = None
    try:
        connection, cursor = connect_to_db(dbname=DB_NAME)
        cursor.execute(
            f"""
            CREATE TABLE IF NOT EXISTS htmls (
                id SERIAL PRIMARY KEY,
                country TEXT,
                html TEXT,
                date DATE
            );
            """
        )
        connection.commit()
        print("Table htmls created successfully")
    except OperationalError as e:
        print(f"The error '{e}' occurred")
    finally:
        if connection:
            cursor.close()
            connection.close()


def populate_queries_table():
    """Populate queries table with questions from runs/google_questions/{country}.txt.
    Each file has lines with "question\tdate".
    If the question-country pair already exists, we skip it.
    """
    connection = None
    try:
        connection, cursor = connect_to_db(dbname=DB_NAME)
        # There is one {country}.txt file for each country, we iterate over them:
        for file in os.listdir(QUESTIONS_DIR):
            if not file.endswith(".txt"):
                continue
            # if not file in ["questions_general.txt"]: # to filter by country
            #     continue
            country = file.split("_")[1].split(".")[0] # extract <country> from "questions_<country>.txt"

            logger.info(f"Populating queries for {country}...")

            logger.info(f"    Reading questions...")
            file_path = os.path.join(QUESTIONS_DIR, file)
            df_questions = read_questions(file_path)

            logger.info(f"    Preprocessing {len(df_questions)} questions...")
            df_questions = preprocess_questions_df(df_questions, country)
            logger.info(f"    Remaining questions: {len(df_questions)}")

            logger.info(f"    Checking for new questions...")
            cursor.execute(f"SELECT question FROM queries WHERE country = '{country}'")
            rows = cursor.fetchall()
            df_questions_db = pd.DataFrame(rows, columns=["question"]) 
            df_full = pd.merge(
                df_questions, df_questions_db, on=["question"], how="outer", indicator=True)
            df_to_add = df_full[df_full["_merge"] == "left_only"].drop(columns="_merge")
            df_to_add["country"] = country
            df_to_add = df_to_add[['question', 'country', 'date', 'url', 'relevance', 'relevance_rank']]

            logger.info(f"    Inserting {len(df_to_add)} new questions...")
            insert_df_to_table(cursor, df_to_add, "queries")
            connection.commit()
            logger.info(f"    {country}: populated successfully")

        logger.info("Queries table populated successfully")

    except OperationalError as e:
        print(f"The error '{e}' occurred")

    finally:
        if connection:
            cursor.close()
            connection.close()


def read_questions(file: str) -> pd.DataFrame:
    df_questions = pd.read_csv(
        file, sep="\t", header=None,
        names=["question", "date", "relevance", "relevance_rank"],
        dtype={
            "question": "str",
            "date": "str",
            "relevance": "Int64",
            "relevance_rank": "Int64",
        },
    )
    # replace any nan with None in "relevance" and "relevance_rank":
    df_questions[["relevance", "relevance_rank"]] = df_questions[
        ["relevance", "relevance_rank"]
    ].replace({pd.NA: None})
    return df_questions


def insert_df_to_table(cursor, df: pd.DataFrame, table_name: str):
    """Insert a pandas DataFrame into a table using psycopg2.extras.execute_values
    """
    cols = ", ".join(df.columns)
    query = f"INSERT INTO {table_name} ({cols}) VALUES %s"
    data = [tuple(row) for row in df.values]
    extras.execute_values(cursor, query, data, template='(%s, %s, %s, %s, %s::int, %s::int)')


def preprocess_questions_df(df: pd.DataFrame, country: str) -> pd.DataFrame:
    """
    """
    df = df.copy()
    # Remove "¿":
    df["question"] = df["question"].str.replace("¿", "")
    df = df.drop_duplicates(subset=["question"])
    # drop questions with 2 words or less:
    df = df[df["question"].str.count(" ") > 1]
    # drop questions with some patterns:
    df["text_lower"] = df["question"].str.lower()
    negative_patterns = (
        # ends with:
        " en wikipedia$", " la wikipedia$", " de wikipedia$", " letra$",
        " acordes$", " karaoke$", " meme$", "\\.com$", " traductor$", " traducir$",
        # starts with:
        "^http", "^www",
        # portuguese (all ç except for "barça"):
        " em ", "^em ", " o que ", "^o que ", "ã", " é ", " com ", r"(?<!bar)ç",
        " tem ", " nao ", " fazer "
    )
    df = df[~df["text_lower"].str.contains("|".join(negative_patterns), regex=True)]
    # Remove if question has 3 or less words followed by "wikipedia":
    pattern = r"^(\w+ ){0,3}wikipedia$"
    df = df[~df["text_lower"].str.match(pattern)]
    # Remove " wikipedia$" or " brainly$" from questions (because they are likely to be searches in either site):
    df["question"] = df["question"].str.replace(" wikipedia$", "")
    df["question"] = df["question"].str.replace(" brainly$", "")
    df = df.drop_duplicates(subset=["question"])
    # Remove duplicates considering accents:
    df = df.sort_values(["question", "date"])
    df["text_no_accents"] = df["text_lower"].apply(lambda x: unidecode(x))
    df = df.drop_duplicates(subset="text_no_accents", keep="last")
    df = df.drop(columns=["text_no_accents"])
    # Remove questions that are exactly query patterns:
    patterns = add_accented_patterns(QUERY_PATTERNS)
    df = df[~df["text_lower"].isin(patterns)]
    df = df.drop(columns=["text_lower"]).reset_index(drop=True)
    # add the search URL:
    df["url"] = df.apply(lambda x: build_search_request(x["question"], country), axis=1)
    return df


if __name__ == "__main__":
    main()
