"""
We gather multiple json files and insert them at once instead of file by file
bsz=500: 1h20m approx
bsz=100: 55m approx
"""


import argparse
import gzip
import json
from pathlib import Path
from typing import List

from psycopg2 import sql, extras
from tqdm import tqdm

from helpers import connect_to_db, set_logger


logger = set_logger()


def main():
    parser = argparse.ArgumentParser(
        description="Insert HTML data from json.gz files into PostgreSQL table."
    )
    parser.add_argument("--input_dir", required=True, type=str,
                        help="Directory containing list of dicts with HTML data in json.gz")
    parser.add_argument("--batch_size", type=int, default=100,
                        help="Number of json.gz files to process and insert at once")
    args = parser.parse_args()

    logger.info("Connecting to database...")
    connection, cursor = connect_to_db(dbname="retrieval")

    input_files = list(Path(args.input_dir).glob("*.json.gz"))
    total_files = len(input_files)
    if total_files == 0:
        logger.error(f"No json.gz files found in {args.input_dir}")
        return
    logger.info(f"Found {total_files} json.gz files")

    logger.info(f"Inserting data in batches of size {args.batch_size}...")
    for i in tqdm(range(0, total_files, args.batch_size), unit="batch"):

        batch_files = input_files[i:i+args.batch_size]
        batch_data = gather_batch_data(batch_files)
        insert_data(connection, cursor, batch_data)

    cursor.close()
    connection.close()

    logger.info("DONE!")


def gather_batch_data(files: List[str]) -> list:
    batch_data = []
    for file in files:
        # get country code from file name:
        country_code = file.stem.split("_")[0] # files are like "es_htmls_123.pkl"
        if country_code == "htmls":
            country_code = ""
        # read file content: list of dicts
        try:
            with gzip.open(file, 'rt') as f:
                data = json.loads(f.read())
        except Exception as e:
            logger.error(f"Error reading file {file}: {e}")
            continue
        # append as tuple to batch_data:
        for item in data:
            batch_data.append((item["id"], country_code, item["html"], item["date"]))
    return batch_data


def insert_data(connection, cursor, data: List[tuple]) -> None:
    insert_query = sql.SQL("""
        INSERT INTO htmls (id, country, html, date)
        VALUES %s
        ON CONFLICT (id) DO NOTHING;
    """)
    extras.execute_values(cursor, insert_query, data)
    connection.commit()


if __name__ == "__main__":
    main()
