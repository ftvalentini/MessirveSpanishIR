"""Given the data in the PostgreSQL queries table with cols [id, question, country, date] 
create a list of URLs to be scraped in the files urls/<country>.txt 
with <id>\t<url> in each line.
"""

import logging
from pathlib import Path

import pandas as pd

from helpers import connect_to_db, set_logger


OUTDIR = "runs/google_questions/urls"


logger = set_logger()


def main():

    logging.info("Fetching queries that need to be scraped from db...")
    connection, cursor = connect_to_db()
    cursor.execute("""
                   SELECT q.id, question, q.country, q.url
                   FROM queries q LEFT JOIN htmls h ON q.id = h.id
                   WHERE h.html IS NULL;"""
                   )
    queries = cursor.fetchall()
    connection.close()

    df_queries = pd.DataFrame(queries, columns=["id", "question", "country", "url"])

    # # Check if weird characters are being encoded correctly:
    # df_tmp = df_queries.loc[df_queries["question"] == "qué quiere decir 💙"].copy()
    # print(df_tmp)
    
    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    
    country_groups = df_queries.groupby("country")
    for country, group in country_groups:
        
        n_rows = group.shape[0]
        logging.info(f"Creating file for {country} with {n_rows} rows...")
        
        outfile = Path(OUTDIR) / f"urls_{country}.txt"
        with open(outfile, "w") as f:
            for _, row in group.iterrows():
                f.write(f"{row['id']}\t{row['url']}\n")
        
        logging.info(f"File created: {outfile}")

    logging.info(f"Done!")


if __name__ == "__main__":
    main()
