"""Sources: 

* https://github.com/allenai/gooaq/blob/main/extraction/question_extraction/google_suggest.py
"""

import argparse
import logging
import json
import os
import re
import requests
import time
import urllib
from collections import Counter
from datetime import datetime
from pathlib import Path
from requests.exceptions import ChunkedEncodingError, ConnectionError, ReadTimeout

from tqdm import tqdm

from helpers import QUERY_PATTERNS, add_accented_patterns


logger = logging.getLogger()
logger.setLevel(logging.INFO)
fmt = logging.Formatter(
    "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
    "%m/%d/%Y %H:%M:%S",
)
console = logging.StreamHandler()
console.setFormatter(fmt)
logger.addHandler(console)


def main():
    # args:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outdir", type=str, required=True, help="Output directory")
    parser.add_argument("--country_code", type=str, required=True, help="Country code (see scripts/google_questions/helpers.py)")
    parser.add_argument("--min_words_to_add", type=int, default=1, help="Minimum number of words to add to each prefix")
    parser.add_argument("--max_words_to_add", type=int, default=29, help="Maximum number of words to add to each prefix")
    parser.add_argument("--max_results", type=int, default=None, help="Maximum number of results to retrieve")
    args = parser.parse_args()

    logger.info(f"Crawling questions for country {args.country_code}")
    crawl_questions(
        args.outdir, args.country_code,
        min_words_to_add=args.min_words_to_add, max_words_to_add=args.max_words_to_add,
        max_results=args.max_results
    )

    logger.info("Done!")


def crawl_questions(
    outdir: str, country_code: str, lang: str = "es", 
    min_words_to_add: int = 1, max_words_to_add: int = 29, max_results: int = None
):

    seed_prefixes = add_accented_patterns(QUERY_PATTERNS)

    logger.info("Loading previous results")
    country_results = read_previous_results(outdir, country_code, seed_prefixes)
    general_results = read_previous_results(outdir, "general", seed_prefixes)

    logger.info("Loading past queried prefixes from log files")
    past_queries = read_past_queries("logs", country_code)
    past_general_prefixes = read_past_general_prefixes("logs") # search prefixes where localization did not work

    def process_request(search_prefix):
        """Make a request and handle the results for a given search prefix
        """
        if search_prefix in past_general_prefixes:
            output = []
            localization_works = False
            print(f" ** '{search_prefix}': skipping (already tested localization does not work)")
        else:
            print(f" ** '{search_prefix}': requesting")
            output = search_with_retry(search_prefix, country_code, lang)
            # If search failed, output is None and the following will exit with error: 
            # 'NoneType' object is not iterable
            # --> this is desirable because it means API is failing systematically
            localization_works = test_if_localization_works(search_prefix, output)
        
        print(f" ** '{search_prefix}': {localization_works=}")
        outfile_name = (
            f"questions_{country_code}.txt"
            if localization_works
            else f"questions_general.txt"
        )
        results = country_results if localization_works else general_results

        for out in output:
            # out is a tuple (result, suggest_relevance, relevance_rank)
            if len(out[0]) < 15:
                print(f"   --> {out[0]}: too short! ❌ Dropping it")
                continue
            if out[0] not in results:
                print(f"   --> {out[0]}: is new! ✅ Adding it")
                results.append(out[0])
                add_item_to_file(out, f"{outdir}/{outfile_name}")  
            else:
                print(f"   --> {out[0]}: already have it! 🥱")


    logger.info("Iterating over query patterns")
    for seed_prefix in seed_prefixes:

        if seed_prefix in past_queries:
            logger.info(f"seed_prefix='{seed_prefix}': skipping (already queried)")
            continue
        else:
            logger.info(f"seed_prefix='{seed_prefix}': searching")
            past_queries.append(seed_prefix)

        for letter in [""] + [chr(i) for i in range(97, 123)] + ["ñ"]: 
            search_prefix = (seed_prefix + letter).strip() # NOTE strip() runs "que" instead of "que "
            process_request(search_prefix)

        logger.info(f"{len(country_results)=} {len(general_results)=}")


    logger.info("Iterating over country results adding words")
    for n in tqdm(range(min_words_to_add, max_words_to_add+1)):

        logger.info(f"Adding {n} words to each prefix in results")

        # sort results by freq of largest matchin prefix + n words
        country_results = sort_results_by_frequency(
            country_results, seed_prefixes, n)

        for result in country_results:
            
            # check if result starts with a seed prefix:
            matching_patterns = [p for p in seed_prefixes if re.match(rf"^(¿)?{p}", result)]
            if len(matching_patterns) == 0:
                logger.info(f"result='{result}': skipping (does not start with any seed prefix)")
                continue
            
            # extract prefix from result: longest matching pattern + n words
            longest_pattern = max(matching_patterns, key=len)
            result_words = result.split()
            pattern_words = longest_pattern.split()
            # check if result is too short:
            if len(result_words) < len(pattern_words) + n:
                logger.info(f"result='{result}': skipping (too short)")
                continue
            new_prefix = " ".join(result_words[:len(pattern_words) + n]) + " "

            if new_prefix in past_queries:
                logger.info(f"result='{result}': skipping (prefix '{new_prefix}' already queried)")
                continue
            else:
                logger.info(f"result='{result}': using")
                logger.info(f"new_prefix='{new_prefix}': searching")
                past_queries.append(new_prefix)

            for letter in [""] + [chr(i) for i in range(97, 123)] + ["ñ"]: 
                search_prefix = (new_prefix + letter).strip() # NOTE strip() runs "que es" instead of "que es "
                process_request(search_prefix)

            logger.info(f"{len(country_results)=} {len(general_results)=}")

            # Stop everything if we reach the max number of results:
            if max_results is not None and len(country_results) >= max_results:
                logger.info(f"Reached max_results={max_results}. Stopping.")
                break
        else:
            continue
        break


def read_previous_results(path: str, country_code: str, seed_prefixes: list) -> list:
    outfile = f"{path}/questions_{country_code}.txt"
    all_results = []
    if os.path.isfile(outfile):
        with open(outfile, encoding="utf-8") as f:
            for line in f:
                all_results.append(line.split("\t")[0]) # each line has question\tdate
    return list(set(all_results))


def extract_prefix(result: str, patterns: list, n_words: int) -> str:
    matching_patterns = [p for p in patterns if re.match(rf"^(¿)?{p}", result)]
    if len(matching_patterns) == 0:
        return result
    # extract prefix from result: longest matching pattern + n words
    longest_pattern = max(matching_patterns, key=len)
    result_words = result.split()
    pattern_words = longest_pattern.split()
    # check if result is too short:
    if len(result_words) < len(pattern_words) + n_words:
        return result
    prefix = " ".join(result_words[:len(pattern_words) + n_words]) + " "
    return prefix


def sort_results_by_frequency(results: list, patterns: list, n_words: int) -> list:
    # extract the longest matching pattern + n_words words of each result:
    prefixes = [extract_prefix(result, patterns, n_words) for result in results]
    prefixes_counts = Counter(prefixes)
    sorted_prefixes = sorted(
        results, key=lambda x: prefixes_counts[extract_prefix(x, patterns, n_words)],
        reverse=True)
    return sorted_prefixes


def search_with_retry(search_prefix, country_code, lang="es", max_retries=10, delay=60):
    for attempt in range(max_retries):
        try:
            output = query_and_return(search_prefix, country_code, lang=lang)
            return output
        except (ConnectionError, ChunkedEncodingError, ReadTimeout) as e:
            logger.error(f"requests error: {e}")
            print(f"Attempt {attempt+1}/{max_retries}: Waiting {delay} seconds and trying again...")
            time.sleep(delay)
    else:
        print(f"Maximum attempts ({max_retries}) reached. Giving up.")
        return None


def read_past_queries(logs_dir: str, country_code: str) -> list:
    """Extract all queries from logs/questions_{country_code}_*.log with the
    following format:
    "prefix='{prefix}': searching"
    """
    past_queries = []
    log_files = list(Path(logs_dir).glob(f"questions_{country_code}_*.log"))
    if len(log_files) == 0:
        print(f"No log files in {logs_dir} for country {country_code}")
    else:
        pattern = rf"prefix='(.+?)': searching"
        for f in log_files:
            print(f"Reading {f}")
            with open(f, "r") as f:
                text = f.read()
            past_queries.extend(re.findall(pattern, text))
        past_queries = list(set(past_queries))
    print(f"Found {len(past_queries)} past queries")
    return past_queries


def read_past_general_prefixes(logs_dir: str) -> list:
    """Extract all queries from logs/questions_*_*.log with the
    following format:
    "** 'query': localization_works=False"
    """
    past_queries = []
    log_files = list(Path(logs_dir).glob("questions_*.log"))
    if len(log_files) == 0:
        print(f"No log files in {logs_dir}")
    else:
        pattern = r"\*\* '(.+?)': localization_works=False"
        for f in log_files:
            print(f"Reading {f}")
            with open(f, "r") as f:
                text = f.read()
            past_queries.extend(re.findall(pattern, text))
        past_queries = list(set(past_queries))
    print(f"Found {len(past_queries)} past prefixes where localization did not work")
    return past_queries


def build_request(prefix: str, country_code: str, lang: str = "es") -> str:
    formatted_query = urllib.parse.quote_plus(prefix)
    # NOTE the results of both urls is the same, but the second one seems much slower sometimes
    url = f'https://www.google.com/complete/search?client=chrome&q={formatted_query}&gl={country_code}&hl={lang}'
    # url = f'http://suggestqueries.google.com/complete/search?client=chrome&q={formatted_query}&gl={country_code}&hl={lang}'
    return url


def query_and_return(prefix: str, country_code: str, lang: str = "es") -> list:
    """NOTE using random headers with fake_useragent yields empty results sometimes
    """
    time.sleep(0.2)
    url = build_request(prefix, country_code, lang=lang)
    headers = {'User-Agent': "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/121.0.0.0 Safari/537.36 Edg/121.0.0.0"}
    r = requests.get(url, headers=headers)
    if r.status_code == 200:
        # save it
        texts = json.loads(r.text)
        results = texts[1]
        suggest_relevance = texts[4]['google:suggestrelevance'] if results else []
        relevance_rank = list(range(1, len(suggest_relevance)+1)) if suggest_relevance else []
        # output: list of tuples (result, suggest_relevance, relevance_rank)
        output = list(zip(results, suggest_relevance, relevance_rank))
        return output
    elif r.status_code >= 500:
        raise requests.exceptions.ConnectionError(f"Status code {r.status_code}: {r.text}")
    else:
        raise Exception(f"Status code {r.status_code}: {r.text}")


def test_if_localization_works(prefix: str, api_output: list) -> bool:
    """If the results of the API are the same as those in a non-Spanish location
    (e.g France), then the localization (parameter gl=) is not working.
    """
    if len(api_output) == 0:
        return False
    france_output = search_with_retry(prefix, "fr")
    # keep the results only (not the relevance and rank):
    api_results = [o[0] for o in api_output]
    france_results = [o[0] for o in france_output]
    return set(api_results) != set(france_results)


def add_item_to_file(api_output: tuple, outfile: str):
    current_date = datetime.now().strftime("%Y-%m-%d")
    question, suggest_relevance, relevance_rank = api_output
    with open(outfile, "a", encoding="utf-8") as f:
        f.write(f"{question}\t{current_date}\t{suggest_relevance}\t{relevance_rank}\n")


if __name__ == "__main__":
    main()
