
import logging
import os
import urllib

import psycopg2


def add_accented_patterns(patterns: list) -> list:
    """Add versions with accents (e.g. quien -> quién, etc.) removing duplicates"""
    replacement_dict = {
        "que": "qué",
        "cual": "cuál",
        "quien": "quién", 
        "cuanto": "cuánto",
        "cuanta": "cuánta",
        "cuando": "cuándo",
        "donde": "dónde",
        "como": "cómo",
        "segun": "según",
    }
    final_patterns = []
    for pattern in patterns:
        final_patterns.append(pattern)
        for key, value in replacement_dict.items():
            final_patterns.append(pattern.replace(key, value))
    return list(set(final_patterns))
    # return sorted(list(set(final_patterns)))


def build_search_request(
        query: str, country_code: str,
        query_suffix: str = "wikipedia", lang: str = "es"
) -> str:
    domain = DOMAINS_DICT[country_code]
    formatted_query = urllib.parse.quote_plus(query + " " + query_suffix)
    # if query already ends with query_suffix, do not add it:
    if query.endswith(query_suffix):
        formatted_query = urllib.parse.quote_plus(query)
    url = f'http://www.google.{domain}/search?q={formatted_query}&gl={country_code}&hl={lang}'
    return url


def connect_to_db(dbname="retrieval"):
    """Source: https://github.com/allenai/gooaq/tree/main/extraction/answer_extraction
    """
    host = 'localhost' # IP to your database
    conn = psycopg2.connect(
        host=host,
        port="5432", # port to your DB
        dbname=dbname, # DB name
        user='postgres', # username for your database
        password=os.getenv('DB_PASSWORD')) # password to your database
    cur = conn.cursor()

    return conn, cur


def set_logger(level: str = "INFO") -> logging.Logger:
    logger = logging.getLogger()
    logger.setLevel(level)
    fmt = logging.Formatter(
        "[%(asctime)s] {%(filename)s:%(lineno)d} %(levelname)s - %(message)s",
        "%m/%d/%Y %H:%M:%S",
    )
    console = logging.StreamHandler()
    console.setFormatter(fmt)
    logger.addHandler(console)
    return logger


DOMAINS_DICT = {
    # source: https://www.google.com/supported_domains
    # sorted by speakers: https://es.wikipedia.org/wiki/Distribuci%C3%B3n_geogr%C3%A1fica_del_idioma_espa%C3%B1ol
    'mx': 'com.mx',
    'us': 'com',
    'co': 'com.co',
    'es': 'es',
    'ar': 'com.ar',
    'pe': 'com.pe',
    've': 'co.ve',
    'cl': 'cl',
    'gt': 'com.gt',
    'ec': 'com.ec',
    'cu': 'com.cu',
    'bo': 'com.bo',
    'do': 'com.do',
    'hn': 'hn',
    'sv': 'com.sv',
    'py': 'com.py',
    'ni': 'com.ni',
    'cr': 'co.cr',
    'pa': 'com.pa',
    'pr': 'com.pr',
    'uy': 'com.uy',
    'general': 'com', # Same as US for queries that are not country-specific
    # 'gq': None # Equatorial Guinea has no google domain
    # Other countries with >1M speakers according to wiki link:
    # morocco, brazil, italy, france.
    # And it is a significant language in: andorra, gibraltar, belize.
}


QUERY_PATTERNS = [
    # NOTE space before words means that they will be used as patterns to search
    # with words before or after them
    # basicas:
    "que ",
    "cual ",
    "cuales ",
    "quien ",
    "quienes ",
    "cuanto ", 
    "cuanta ", 
    "cuantos ", 
    "cuantas ",
    "cuando ",
    "donde ",
    "como ",
    "porque ",
    # otras:
    "adonde "
    "debe ",
    "deberia ",
    "deberian ",
    "puede ",
    "pueden ",
    "podria ",
    # con multiples palabras:
    "por que ",
    "razones por las que ",
    "razones para ",
    "razones de ",
    "razones del ",
    "razones de por que ",
    "buenas razones para ",
    "motivos por los que ",
    "motivos para ",
    "motivos de ",
    "motivos del ",
    "motivos de por que ",
    "buenos motivos para ",
    "pros y contras de ",
    "por que debe ",
    "por que deberia ",
    "por que no ",
    # preposition + question word:
    "a que ",
    "a cual ",
    "a cuales ",
    "a quien ",
    "a quienes ",
    "a cuantos ",
    "a cuantas ",
    "a donde ",
    "ante que ",
    "ante cual ",
    "ante cuales ",
    "ante quien ",
    "ante quienes ",
    "bajo que ",
    "bajo cual ",
    "bajo cuales ",
    "con que ",
    "con cual ",
    "con cuales ",
    "con quien ",
    "con quienes ",
    "con cuanto ",
    "con cuanta ",
    "con cuantos ",
    "con cuantas ",
    "de que ",
    "de cual ",
    "de quien ",
    "de quienes ",
    "de cuanto ",
    "de cuanta ",
    "de cuantos ",
    "de cuantas ",
    "de donde ",
    "desde que ",
    "desde donde ",
    "durante que ",
    "durante cual ",
    "durante cuales ",
    "durante cuanto ",
    "durante cuantos ",
    "en que ",
    "en cual ",
    "en cuales ",
    "en quien ",
    "en cuanto ",
    "en cuantos ",
    "en cuantas ",
    "en donde ",
    "entre que ",
    "entre cuales ",
    "entre quienes ",
    "hacia que ",
    "hacia cual ",
    "hacia quien ",
    "hasta que ",
    "hasta cual ",
    "hasta cuanto ",
    "hasta cuanta ",
    "hasta cuantos ",
    "hasta cuantas ",
    "hasta cuando ",
    "hasta donde ",
    "mediante que ",
    "mediante cual ",
    "mediante cuales ",
    "para que ",
    "para cual ",
    "para quien ",
    "para quienes ",
    "para cuantos ",
    "para cuantas ",
    "para donde ",
    "por cuanto ",
    "por cuanta ",
    "por cuantos ",
    "por cuantas ",
    "por donde ",
    "segun quien ",
    "segun quienes ",
    "sobre que ",
    "sobre cual ",
    "sobre quien ",
    "sobre quienes ",
    "tras cuanto ",
    "tras cuantos ",
    "tras cuantas ",
]

