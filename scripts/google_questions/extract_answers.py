"""Source: https://github.com/allenai/gooaq/tree/main/extraction/answer_extraction

We currently manually edit the WHERE clauses in the SQL queries to improve
specific extractions, and increase the VERSION manually; this is not ideal.

Some types found are:
"conversión de hora local", "conversor de unidades", "cómo llegar",
"resultados adicionales", "resultados locales", "avisos sobre resultados filtrados",
"resultados de la web", "resultado web con vínculos a sitio", "conversor de divisas", 
"avisos sobre resultados filtrados", "resultado de fechas",
"resultado de pronóstico del clima", "resultados deportivos", "resultados de twitter",
"resultados en el mapa"
"""

import copy
import multiprocessing

from bs4 import BeautifulSoup
from tqdm import tqdm
from psycopg2 import sql

from helpers import connect_to_db, set_logger


VERSION = 6 # increment version to go through pages we are uncertain about again
BATCH_SIZE = 80
MAX_WORKERS = 12


logger = set_logger()


def main():
    ids_to_parse = get_ids_to_parse()
    n_ids = len(ids_to_parse)
    n_batches = n_ids // BATCH_SIZE

    logger.info(f"Found {n_ids} IDs to parse -- running in {n_batches} batches of size {BATCH_SIZE}")

    with multiprocessing.Pool(processes=MAX_WORKERS) as pool, tqdm(total=n_batches) as pbar:
        for _ in pool.imap_unordered(do_batch, [ids_to_parse[i:i+BATCH_SIZE] for i in range(0, n_ids, BATCH_SIZE)]):
            pbar.update(1)

    logger.info("DONE!")


def get_ids_to_parse() -> list:
    conn, cur = connect_to_db() 
    cur.execute(
        f"""
        SELECT h.id
        FROM htmls AS h
            LEFT JOIN extractions AS e ON h.id = e.id
        WHERE h.html IS NOT NULL
            AND (e.extract_v < {VERSION} OR e.extract_v IS NULL)
            AND e.answer IS NULL
            AND e.short_answer IS NULL
            AND e.answer_type IS NULL
            --AND e.answer_type IN ('feat_snip', 'descript', 'knowledge', 'rich_list', 'rich_set', 'rich_snip')
            --AND e.answer LIKE '%›%'
        ;"""
    )
    res = cur.fetchall()
    ids = [x[0] for x in res]
    cur.close()
    conn.close()
    return ids


def do_batch(ids: list) -> None:
    conn, cur = connect_to_db()

    ids_str = ', '.join([str(x) for x in ids])
    cur.execute(
        f"""
        SELECT h.id, html
        FROM htmls AS h
            LEFT JOIN extractions AS e ON h.id = e.id
        WHERE h.html IS NOT NULL AND h.id IN ({ids_str})
        FOR UPDATE OF h SKIP LOCKED
        LIMIT {BATCH_SIZE};"""
    )
    # LIMIT {BATCH_SIZE} just in case, should be redundant
    rows = cur.fetchall()

    # If no more htmls to fetch, end (should not happen)
    if not rows:
        return

    data_to_insert = []
    for id, html in rows:
        extraction_type = None
        short_answer = None
        long_answer = None
        url = None

        doc = BeautifulSoup(html, 'html.parser')

        featured = doc.h2
        featured_type = featured.get_text().lower() if featured else None # casing might be inconsistent, so just always lowercase

        expanded_search = check_expanded_search(doc)

        try:
            if featured_type == 'fragmento destacado de la web':
                snippet = featured.parent.div
                url = get_url_from_featured_snippet(snippet)
                extraction_type, short_answer, long_answer = handle_featured_snippet(snippet)
            elif featured_type == 'conversor de unidades':
                extraction_type, short_answer, long_answer = 'unit_conv', "", None
                # extraction_type, short_answer, long_answer = handle_unit_converter(featured, question)
            elif featured_type == 'conversor de divisas':
                extraction_type, short_answer, long_answer = handle_currency_converter(featured)
            elif featured_type == 'translation result':
                extraction_type, short_answer, long_answer = handle_translation_result(featured)
            elif featured_type == 'resultados locales':
                extraction_type, short_answer, long_answer = handle_local_results(featured)
            elif featured_type == 'conversión de hora local':
                extraction_type, short_answer, long_answer = handle_local_time_conversion(featured)
            elif featured_type == 'hora local':
                extraction_type, short_answer, long_answer = handle_local_time(featured)
            elif featured_type == 'resultado de pronóstico del clima':
                extraction_type, short_answer, long_answer = handle_weather(featured)
            elif featured_type == 'cómo llegar':
                extraction_type, short_answer, long_answer = handle_directions(featured)
            # elif featured_type == 'description':
            #     extraction_type, short_answer, long_answer = handle_description(featured)
            elif featured_type == 'overview':
                extraction_type, short_answer, long_answer = handle_overview(doc)
            elif featured_type == 'resultados deportivos':
                extraction_type, short_answer, long_answer = handle_sports_results(featured)
            elif featured_type in ['resultados en el mapa', 'resultados de mapa']:
                extraction_type, short_answer, long_answer = handle_map_results(featured)
            elif featured_type == 'resultados de twitter':
                extraction_type, short_answer, long_answer = handle_twitter(featured)
            elif featured_type == 'resultado de fechas':
                extraction_type, short_answer, long_answer = handle_twitter(featured)
            else:
                extracted_data = handle_other_answer_markers(doc)
                if extracted_data:
                    extraction_type, short_answer, long_answer, url = extracted_data
                elif (
                    featured_type == 'resultados de la web' or
                    featured_type == 'resultados web' or
                    featured_type == 'people also ask' or
                    featured_type == 'avisos sobre resultados filtrados' or
                    featured_type == 'resultado web con vínculos a sitio' or
                    featured_type == 'resultado web con enlaces de sitio' or
                    featured_type == 'resultados complementarios' or
                    featured_type is None
                ):
                    # e.g. 'avisos sobre resultados filtrados': 91481 117567
                    extraction_type, short_answer, long_answer = handle_no_snippet(featured)
                elif featured_type == 'resultados adicionales':
                    extraction_type, short_answer, long_answer = handle_other_results(featured)
                else:
                    logger.warning(f'        Unknown feature in {id}: "{featured_type}"')
        except Exception as e:
            logger.error(f'Extraction for {id} failed: {e}')
            continue

        data_to_insert.append((id, short_answer, long_answer, url, extraction_type, VERSION, expanded_search))
    
    insert_query = sql.SQL("""
            INSERT INTO extractions (id, short_answer, answer, answer_url, answer_type, extract_v, expanded_search)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
            ON CONFLICT (id)
            DO UPDATE
              SET
                short_answer = EXCLUDED.short_answer,
                answer = EXCLUDED.answer,
                answer_url = EXCLUDED.answer_url,
                answer_type = EXCLUDED.answer_type,
                extract_v = EXCLUDED.extract_v,
                expanded_search = EXCLUDED.expanded_search
        ;""")
    cur.executemany(insert_query, data_to_insert)
    conn.commit()
    cur.close()
    conn.close()


def check_expanded_search(doc):
    """Check if doc says "se incluyen resultados de..." because this might change the results unexpectedly
    e.g. 131561:
    "Se incluyen resultados de quién descubrió la penicilina wikipedia
    Buscar solo quién descubrió la ampicilina wikipedia"
    """
    clarification = doc.p
    clarification_text = clarification.get_text().lower() if clarification else None
    expanded_search = True if clarification_text and 'se incluyen resultados de' in clarification_text else False
    return expanded_search


def handle_featured_snippet(snippet):
    # We'll consider fragments highlighted in blue (if any) in the
    # featured snippet to be the short answer, and the whole snippet to be the
    # long answer.
    long_div = snippet.find('div', attrs={'role': 'heading'}) # short answer is the first <b> tag in the snippet
    short_answer_b = snippet.find('b')
    short_answer = short_answer_b.get_text() if short_answer_b else None
    if long_div and long_div.span:
        long_answer = long_div.span.get_text()
        if "›" in long_answer:
            # Sometimes there are uncomfortable "links" in the text, indicated by "›", whose text we remove.
            # e.g. 4851852 402574 4966871
            # There are some false positives where this processing should be neutral e.g. 4641988
            long_div_copy = copy.copy(long_div)
            for div in long_div_copy.find_all('div'):
                div.decompose()
            long_answer = long_div_copy.get_text()
        if short_answer and "›" in short_answer:
            short_answer_b_copy = copy.copy(short_answer_b)
            for div in short_answer_b_copy.find_all('div'):
                div.decompose()
            short_answer = short_answer_b_copy.get_text()            
        return 'feat_snip', short_answer, long_answer
    else:
        ol = snippet.find('ol')
        ul = snippet.find('ul')
        # Usually when google provides an answer as a "title"
        headings = snippet.find_all('div', attrs={'role': 'heading'})
        if len(headings) > 1:
            # keep the longest text from headings:
            idx_longest = max(range(len(headings)), key=lambda i: len(headings[i].get_text()))
            longest_heading = headings[idx_longest]
            long_answer = longest_heading.get_text()
            if "›" in long_answer:
                # e.g. 112741, 45252
                heading_copy = copy.copy(longest_heading)
                for div in heading_copy.find_all('div'):
                    div.decompose()
                long_answer = heading_copy.get_text()
            return 'feat_snip', short_answer, long_answer
        elif ol and not ol.has_attr('role'):
            long_list = [x.get_text() for x in ol.find_all('li')]
            return 'rich_list', short_answer, str(long_list)
        elif ul:
            # NOTE in rich_set the short_answer is usually a header for the list, 
            # and the long_answer is the list itself e.g. 32714 4344997 
            # Sometimes not: e.g. 4338696 4426039 
            long_list = [x.get_text() for x in ul.find_all('li')]
            return 'rich_set', short_answer, str(long_list)
        else:
            # Usually stuff from wiki side boxes e.g. 59528
            return 'rich_snip', short_answer, None


def get_url_from_featured_snippet(snippet):
    r_div = snippet.find('div', attrs={'class': 'g'})
    if r_div:
        return r_div.a['href']
    else:
        links = snippet.find_all('a')
        if links:
            return links[0]['href']


def get_url_from_description(div):
    a_tag = div.find("a")
    if a_tag:
        return a_tag['href']
    return None


def get_url_from_kp_header(div):
    next_div = div.find_next_sibling()
    url_tag = next_div.find("a") if next_div else None
    if url_tag:
        return url_tag['href']
    return None


def get_split(question, delimiter):
    split = question.split(delimiter)
    if len(split) == 2 and len(split[1]) > 0:
        return split[1]


def handle_unit_converter(featured, question):
    equals = featured.parent.div(text='=')[0]
    count = equals.find_next('input')
    count_value = count.get('value')
    unit = count.find_next('option', {'selected': '1'})
    unit_value = ''
    if unit:
        unit_value = unit.get_text()
    else: # see 13783 and 19581
        unit_value = get_split(question, ' how many ')
        if unit_value is None:
            unit_value = get_split(question, ' equal to ')

    short_answer = count_value # sometimes it's just PEBKAC and no units available; see 20802
    if unit_value:
        short_answer = '{0} {1}'.format(count_value, unit_value)
    return 'unit_conv', short_answer, None


def handle_currency_converter(featured):
    input = featured.parent.find('select')
    count = input.find_next('input')
    count_value = count.get('value')
    unit = count.find_next('option', {'selected': '1'})
    unit_value = unit.get_text()
    short_answer = '{0} {1}'.format(count_value, unit_value)
    return 'curr_conv', short_answer, None


def handle_translation_result(featured):
    # todo: 8349104 as an example of one with another result
    short_answer = featured.parent.find('pre', {'id': 'tw-target-text'}).get_text()
    return 'tr_result', short_answer, None


def handle_local_results(featured):
    return 'local_rst', None, None


def handle_local_time_conversion(featured):
    short_answer = featured.parent.find('div', {'class': 'vk_bk'}).get_text()
    return 'time_conv', short_answer, None


def handle_local_time(featured):
    # strip because sometimes there's whitespace at the end due to div spacing
    short_answer = featured.parent.find('div', {'class': 'vk_bk'}).get_text().strip()
    return 'localtime', short_answer, None


def handle_weather(featured):
    return 'weather', None, None


def handle_twitter(featured):
    return 'twitter', None, None


def handle_date(featured):
    snippet = featured.parent.div
    heading = snippet.find("div", {"role": "heading"})
    texts = [div.get_text() for div in heading.div.find_all("div", recursive=False)]
    short_answer = texts[0]
    long_answer = texts[0] + " - " + texts[1] if len(texts) > 1 else None
    return 'date', short_answer, long_answer


def handle_kp_header(header):
    gsrt = header.find('div', {'class': 'gsrt'})
    if gsrt:
        short_answer = gsrt.div.get_text()
        # to get long_answer, get the div immediately after kp-header, at the same level:
        next_div = header.find_next_sibling()
        long_answer = None
        if next_div and next_div.span:
            txt = next_div.span.get_text()
            long_answer = None if txt.strip() == "" else txt
        # If gsrt has more than one div with text, get the 2nd one e.g. 592918:
        if long_answer is None and len(gsrt.find_all("div", recursive=False)) > 1:
            long_answer = gsrt.find_all("div", recursive=False)[1].get_text()
        # long_answer can be None or "" if there's no chunk e.g. 5255566 4500595 592725
        # example of not None: 33919, 97461
        if "›" in long_answer:
            # Sometimes there are uncomfortable "links" in the text, indicated by "›", whose text we remove.
            # e.g. 4366464 492024
            span_copy = copy.copy(next_div.span)
            for div in span_copy.find_all('div'):
                div.decompose()
            long_answer = span_copy.get_text()
        return 'knowledge', short_answer, long_answer
    else:
        return None, None, None


def handle_directions(featured):
    return 'direction', None, None


def handle_description(div):
    header = div.find('h3')
    header_name = header.get_text() if header else None
    if header_name == 'Descripción':
        long_answer = div.span.get_text()
        return 'descript', None, long_answer
    else:
        return None, None, None


def handle_overview(doc):
    short_ans = doc.a.get_text()
    return 'overview', short_ans, None


def handle_sports_results(featured):
    return 'sports', None, None


def handle_map_results(featured):
    return 'map', None, None


def handle_no_snippet(featured):
    # todo: 1119248 and 8349104 as examples of incorrect no_answer extractions
    return 'no_answer', None, None


def handle_other_results(featured):
    snippet = featured.parent.div
    health_info = snippet.find("a", href=lambda x: x and "p=medical_conditions" in x)
    if health_info:
        # e.g. 113885 134834 
        return 'other_health', None, None
    # restaurants and other places e.g. 81105 73538 73639 120816 (data-maindata contains "LOCAL_NAV")
    # hotels e.g. 76454 (data-maindata contains "HOTELS")
    # other things e.g. 131821 ("ver resultados de:")
    return 'other', None, None


def handle_other_answer_markers(doc):
    answered_div = doc.find('div', {'class': 'answered-question'})
    if answered_div:
        featured = doc.h2
        snippet = featured.parent.div
        url = get_url_from_featured_snippet(snippet)
        extraction_type, short_answer, long_answer = handle_featured_snippet(answered_div)
        # NOTE to inspect these:
        logger.warning(f"        class='answered-question' in doc with url: {url}")
        return extraction_type, short_answer, long_answer, url
    
    kp_header = doc.find('div', {'class': 'kp-header'})
    if kp_header:
        extraction_type, short_answer, long_answer = handle_kp_header(kp_header)
        if extraction_type is not None:
            url = get_url_from_kp_header(kp_header) if long_answer is not None else None
            return extraction_type, short_answer, long_answer, url
    
    description = doc.find('div', {'data-attrid': 'description'})            
    # Maybe safer: description = featured.find_next('div', {'data-attrid': 'description'}) 
    if description:
        extraction_type, short_answer, long_answer = handle_description(description)
        if extraction_type == 'descript':
            url = get_url_from_description(description)
            return extraction_type, short_answer, long_answer, url
    
    return None


if __name__ == '__main__':
    main()
