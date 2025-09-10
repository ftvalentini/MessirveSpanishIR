
import gzip
import json
import logging
import os
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List

import requests
from bs4 import BeautifulSoup
from fake_useragent import UserAgent
from googleapiclient import discovery
from googleapiclient.http import MediaFileUpload, MediaIoBaseDownload
from google.oauth2 import service_account


RUN_TEST = False


logger_level = "DEBUG" if RUN_TEST else "INFO"
logger = logging.getLogger()
logger.setLevel(logger_level)

# NOTE some URLs with featured snippets:
    # (1, "https://www.google.com/search?q=el+tomate+es+una+fruta+o+un+vegetal+wikipedia&gl=us&hl=es"),
    # (2, "https://www.google.com/search?q=por+que+el+cielo+es+azul+wikipedia&gl=us&hl=es"),


def handler(event, context):

    MAX_THREADS = event.get("max_threads", 15) # n of concurrent requests
    SLEEP_INTERVAL = event.get("sleep_interval", 15) # sleep every n requests
    MAX_SECONDS_WAIT = event.get("max_seconds_wait", 5) # max seconds to wait when sleeping
    MAX_URLS_TO_PARSE = event.get("max_urls_to_parse", 200)
    MIN_SUCCESS_RATE = event.get("min_success_rate", 0.0)

    input_file_name = None
    if RUN_TEST:
        INPUT_FOLDER_ID = event.get("input_folder_id", None) # If None, will read test_urls.json
        OUTPUT_FOLDER_ID = event["output_folder_id"]
    else:
        INPUT_FOLDER_ID = os.environ.get("INPUT_FOLDER_ID")
        OUTPUT_FOLDER_ID = os.environ.get("OUTPUT_FOLDER_ID")
    
    file_prefix = ""
    if INPUT_FOLDER_ID is not None:
        input_file_name = event.get("InputFile")
        file_prefix = input_file_name.split("_")[0] + "_" # should be country code

    SERVICE_ACCOUNT_KEY_FILE = 'google_service_account_key.json'

    logger.debug("Authenticating with Google Drive")
    drive_service = get_google_drive_service(SERVICE_ACCOUNT_KEY_FILE)

    if INPUT_FOLDER_ID is None:
        # run a test with some URLs:
        with open("test_urls.json", "r") as f:
            urls = json.load(f)
        urls_to_parse = urls[:MAX_URLS_TO_PARSE]
    else:
        try:
            input_file_id = download_file(drive_service, INPUT_FOLDER_ID, input_file_name)
        except FileNotFound as e:
            logger.error(f"File not found: {e}")
            return {
                "statusCode": 404,
                "success_pct": 0.0,
                "message": f"File not found: {e}"
            }
        except Exception as e:
            logger.error(f"Failed to download file: {e}")
            return {
                "statusCode": 500,
                "success_pct": 0.0,
                "message": f"Failed to download file: {e}"
            }
        urls_to_parse = read_json_file(input_file_name)[:MAX_URLS_TO_PARSE]

    with ThreadPoolExecutor(max_workers=MAX_THREADS) as executor:
        
        futures = [executor.submit(parse_page, d["id"], d["url"]) for d in urls_to_parse]
        results = []
        request_count = 0
        for future in as_completed(futures):
            try:
                result = future.result()
                results.append(result)
                request_count += 1
                if request_count % SLEEP_INTERVAL == 0:
                    wait_time = random.randint(1, MAX_SECONDS_WAIT)
                    logger.debug(f"Sleeping for {wait_time} seconds")
                    time.sleep(wait_time)
            except Exception as e:
                logger.warning(f"An error occurred: {e}")
                logger.info("All remaining tasks will be cancelled")
                for future in futures:
                    future.cancel()
                break

    successful_results = [res for res in results if res["success"]]
    failed_results = [res for res in results if not res["success"]]
    remaining_urls = [
        {"id": d["id"], "url": d["url"]} for d in urls_to_parse if d["url"] not in [res["url"] for res in results]
    ]

    logger.info(
        f"Successful: {len(successful_results)} / {len(urls_to_parse)}" 
        f" | Not processed (because of 429): {len(remaining_urls)}"
        f" | Failed (without target element): {len(failed_results)}"
    )

    # if the success rate was 0, raise an exception:
    success_percentage = round(len(successful_results) / len(urls_to_parse) * 100, 1)
    if len(successful_results) / len(urls_to_parse) <= MIN_SUCCESS_RATE:
        logger.warning(f"Exiting: out of {len(urls_to_parse)} URLs: "
                          f"{len(remaining_urls)} not processed, {len(failed_results)} failed")
        return {
            "statusCode": 400,
            "success_pct": success_percentage,
            "message": f"Exiting: out of {len(urls_to_parse)} URLs: "
                          f"{len(remaining_urls)} not processed, {len(failed_results)} failed"
        }

    output = create_output(results)

    logger.debug("Uploading output to Google Drive")
    output_file_name = f"{file_prefix}htmls_{results[0]['id']}.json.gz" # we use the id of the first result to name the file
    try:
        upload_zip_to_drive(drive_service, output, output_file_name, OUTPUT_FOLDER_ID)
    except Exception as e:
        logger.error(f"Failed to upload output file to Drive: {e}")
        return {
            "statusCode": 500,
            "success_pct": success_percentage,
            "message": f"Failed to upload output file: {e}"
        }
    else:
        logger.info(f"File {output_file_name} uploaded to Google Drive (folder id: {OUTPUT_FOLDER_ID})")

    if INPUT_FOLDER_ID is not None:
        logger.debug("Deleting input file from Google Drive")
        try:
            delete_file(drive_service, input_file_id)
        except Exception as e:
            logger.error(f"Failed to delete input file from Drive: {e}")
            return {
                "statusCode": 500,
                "success_pct": success_percentage,
                "message": f"Failed to delete input file: {e}"
            }
        else:
            logger.info(f"Input file {input_file_name} deleted from Google Drive")
        # upload jzon.gz file with remaining_urls to Google Drive, if any:
        if remaining_urls:
            new_input_name = f"{file_prefix}{remaining_urls[0]['id']}.json.gz"
            try:
                upload_zip_to_drive(drive_service, remaining_urls, new_input_name, INPUT_FOLDER_ID)
            except Exception as e:
                logger.error(f"Failed to upload new input file with remaining URLs to Drive: {e}")
            else:
                logger.info(f"File {new_input_name} with remaining URLs uploaded to Drive")

    return {
        "statusCode": 200,
        "success_pct": success_percentage,
        "n_successful": len(successful_results),
        "n_failed": len(failed_results),
        "n_remaining": len(remaining_urls),
    }


class FileNotFound(Exception):
    pass


class BatchFailed(Exception):
    pass


def parse_page(id: int, url: str) -> dict:
    ua = UserAgent(platforms="pc").random # NOTE the html of "mobile" doesn't have the same structure
    headers = {'User-Agent': ua}
    # logger.info(f"UA: {ua} | URL: {url}")
    response = requests.get(url, headers=headers)
    html_content = response.content
    soup = BeautifulSoup(html_content, 'html.parser')
    title = soup.title.get_text()
    if response.status_code == 200:
        target_element = soup.find('div', {'id': 'rcnt'})
        if target_element:
            res = str(target_element)
            logger.debug(f"Success: {url}")
            return {
                "id": id, "success": True, "html": res, 
                'url': url, 'title': title, "ua": headers['User-Agent']
                }
        else:
            logger.warning(f"Failed to find element in: {url}")
            return {
                "id": id, "success": False, "html": html_content, 
                'url': url, 'title': title,  "ua": headers['User-Agent'],
                }
    elif response.status_code == 429: # too many requests error
        time.sleep(2) # This might help to complete other tasks that are still running and might succeed
        raise Exception(f"Too many requests: {url}")
    elif response.status_code is not None:
        logger.warning(f"Failed with status code: {response.status_code}: {url}")
        return {
            "id": id, "success": False, "html": html_content,
            'url': url, 'title': title,  "ua": headers['User-Agent'],
            }
    else:
        logger.warning(f"Failed: {url}")
        return {
            "id": id, "success": False, 
            'url': url,
            }


def get_google_drive_service(key_file: str):
    scopes = ['https://www.googleapis.com/auth/drive']
    credentials = service_account.Credentials.from_service_account_file(
        key_file, scopes=scopes)
    return discovery.build('drive', 'v3', credentials=credentials, cache_discovery=False)


def download_file(drive_service, folder_id: str, file_name: str) -> str:
    """Download a specific file from the specified folder by name.
    Returns the file ID.
    """
    results = drive_service.files().list(
        q=f"'{folder_id}' in parents and name='{file_name}'",
        pageSize=100,
        fields="files(id, name)"
    ).execute(num_retries=5)
    files = results.get('files', [])
    if not files:
        raise FileNotFound(f"{file_name} file not found in folder {folder_id}")
    file_info = files[0]  # NOTE Assuming there's only one file with the specified name
    file_id = file_info.get('id')
    request = drive_service.files().get_media(fileId=file_id)
    logger.info(f"Downloading file: {file_name}")
    with open(f'/tmp/{file_name}', 'wb') as f:
        downloader = MediaIoBaseDownload(f, request)
        done = False
        while not done:
            status, done = downloader.next_chunk(num_retries=5)
    return file_id


def delete_file(drive_service, file_id: str):
    """Delete a file from Google Drive using its ID."""
    drive_service.files().delete(fileId=file_id).execute(num_retries=5)


def read_json_file(file_name: str) -> List[Dict]:
    file_handler = gzip.open if file_name.endswith('.gz') else open
    with file_handler(f'/tmp/{file_name}', 'rb') as f:
        data = json.load(f)
    return data


def upload_file_to_drive(drive_service, input_path, output_file_name, folder_id):
    file_metadata = {'name': output_file_name}
    if folder_id is not None:
        file_metadata['parents'] = [folder_id]
    media = MediaFileUpload(input_path)
    drive_service.files().create(body=file_metadata, media_body=media, fields='id').execute(num_retries=5)


def upload_zip_to_drive(
        drive_service, data: List[Dict], output_file_name: str, folder_id: str
    ):
    tmp_file_path = f'/tmp/{output_file_name}'
    logging.debug(f"Saving {output_file_name} tmp file")
    with gzip.open(tmp_file_path, "wt") as f:
        json.dump(data, f)
    logging.debug(f"Done saving")
    upload_file_to_drive(drive_service, tmp_file_path, output_file_name, folder_id)


def create_output(results: List[Dict]) -> List[Dict]:
    """Create the output to be saved as a pickle file"""
    # a list of dictionaries with keys: id, date, html:
    current_date = time.strftime("%Y-%m-%d")
    output = [{
        "id": res["id"],
        "date": current_date,
        "html": res["html"]
        } for res in results]
    return output
