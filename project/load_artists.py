from copy import copy
from itertools import chain
import re
from pathlib import Path

import requests
import pandas
from tqdm import tqdm

MUSIC_FANDOM_URL = "https://music.fandom.com/api.php"
DATA_PATH = Path('data')


def get_links(text):
    pattern2_addr = re.findall(r".*", text)

    addresses = [p.replace(" ", " ") for p in chain(pattern2_addr)]
    return addresses


def get_all_subcategories(category_name="Category:Artists"):
    payload = {
        'format': 'json',
        'action': 'query',
        'list': 'categorymembers',
        'cmtype': 'subcat',  # 'page' for pages, 'subcat' for subcategories
        'cmtitle': category_name,
        'cmlimit': 'max'
    }
    responses = repeated_request(payload)
    subcategories = [members['title']
                     for r in responses
                     for members in r["query"]['categorymembers']]
    return subcategories


def repeated_request(payload):
    responses = []
    first_time = True
    should_continue = False
    continuation_info = None
    while first_time or should_continue:
        first_time = False
        if should_continue:
            payload["cmcontinue"] = continuation_info
        r = requests.get(MUSIC_FANDOM_URL, params=payload)
        req_json = r.json()
        responses.append(req_json)

        should_continue = req_json.get("continue") is not None
        if should_continue:
            continuation_info = req_json["continue"]["cmcontinue"]
    return responses


def get_items_from_category(category_name):
    payload = {
        'format': 'json',
        'action': 'query',
        'list': 'categorymembers',
        'cmtype': 'page',
        'cmtitle': category_name,
        'cmlimit': 'max'
    }

    responses = repeated_request(payload)
    artists = [members['title']
               for r in responses
               for members in r["query"]['categorymembers']]
    return artists


def get_page_content(page_name):
    payload = {
        'format': 'json',
        'action': 'query',
        'prop': 'revisions',
        'titles': page_name,
        'rvslots': '*',
        'rvprop': 'content',
    }

    r = requests.get(MUSIC_FANDOM_URL, params=payload)
    req_json = r.json()
    pages = [page for _, page in req_json['query']['pages'].items()]
    revisions = [p['revisions'] for p in pages]
    content = [r['slots']['main']['*'] for r in chain(*revisions)]
    if len(content) != 1:
        raise Exception("len(content) != 1")
    content_value = content[0]
    return content_value


def main():
    # categories_of_artists = get_all_subcategories(category_name="Category:Artists")
    artists = get_items_from_category(category_name="Category:Artists")
    artists_clean = [a.replace("/", "_") for a in artists]
    processed = set(DATA_PATH.glob("*.txt"))
    for clean, artist in tqdm(zip(artists_clean, artists)):
        path = DATA_PATH / f"{clean}.txt"
        if path in processed: continue
        # print(path)

        content = get_page_content(page_name=artist)
        with open(path, 'w') as f:
            f.write(content)


if __name__ == '__main__':
    main()
