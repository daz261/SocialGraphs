import os
from itertools import chain
from pathlib import Path
import urllib.parse

import pandas as pd
import requests
from bs4 import BeautifulSoup

ROOT_DIR = Path(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = ROOT_DIR / 'data'

MUSIC_FANDOM_URL = "https://music.fandom.com/api.php"
WIKIPEDIA_URL = "https://en.wikipedia.org/w/api.php"


def get_subcategories_from_category_recurrent(category_name, base_url):
    items = get_all_subcategories(category_name, base_url)
    for item in items:
        items += get_subcategories_from_category_recurrent(item, base_url)
    return items


def get_links_for_recurrent_subcategories(category_name, base_url):
    subcategories = get_subcategories_from_category_recurrent(category_name, base_url)
    for subcat in subcategories:
        pass  # TODO: work in progress


def get_links_from_page(category_name, url):
    try:
        decoded_url = urllib.parse.unquote(category_name)
    except Exception as e:
        print(e)
    payload = {
        'format': 'json',
        'action': 'query',
        'prop': 'links',
        'pllimit': 'max',
        'titles': decoded_url,
        # 'plnamespace': '1',
    }

    responses = repeated_request(payload, base_url=url)
    links = [link['title']
             for r in responses
             for _, pages in r["query"]["pages"].items()
             for link in pages["links"]
             ]
    return links


def get_page_content(page_name, base_url):
    payload = {
        'format': 'json',
        'action': 'query',
        'prop': 'revisions',
        'titles': page_name,
        'rvslots': '*',
        'rvprop': 'content',
    }

    r = requests.get(base_url, params=payload)
    req_json = r.json()
    pages = [page for _, page in req_json['query']['pages'].items()]
    revisions = [p['revisions'] for p in pages]
    content = [r['slots']['main']['*'] for r in chain(*revisions)]
    if len(content) != 1:
        raise Exception("len(content) != 1")
    content_value = content[0]
    return content_value


def get_all_subcategories(category_name, base_url):
    payload = {
        'format': 'json',
        'action': 'query',
        'list': 'categorymembers',
        'cmtype': 'subcat',  # 'page' for pages, 'subcat' for subcategories
        'cmtitle': category_name,
        'cmlimit': 'max'
    }
    responses = repeated_request(payload, base_url=base_url)
    subcategories = [members['title']
                     for r in responses
                     for members in r["query"]['categorymembers']]
    return subcategories


def get_items_from_category(category_name, base_url):
    payload = {
        'format': 'json',
        'action': 'query',
        'list': 'categorymembers',
        'cmtype': 'page',
        'cmtitle': category_name,
        'cmlimit': 'max'
    }

    responses = repeated_request(payload, base_url=base_url)
    artists = [members['title']
               for r in responses
               for members in r["query"]['categorymembers']]
    return artists


def load_table_from_wiki(page_name):
    wikiurl = f"https://en.wikipedia.org/wiki/{page_name}"
    # table_class = "wikitable sortable jquery-tablesorter" # TODO: check if this should be used
    response = requests.get(wikiurl)
    soup = BeautifulSoup(response.text, 'html.parser')
    indiatable = soup.find_all('table', {'class': "wikitable"})
    dataframes = pd.read_html(str(indiatable))
    if len(dataframes) != 1:
        print(f"{page_name} has {len(dataframes)} dataframes!")
    df = pd.concat(dataframes)
    df = df.dropna(axis=0, subset=["Date", "Date.1"])
    df = df[["Date", "Date.1", "Album", "Artist", "Notes"]].reset_index()
    df = df.drop(labels=["index"], axis=1)

    text_to_links = {a.contents[0]: a["href"].replace("/wiki/", "")
                     for a in soup.find_all('a', href=True)
                     if a.text and a["href"].startswith("/wiki")}
    df["Album link"] = df["Album"].apply(lambda album: text_to_links.get(album))
    df["Artist link"] = df["Artist"].apply(lambda artist: text_to_links.get(artist))
    return df


def repeated_request(payload, base_url):
    responses = []
    first_time = True
    should_continue = False
    continuation_info = None
    continue_field = ""
    while first_time or should_continue:
        first_time = False
        if should_continue:
            payload[continue_field] = continuation_info
        r = requests.get(base_url, params=payload)
        req_json = r.json()
        responses.append(req_json)

        should_continue = req_json.get("continue") is not None
        if should_continue:
            if "cmcontinue" in req_json["continue"]:
                continuation_info = req_json["continue"]["cmcontinue"]
                continue_field = "cmcontinue"
            elif "plcontinue" in req_json["continue"]:
                continuation_info = req_json["continue"]["plcontinue"]
                continue_field = "plcontinue"
    return responses
