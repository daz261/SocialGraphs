from ..util import repeated_request
import pandas as pd
import requests
from bs4 import BeautifulSoup


def get_links_from_page(category_name, url):
    payload = {
        'format': 'json',
        'action': 'query',
        'prop': 'links',
        'pllimit': 'max',
        'titles': category_name,
        'plnamespace': '*',
    }

    responses = repeated_request(payload, base_url=url)
    links = [link['title']
             for r in responses
             for _, pages in r["query"]["pages"].items()
             for link in pages["links"]]
    return links


def load_table_from_wiki():
    wikipedia_url = "https://en.wikipedia.org/w/api.php"
    # links = get_links_from_page("1990_in_music", wikipedia_url)
    # print(links)

    wikiurl = "https://en.wikipedia.org/wiki/1990_in_music"
    # table_class = "wikitable sortable jquery-tablesorter"
    response = requests.get(wikiurl)
    soup = BeautifulSoup(response.text, 'html.parser')
    indiatable = soup.find('table', {'class': "wikitable"})
    df = pd.read_html(str(indiatable))


if __name__ == '__main__':
    wikipedia_url = "https://en.wikipedia.org/w/api.php"
    # links = get_links_from_page("1990_in_music", wikipedia_url)
    # print(links)

    df = pd.read_csv("music_1990.csv")
    for _, data in df.iterrows():
        artist = data["Artist"]
        album = data["Album"]
        links_artist = get_links_from_page(artist, url=wikipedia_url)
        links_album = get_links_from_page(album, url=wikipedia_url)
        print
