import pandas as pd

from project.util import load_table_from_wiki, DATA_PATH, get_links_from_page, WIKIPEDIA_URL, get_items_from_category, \
    MUSIC_FANDOM_URL

if __name__ == '__main__':
    years = [str(y) for y in range(1990, 2000)]
    dataframes = []
    for year in years:
        df = load_table_from_wiki(page_name=f"{year}_in_music")
        dataframes.append(df)
    big_df = pd.concat(dataframes).reset_index()
    big_df = big_df.drop(labels=["index"], axis=1)
    # big_df = big_df.dropna(axis=0, subset=["Album link"])
    big_df.to_csv(DATA_PATH / "album_artist_date_table.csv", index=False)

    big_df = pd.read_csv("data/album_artist_date_table.csv")
    artists = get_items_from_category(category_name="Category:Artists", base_url=MUSIC_FANDOM_URL)

    all_artists = set(big_df["Artist"]) | set(artists)
    found_match = []


    def get_matches(album_link):
        from_album = get_links_from_page(album_link, url=WIKIPEDIA_URL)
        matches = [l for l in from_album if l in all_artists]
        return matches


    from tqdm import tqdm

    tqdm.pandas()
    big_df["Artist Collaborators"] = big_df["Album link"].progress_apply(get_matches)
    #big_df.to_csv("data/updated_artist_matches.csv")
