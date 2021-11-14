from tqdm import tqdm

from project.util import get_items_from_category, DATA_PATH, get_page_content, MUSIC_FANDOM_URL


def main():
    # categories_of_artists = get_all_subcategories(category_name="Category:Artists")
    artists = get_items_from_category(category_name="Category:Artists", base_url=MUSIC_FANDOM_URL)
    artists_clean = [a.replace("/", "_") for a in artists]
    processed = set(DATA_PATH.glob("*.txt"))
    for clean, artist in tqdm(zip(artists_clean, artists)):
        path = DATA_PATH / f"{clean}.txt"
        if path in processed: continue
        # print(path)

        content = get_page_content(page_name=artist, base_url=MUSIC_FANDOM_URL)
        with open(path, 'w') as f:
            f.write(content)


if __name__ == '__main__':
    main()
