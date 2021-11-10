from tqdm import tqdm

from project.load_artists import get_page_content, get_items_from_category, get_all_subcategories, DATA_PATH


def main():
    albums_1995 = get_items_from_category("Category:1995_albums", )

    albums = get_items_from_category(category_name="Category:Albums")
    file_names = [a.replace("/", "_") for a in albums]
    dir_path = DATA_PATH / "album_contents"
    processed = set(dir_path.glob("*.txt"))

    for clean, album in tqdm(zip(file_names, albums)):
        path = dir_path / f"{clean}.txt"
        if path in processed: continue
        content = get_page_content(page_name=album)
        with open(path, 'w') as f:
            f.write(content)


if __name__ == '__main__':
    main()
