from project.util import get_subcategories_from_category_recurrent, WIKIPEDIA_URL, get_items_from_category

if __name__ == '__main__':
    # items = get_subcategories_from_category_recurrent("Category:Musical_groups", base_url=WIKIPEDIA_URL)
    albums = get_subcategories_from_category_recurrent("Category:Albums by artist", WIKIPEDIA_URL)
    print()