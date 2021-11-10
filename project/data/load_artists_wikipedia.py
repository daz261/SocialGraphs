import pandas as pd

from project.util import load_table_from_wiki, DATA_PATH

if __name__ == '__main__':
    years = [str(y) for y in range(1990, 2000)]
    dataframes = []
    for year in years:
        df = load_table_from_wiki(page_name=f"{year}_in_music")
        dataframes.append(df)
    big_df = pd.concat(dataframes).reset_index()
    big_df.to_csv(DATA_PATH / "album_artist_date_table.csv")
