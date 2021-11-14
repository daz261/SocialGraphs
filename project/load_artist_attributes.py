import pandas as pd
import requests
from bs4 import BeautifulSoup
import time

pd_links = pd.read_csv("album_artist_date_table.csv")

def fire_get_request(url, headers, timeout=(5,25)):
    start_time = time.time()
    response=requests.get(url, timeout=timeout,headers=headers)
    end_time=time.time() - start_time
    return {'responsetime':end_time,'response':response.text}


def infotext(df):

    removed_chars = "[]1234567890"
    year_dict = {}
    label_dict = {}
    genre_dict = {}

    for i, artist_link in enumerate(df['Artist link']):
        wikiurl = f"https://en.wikipedia.org/wiki/{artist_link}"
        try:
            #requests.post(url, headers, timeout=10)
            response = requests.get(wikiurl, timeout=10)
        except requests.exceptions.Timeout:
            print ("Timeout occurred")
        soup = BeautifulSoup(response.text, 'html.parser')
        indiatable = soup.find('table')
        try:
            #get the infobox as a table
            #index = 0 because the infobox is always the first table in the Wiki page
            dfs_1 = pd.read_html(str(indiatable))[0]
            #re-index
            df_1 = dfs_1.set_index(dfs_1.columns[0])
            #transpose the table (switch rows with columns)
            df_2 = df_1.transpose()
            #extract the 'years active'
            years_active = str(df_2['Years active']).split('\n')[0]
            #split by 4 spaces
            years_active2 =years_active.split("    ")[1]
            #print(years_active2)
            #create dict
            year_dict[artist_link]=years_active2
            label = str(df_2['Labels']).split('\n')[0].split("    ")[1]
            #print(label)
            # create dict
            label_dict[artist_link] = label
            genre = str(df_2['Genres']).split('\n')[0].split("    ")[1]
            for char in removed_chars:
                genre = genre.replace(char, "")
            #print(genre)
            # create dict
            genre_dict[artist_link] = genre
        except:
            #print(artist_link)
            pass

    df["Years Active"] = df["Artist link"].apply(lambda artist: year_dict.get(artist))
    df["Labels"] = df["Artist link"].apply(lambda artist: label_dict.get(artist))
    df["Genre"] = df["Artist link"].apply(lambda artist: genre_dict.get(artist))
    return df

def main():
    df = infotext(pd_links)
    df.to_csv("attributes_album_artist_date_table.csv")

if __name__ == '__main__':
    main()