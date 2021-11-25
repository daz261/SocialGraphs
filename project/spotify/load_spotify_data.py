import numpy as np
import pandas as pd
import ast
from tqdm.notebook import tqdm
import os
from tqdm import tqdm

from project.spotify.get_spotify_access import refresh_spotify_access
from project.util import DATA_PATH
from time import time


def infinite_retry(query, n=1):
    if n != 1:
        print("Attempt nr", n)
    try:
        output = query()
        return output
    except Exception as e:
        print(e)
        return infinite_retry(query, n + 1)


def find_album(sp, artist, album):
    response = infinite_retry(lambda: sp.search(q="album:" + album, type="album", limit=50))
    artist_to_URIs = {artist["name"]: (item["uri"], artist["uri"])
                      for item in response['albums']["items"] for
                      artist in item["artists"]}
    if artist in artist_to_URIs:
        return artist_to_URIs[artist]
    while response.get("next"):
        response = infinite_retry(lambda: sp.next(response))
        artist_to_album_uri = {artist["name"]: item["uri"]
                               for item in response['albums']["items"]
                               for artist in item["artists"]}
        if artist in artist_to_album_uri:
            return artist_to_album_uri[artist]
    return None, None


def search_album_id(list_of_albums, list_of_artists):
    sp = refresh_spotify_access()

    albums = []
    not_found = []

    for album, artist in tqdm(list(zip(list_of_albums, list_of_artists))):
        album_uri, artist_uri = find_album(sp, artist, album)
        if album_uri is None:
            not_found.append({"album": album, "artist": artist})
        else:
            albums.append({"album": album, "artist": artist, "album_uri": album_uri, "artist_uri": artist_uri})

    return pd.DataFrame(albums), pd.DataFrame(not_found)


def get_genre_label(sp, spotify_album, artist_id):
    artist_genre = infinite_retry(lambda: sp.artist(artist_id)["genres"])
    label = spotify_album["label"]
    return artist_genre, label


def get_album_features(row):
    album_id = row["album_uri"]
    artist_id = row["artist_uri"]

    audio_feature_names = ['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness',
                           'instrumentalness', 'liveness', 'valence', 'tempo']
    sp = refresh_spotify_access()
    album = infinite_retry(lambda: sp.album(album_id))
    album_collabs = get_collabs_from_album(album)
    release_date = album["release_date"]
    album_tracks = album['tracks']['items']
    artist_genre, label = get_genre_label(sp, album, artist_id)
    album_features = []
    for track in album_tracks:
        all_audio_features = infinite_retry(lambda: sp.audio_features(track['id']))
        if len(all_audio_features) == 0:
            continue
        else:
            try:
                album_features.append({feature: all_audio_features[0][feature] for feature in audio_feature_names})
            except Exception as e:
                print(artist_id)
                print(album_id)
                print(e)
    if len(album_features) == 0:
        avg_features = {feature: None for feature in audio_feature_names}
    else:
        avg_features = {feature: round(np.mean([x[feature] for x in album_features]), 5) for feature in
                        audio_feature_names}
    avg_features["artist_genre"] = artist_genre
    avg_features["label"] = label
    avg_features["release_date"] = release_date
    avg_features["album_id"] = album_id
    avg_features["artist_id"] = artist_id
    avg_features['collaborations'] = album_collabs
    return avg_features


def get_collabs_from_album(album):
    artist_names = [artist['name'] for artist in album['artists']]
    album_collabs = tuple([artist['name']
                           for track in album['tracks']['items']
                           for artist in track['artists']
                           if artist['name'] not in artist_names])
    album_collabs = tuple(set(album_collabs))
    return album_collabs


if __name__ == '__main__':
    df_wiki = pd.read_csv(DATA_PATH / "collaboration_clean.csv")
    albums = list(df_wiki['Album'])
    artists = list(df_wiki['Artist'])
    # df_albums, df_not_found = search_album_id(albums, artists) # Uncomment to run the code!
    df_albums, df_not_found = pd.read_csv("albums.csv"), pd.read_csv("not_found.csv")
    tqdm.pandas()
    audio_features = df_albums.progress_apply(get_album_features, axis='columns', result_type='expand')
    x = pd.concat([df_albums, audio_features], axis=1)
    x.to_csv("spotify_final.csv", index=False)
