"""
Created on Sun Nov 21 11:58:24 2021

@author: Nikolaj
"""
import spotipy
import os

from project.spotify.get_spotify_environment import set_spotify_environment

script_dir = os.getcwd()
spotify_keys_path = os.path.join(script_dir, 'spotify_keys.txt')
# Set the client_id and client_secret environmental variables
set_spotify_environment(spotify_keys_path)
from ast import literal_eval
from time import time

sp = None
with open(".cache") as f:
    expiration = literal_eval(f.read())["expires_at"]


def refresh_spotify_access():
    global sp
    global expiration
    if int(time()) + 10 > expiration or sp is None:
        token = spotipy.oauth2.SpotifyClientCredentials(client_id=os.getenv('CLIENT_ID'),
                                                        client_secret=os.getenv('CLIENT_SECRET'))
        cache_token = token.get_access_token(as_dict=False)
        sp = spotipy.Spotify(cache_token)
        with open(".cache") as f:
            expiration = literal_eval(f.read())["expires_at"]
        print(f"RETRIEVING TOKEN, WILL EXPIRE IN {expiration - int(time())} seconds")
    return sp
