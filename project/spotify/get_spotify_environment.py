# -*- coding: utf-8 -*-
"""
Created on Sun Nov 21 10:58:44 2021

@author: Nikolaj
"""
import os
import json


# Define spotify developer generated keys
# accessed fromm https://developer.spotify.com/dashboard/applications/303d6e502364476b811ea5c0180415c5
def set_spotify_environment(file):
    with open(file, 'r') as f:
        spotify_keys_dict = json.load(f)
        # Set environment variables
        os.environ['CLIENT_ID'] = spotify_keys_dict['CLIENT_ID']
        os.environ['CLIENT_SECRET'] = spotify_keys_dict['CLIENT_SECRET']
