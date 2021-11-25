import ast
from typing import Dict, Any


def extract_node_features(df) -> Dict[str, Any]:
    """
    Extracts summary values from the dataframe to a node
    """
    # TODO: Here goes average over Spotify features
    genre_features = get_genre_features(df)

    feature_dict = {**genre_features, }  # Combines all

    return feature_dict


def get_genre_features(df) -> dict:
    try:
        if df["Genre"]:
            genres = []
            for genre_string in df["Genre"].to_list():
                genre_list = ast.literal_eval(genre_string)
                genres.extend(genre_list[0])
            genre_dict = {g.lower(): True for g in genres}
    except Exception as e:
        print("TODO: fix genre parsing. Look out for nan")
    return None
