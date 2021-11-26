import networkx as nx
import pandas as pd

from project.graph_building.network_enhancements import add_node_degrees, add_community_id
from project.util import DATA_PATH
import ast

month_to_ordinal = {'JANUARY': "01",
                    'FEBRUARY': "02",
                    'MARCH': "03",
                    'APRIL': "04",
                    'MAY': "05",
                    'JUNE': "06",
                    'JULY': "07",
                    'AUGUST': "08",
                    'SEPTEMBER': "09",
                    'OCTOBER': "10",
                    'NOVEMBER': "11",
                    'DECEMBER': "12"}


def is_row_necessary(row) -> bool:
    if row["Artist"] == row["Album"]: return False  # Ugly quick fix. We should rerun files
    if row["Artist"] == "Various Artists": return False
    if "remix" in str(row["Notes"]).lower(): return False
    if "box set" in str(row["Notes"]).lower():
        return False
    if "compilation" in str(row["Notes"]).lower(): return False
    return True


def parse_year(row):
    year, month, day = row["Year"], row["Date"], str(row["Date.1"])
    month_number = month_to_ordinal[month]
    if day == "?":
        day = "1"
    if len(day) == 1:
        day = "0" + day
    elif len(day) == 2:
        pass
    else:
        raise Exception(f"Bad day value: {day}")
    date = pd.to_datetime(f'{year}{month_number}{day}', format='%Y%m%d')
    return date


def preprocess_df():
    collaboration_df = pd.read_csv(DATA_PATH / "attributes_album_artist_date_table_v2.csv")
    collaboration_df = collaboration_df.drop(columns=["Unnamed: 0.1", "Unnamed: 0"])
    df_matches = pd.read_csv(DATA_PATH / "updated_artist_matches.csv")
    collaboration_df["Artist references"] = df_matches["Artist references"].apply(ast.literal_eval)
    collaboration_df["Date"] = collaboration_df.apply(parse_year, axis=1)
    collaboration_df = collaboration_df.drop(columns=["Year", "Date.1"])
    collaboration_df = collaboration_df[collaboration_df.apply(is_row_necessary, axis=1)]

    spotify_df = pd.read_csv(DATA_PATH / "spotify_final.csv")
    collaboration_df = pd.merge(collaboration_df, spotify_df,
                                right_on=["album", "artist"],
                                left_on=["Album", "Artist"])

    billboard_df = pd.read_csv(DATA_PATH / "billboard_df.csv",
                               usecols=["Album", "Artist", "last_week", "weeks_on_chart", "peak_rank"])
    collaboration_df = pd.merge(collaboration_df, billboard_df,
                                on=["Album", "Artist"])

    return collaboration_df


def prepare_pairs(collaboration_df):
    # DataFrame.explode changes each element in list to a new row
    pairs_df = collaboration_df.explode("Artist references", ignore_index=True)
    pairs_df = pairs_df.dropna(subset=["Artist references"])  # I've found some empty values in artist references
    pairs_df = pairs_df[pairs_df["Artist"] != pairs_df["Artist references"]]

    # pairs_df2 = collaboration_df.explode("collaborations", ignore_index=True)
    # pairs_df2 = pairs_df2.dropna(subset=["collaborations"])  # I've found some empty values in artist references
    # pairs_df2 = pairs_df[pairs_df["Artist"] != pairs_df["Artist references"]]
    return pairs_df.reset_index(drop=True)


def get_edge_features(df) -> dict:
    """
    Extracts summary values from the dataframe to an edge
    """

    return {}


audio_feature_names = ['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness',
                       'instrumentalness', 'liveness', 'valence', 'tempo']


def get_node_features(pairs_df, artist):
    sub_df = pairs_df[(pairs_df["Artist"] == artist) | (pairs_df["Artist references"] == artist)]
    features_dict = {feature_name: round(sub_df[~sub_df[feature_name].isnull()][feature_name].mean(), 4) for
                     feature_name in audio_feature_names}
    genres = list({g
                   for row in {row for row in sub_df["Genre"] if isinstance(row, str)}
                   for genres in ast.literal_eval(row)
                   for g in genres})

    features_dict["genres"] = genres
    features_dict["peak_rank"] = sub_df["peak_rank"].max()
    features_dict["weeks_on_chart"] = sub_df["weeks_on_chart"].sum()
    features_dict["last_week"] = sub_df["last_week"].min()

    return features_dict


def construct_graph(pairs_df, album_df):
    G = nx.DiGraph()
    artist_with_node_features = set()

    for artists, df in pairs_df.groupby(["Artist", "Artist references"]):
        artist1, artist2 = artists
        weight = len(df)
        edge_features = get_edge_features(df)

        if artist1 not in artist_with_node_features:
            node_features = get_node_features(pairs_df, artist1)
            G.add_node(artist1, **node_features)
            artist_with_node_features.add(artist1)

        if artist2 not in artist_with_node_features:
            node_features = get_node_features(pairs_df, artist2)
            G.add_node(artist2, **node_features)
            artist_with_node_features.add(artist2)

        G.add_edge(artist1, artist2, weight=weight, **edge_features)
    return G


def enhance_network(G):
    G = add_node_degrees(G)
    G = add_community_id(G)
    return G


def build_network():
    collaboration_df = preprocess_df()
    collaboration_df.to_csv(DATA_PATH / "collaboration_clean.csv", index=False)
    pairs_df = prepare_pairs(collaboration_df)
    G = construct_graph(pairs_df, collaboration_df)
    G = enhance_network(G)
    return G


def main():
    G = build_network()
    nx.write_gpickle(G, "G.pickle")


if __name__ == '__main__':
    main()
