import networkx as nx
import numpy as np

from project.graph_building.build_network import audio_feature_names


def unison_shuffled_copies(a, b):
    assert len(a) == len(b)
    p = np.random.permutation(len(a))
    return a[p], b[p]


# {'danceability': 0.6259, 'energy': 0.713, 'loudness': -9.0392, 'mode': 0.6989, 'speechiness': 0.0683,
#  'acousticness': 0.1288, 'instrumentalness': 0.0056, 'liveness': 0.193, 'valence': 0.7513, 'tempo': 130.5836,
#  'genres': ['Comedy', 'parody'], 'peak_rank': 71, 'weeks_on_chart': 36, 'last_week': 4.0, 'in_degree': 4,
#  'out_degree': 75, 'partition_id': 0}

def prepare_input(node_name1, node_name2, node_to_attr, G):
    attr1 = node_to_attr[node_name1]
    attr2 = node_to_attr[node_name2]
    feature_dict = {}
    # feature_dict["same_partition"] = int(
    #     attr1["partition_id"] == attr2["partition_id"])  # DATA LEAK IF EVALUATED ON THE SAME NETWORK

    feature_dict["in_degree_diff"] = int(attr1["in_degree"] - attr2["in_degree"])
    feature_dict["out_degree_diff"] = int(attr1["out_degree"] - attr2["out_degree"])
    feature_dict["genre_overlap"] = len(set(attr1.get("genres", [])).intersection(attr2.get("genres", [])))

    weeks_on_chart1 = attr1["weeks_on_chart"] if attr1["weeks_on_chart"] else 0
    weeks_on_chart2 = attr2["weeks_on_chart"] if attr2["weeks_on_chart"] else 0

    feature_dict["weeks_in_chart_diff"] = abs(weeks_on_chart1 - weeks_on_chart2)

    feature_dict["one_is_known"] = int(
        weeks_on_chart1 * weeks_on_chart2 == 0 and (weeks_on_chart1 != 0 or weeks_on_chart2 != 0))
    feature_dict["both_are_known"] = int((weeks_on_chart1 * weeks_on_chart2) != 0)
    feature_dict["neither_are_known"] = int(weeks_on_chart1 + weeks_on_chart2 == 0)

    for feature in audio_feature_names:
        feature_dict[f"{feature}_diff"] = abs(attr1[feature] - attr2[feature])

    features_list = list(sorted(feature_dict.items(), key=lambda d: d[0]))
    features_only = np.array([v for k, v in features_list])
    names_only = np.array([k for k, v in features_list])
    return features_only, names_only


def prepare_dataset(G: nx.DiGraph, shuffle=True):
    G_undir: nx.Graph = G.to_undirected()
    node_to_attr = dict(G.nodes(data=True))
    X1 = []
    y = []
    y_names = None
    for edge in G_undir.edges():
        node1, node2 = edge
        x, y_names = prepare_input(node1, node2, node_to_attr, G)
        X1.append(x)
        y.append(1)

    unconnected_pairs_count_limit = 2 * len(y)
    try_limit = 2 * unconnected_pairs_count_limit
    X2 = []
    attempts = 0
    while len(X2) < unconnected_pairs_count_limit and attempts < try_limit:
        attempts += 1

        node1 = np.random.choice(G_undir)
        node2 = np.random.choice(G_undir)
        if node1 == node2 or node1 in G.neighbors(node2):
            continue
        x, y_names = prepare_input(node1, node2, node_to_attr, G)
        X2.append(x)
        y.append(0)
    X = X1 + X2
    y = np.array(y)
    X = np.stack(X)

    if shuffle:
        X, y = unison_shuffled_copies(X, y)

    return X, y, y_names


def prepare_dataset_randomly(G: nx.DiGraph, N=10000):
    G_undir: nx.Graph = G.to_undirected()
    node_to_attr = dict(G.nodes(data=True))

    pair_count_limit = N
    X = []
    y = []
    while len(X) < pair_count_limit:
        node1 = np.random.choice(G_undir)
        node2 = np.random.choice(G_undir)
        if node1 == node2:
            continue
        x, y_names = prepare_input(node1, node2, node_to_attr, G)
        X.append(x)
        y.append(node1 in G.neighbors(node2))
    y = np.array(y)
    X = np.stack(X)

    return X, y, y_names


if __name__ == '__main__':
    G: nx.DiGraph = nx.read_gpickle("../G.pickle")
    X, y, y_names = prepare_dataset(G)
    #
    print()
