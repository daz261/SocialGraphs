import networkx as nx
import numpy as np

from project.graph_building.build_network import audio_feature_names


# {'danceability': 0.6259, 'energy': 0.713, 'loudness': -9.0392, 'mode': 0.6989, 'speechiness': 0.0683,
#  'acousticness': 0.1288, 'instrumentalness': 0.0056, 'liveness': 0.193, 'valence': 0.7513, 'tempo': 130.5836,
#  'genres': ['Comedy', 'parody'], 'peak_rank': 71, 'weeks_on_chart': 36, 'last_week': 4.0, 'in_degree': 4,
#  'out_degree': 75, 'partition_id': 0}


def prepare_input(node_name1, node_name2, node_to_attr, G):
    global  count
    attr1 = node_to_attr[node_name1]
    attr2 = node_to_attr[node_name2]

    feature_dict = {}
    feature_dict["are_neighbors"] = int(node2 in G.neighbors(node1))
    feature_dict["same_partition"] = int(
        attr1["partition_id"] == attr2["partition_id"])  # DATA LEAK IF EVALUATED ON THE SAME NETWORK

    feature_dict["in_degree_diff"] = int(attr1["in_degree"] == attr2["in_degree"])
    feature_dict["out_degree_diff"] = int(attr1["out_degree"] == attr2["out_degree"])
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
    return features_only


if __name__ == '__main__':
    G: nx.DiGraph = nx.read_gpickle("../G.pickle")
    G_undir: nx.Graph = G.to_undirected()
    node_to_attr = dict(G.nodes(data=True))
    X = []
    y = []

    for edge in G_undir.edges():
        node1, node2 = edge
        X.append(prepare_input(node1, node2, node_to_attr, G))
        y.append(1)
    X = np.stack(X)
    print()
