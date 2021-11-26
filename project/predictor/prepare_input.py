import networkx as nx
import numpy as np

from project.graph_building.build_network import audio_feature_names


def prepare_input(node_name1, node_name2, node_to_attr):
    attr1 = node_to_attr[node_name1]
    attr2 = node_to_attr[node_name2]

    connection_features = {}
    connection_features["are_neighbors"] = node2 in G.neighbors(node1)
    connection_features["same_partition"] = int(attr1["partition_id"] == attr2["partition_id"])
    connection_features["in_degree_diff"] = int(attr1["in_degree"] == attr2["in_degree"])
    connection_features["out_degree_diff"] = int(attr1["out_degree"] == attr2["out_degree"])
    connection_features["genre_overlap"] = len(set(attr1["Genres"]).intersection(set(attr2["Genres"])))


    for feature in audio_feature_names:
        connection_features[f"{feature}_diff"] = abs(attr1[feature] - attr2[feature])


    edge_weight = "TODO"

    print()


if __name__ == '__main__':
    G: nx.DiGraph = nx.read_gpickle("../G.pickle")
    G_undir: nx.Graph = G.to_undirected()
    node_to_attr = dict(G.nodes(data=True))

    for edge in G_undir.edges():
        node1, node2 = edge
        prepare_input(node1, node2, node_to_attr)
