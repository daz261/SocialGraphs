import networkx as nx
import numpy as np
import pandas as pd

from project.graph_building.build_network import build_network


def prepare_dataset_success(G: nx.DiGraph):
    node_to_attr = dict(G.nodes(data=True))
    X_list = []
    y_names = ['danceability', 'energy', 'loudness', 'mode', 'speechiness', 'acousticness', 'instrumentalness',
               'liveness', 'valence', 'tempo', 'in_degree', 'out_degree', 'sentiment']
    y = []
    for node, attr in node_to_attr.items():
        x = np.array([attr.get(y_name, 0) for y_name in y_names])
        X_list.append(x)
        y.append(attr.get('weeks_on_chart', 0))
    X, y = np.stack(X_list), np.array(y)

    return X, y, y_names


if __name__ == '__main__':
    train_range = pd.date_range(start='1/01/1990', end='1/01/1995')
    test_range = pd.date_range(start='1/01/1995', end='1/01/2001')

    G_train = build_network(date_range=train_range)
    # G_test = build_network(date_range=test_range)
    X, y, y_names = prepare_dataset_success(G_train)
    #

    print()
