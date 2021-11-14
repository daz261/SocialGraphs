import networkx as nx
import pandas as pd
from fa2 import ForceAtlas2

from project.util import DATA_PATH
import ast

import matplotlib.pyplot as plt


def plot_with_fa(G):
    def node_degree_to_size(degree):
        return degree

    node_size = [node_degree_to_size(d) for k, d in dict(G.degree).items()]
    node_label = {k: k for k, d in dict(G.degree).items() if d > 40}

    forceatlas2 = ForceAtlas2(
        # Behavior alternatives
        outboundAttractionDistribution=True,  # Dissuade hubs
        edgeWeightInfluence=1.0,

        # Performance
        jitterTolerance=7,  # Tolerance
        barnesHutOptimize=True,
        barnesHutTheta=1.6,

        # Tuning
        scalingRatio=1.0,
        strongGravityMode=False,
        gravity=15.0,

        # Log
        verbose=True)

    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=15000)
    nx.draw_networkx_nodes(G, positions, alpha=1, node_color=None, node_size=node_size)
    nx.draw_networkx_edges(G, positions, edge_color="green", alpha=0.15)
    nx.draw_networkx_labels(G, positions, node_label, font_size=10, font_color='black')

    plt.axis('off')
    plt.show()


def main():
    df = pd.read_csv(DATA_PATH / "updated_artist_matches.csv")
    df["Artist references"] = df["Artist references"].apply(ast.literal_eval)
    G = nx.Graph()
    for _, row in df.iterrows():
        if row["Artist"] == "Various Artists":
            continue
        references = row["Artist references"]
        artist = row["Artist"]
        edges_to_add = [(artist, ref) for ref in references if ref != artist and ref != "Various Artists"]
        G.add_edges_from(edges_to_add)
    nx.write_gpickle(G, "G.pickle")
    plt.rcParams['figure.figsize'] = [50, 50]
    plot_with_fa(G)



if __name__ == '__main__':
    main()
