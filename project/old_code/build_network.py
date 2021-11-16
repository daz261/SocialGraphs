import networkx as nx
import pandas as pd
from fa2 import ForceAtlas2
import numpy as np
import seaborn.apionly as sns
import matplotlib.animation

from project.util import DATA_PATH
import ast

import matplotlib.pyplot as plt

pos = None


def node_degree_to_size(degree):
    return 5 + degree ** 2 / 10


def plot_with_fa(G):
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
        verbose=True
    )

    positions = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=15000)
    nx.draw_networkx_nodes(G, positions, alpha=1, node_color=None, node_size=node_size)
    nx.draw_networkx_edges(G, positions, edge_color="green", alpha=0.15)
    nx.draw_networkx_labels(G, positions, node_label, font_size=20, font_color='black')

    plt.axis('off')
    plt.show()


def build_network(df):
    df["Artist references"] = df["Artist references"].apply(ast.literal_eval)
    G = nx.Graph()
    for _, row in df.iterrows():
        if row["Artist"] == "Various Artists":
            continue
        references = row["Artist references"]
        artist = row["Artist"]
        edges_to_add = [(artist, ref) for ref in references if ref != artist and ref != "Various Artists"]
        G.add_edges_from(edges_to_add)
    return G


forceatlas2 = ForceAtlas2(
    # # Behavior alternatives
    outboundAttractionDistribution=True,  # Dissuade hubs
    edgeWeightInfluence=1.0,
    # Performance
    jitterTolerance=7,  # Tolerance
    barnesHutOptimize=True,
    barnesHutTheta=1.6,
    # # Tuning
    scalingRatio=1.0,
    strongGravityMode=False,
    gravity=15,
    # # Log
    verbose=False
)


def build_gif(df):
    year_min, year_max = df.Year.min(), df.Year.max()

    G = build_network(df)
    global pos
    pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=1)
    node_label = {k: k for k, d in dict(G.degree).items() if d > 90}
    node_size = [node_degree_to_size(d) for k, d in dict(G.degree).items()]

    plt.rcParams["figure.figsize"] = (20, 20)
    frames = 500
    interval = 130
    steps_per_iteration = 500

    def update(data):
        print(f"{1 + data} / {frames}")
        global pos
        fig.clear()
        # G = build_network(df[df[]])
        pos = forceatlas2.forceatlas2_networkx_layout(G, pos=pos, iterations=steps_per_iteration,
                                                      weight_attr=None)  # TODO: weight attr

        nx.draw_networkx_nodes(G, pos, alpha=1, node_color=None, node_size=node_size)
        nx.draw_networkx_edges(G, pos, edge_color="green", alpha=0.15)
        nx.draw_networkx_labels(G, pos, labels=node_label, font_size=10, font_color='black')

    fig = plt.gcf()

    ani = matplotlib.animation.FuncAnimation(fig, update,
                                             frames=frames,
                                             interval=interval,
                                             repeat=False,
                                             blit=False)
    ani.save(str(DATA_PATH / "full.gif"))


def main():
    df = pd.read_csv(DATA_PATH / "updated_artist_matches.csv")
    build_gif(df)


if __name__ == '__main__':
    main()
