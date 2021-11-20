from fa2 import ForceAtlas2
from matplotlib import pyplot as plt

from project.visualisations.graph_gif import node_degree_to_size
import networkx as nx

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


def show_graph(G, title, iterations=10000, show_edges=True):
    pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=iterations)
    node_label = {k: k for k, d in dict(G.degree).items() if d > 50}
    node_size = [node_degree_to_size(d) for k, d in dict(G.degree).items()]
    plt.rcParams["figure.figsize"] = (20, 20)
    nx.draw_networkx_nodes(G, pos, alpha=1, node_color=None, node_size=node_size)
    if show_edges:  # It takes a while to plot all connections, sometimes it makes sense to drop them.
        nx.draw_networkx_edges(G, pos, edge_color="green", alpha=0.15)
    nx.draw_networkx_labels(G, pos, labels=node_label, font_size=10, font_color='black')
    plt.title(title)
    plt.show()
