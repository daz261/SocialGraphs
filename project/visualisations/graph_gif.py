import matplotlib
import networkx as nx
from fa2 import ForceAtlas2
import matplotlib.pyplot as plt

from project.util import DATA_PATH

forceatlas2 = ForceAtlas2(
    # # Behavior alternatives
    outboundAttractionDistribution=True,  # Dissuade hubs
    edgeWeightInfluence=2.0,
    # Performance
    jitterTolerance=7,  # Tolerance
    barnesHutOptimize=True,
    barnesHutTheta=1.6,
    # # Tuning
    scalingRatio=1.0,
    strongGravityMode=False,
    gravity=2,
    # # Log
    verbose=False
)

pos = None


def node_degree_to_size(degree):
    return 10 + degree ** 1.5 / 20


def build_gif(G, frames=100, interval=230, steps_per_iteration=1, fig_size=(20, 20)):
    pos = forceatlas2.forceatlas2_networkx_layout(G, pos=None, iterations=50)
    node_label = {k: k for k, d in dict(G.degree).items() if d > 50}
    node_size = [node_degree_to_size(d) for k, d in dict(G.degree).items()]

    plt.rcParams["figure.figsize"] = fig_size

    def update(data):
        print(f"{1 + data} / {frames}")
        global pos
        fig.clear()
        pos = forceatlas2.forceatlas2_networkx_layout(G, pos=pos, iterations=steps_per_iteration,
                                                      weight_attr=None)  # TODO: weight attr

        nx.draw_networkx_nodes(G, pos, alpha=1, node_color=None, node_size=node_size)
        # nx.draw_networkx_edges(G, pos, edge_color="green", alpha=0.15) # Time consuming
        nx.draw_networkx_labels(G, pos, labels=node_label, font_size=10, font_color='black')

    fig = plt.gcf()
    ani = matplotlib.animation.FuncAnimation(fig, update,
                                             frames=frames,
                                             interval=interval,
                                             repeat=False,
                                             blit=False)
    ani.save(str(DATA_PATH / "full2.gif"))
