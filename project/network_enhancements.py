import networkx as nx
from community import community_louvain


def add_node_degrees(G: nx.DiGraph) -> nx.DiGraph:
    in_degrees = {node: G.in_degree(node) for node in G.nodes()}
    out_degrees = {node: G.out_degree(node) for node in G.nodes()}
    nx.set_node_attributes(G, in_degrees, "in_degree")
    nx.set_node_attributes(G, out_degrees, "out_degrees")
    return G


def add_community_id(G: nx.DiGraph) -> nx.DiGraph:
    G_undir = G.to_undirected()
    partition_ids = community_louvain.best_partition(G_undir)
    nx.set_node_attributes(G, partition_ids, "partition_id")
    return G
