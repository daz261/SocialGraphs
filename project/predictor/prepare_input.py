import networkx as nx


def prepare_input(G, node_name1, node_name2):
    node1 = G[node_name1]
    node2 = G[node_name2]

    are_neighbors = node2 in node1.neighbors
    # TODO: tone difference etc


if __name__ == '__main__':
    G = nx.read_gpickle("../G.pickle")
    print()
