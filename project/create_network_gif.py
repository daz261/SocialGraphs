import pandas as pd

import networkx as nx

from project.util import DATA_PATH

if __name__ == '__main__':
    df = pd.read_csv(DATA_PATH / "updated_artist_matches.csv")
    print()