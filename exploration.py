# explore_graphs.py
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx
import networkx as nx

# Create output directory
os.makedirs("exploration_outputs", exist_ok=True)

# Load datasets
datasets = {
    "Cora": Planetoid(root='data/Cora', name='Cora', transform=NormalizeFeatures())[0],
    "Chameleon": WikipediaNetwork(root='data/Chameleon', name='chameleon', transform=NormalizeFeatures())[0]
}

# ========== Graph Statistics ==========
def compute_graph_stats(name, data):
    G = to_networkx(data, to_undirected=True)
    degrees = [d for _, d in G.degree()]
    clustering = nx.clustering(G)
    triangles = sum(nx.triangles(G).values()) // 3
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    density = nx.density(G)

    try:
        sp_length = nx.average_shortest_path_length(G.subgraph(largest_cc))
        diameter = nx.diameter(G.subgraph(largest_cc))
    except:
        sp_length = float('nan')
        diameter = float('nan')

    pr = nx.pagerank(G)
    bc = nx.betweenness_centrality(G)

    row, col = data.edge_index
    y = data.y
    homophily = (y[row] == y[col]).sum().item() / row.size(0)

    return {
        "Dataset": name,
        "Num Nodes": G.number_of_nodes(),
        "Num Edges": G.number_of_edges(),
        "Avg Degree": np.mean(degrees),
        "Max Degree": np.max(degrees),
        "Clustering Coef (avg)": np.mean(list(clustering.values())),
        "Triangles": triangles,
        "Connected Components": len(components),
        "Largest Component Size": len(largest_cc),
        "Density": density,
        "Avg Shortest Path": sp_length,
        "Diameter": diameter,
        "Homophily": homophily,
        "Avg Betweenness": np.mean(list(bc.values())),
        "Avg PageRank": np.mean(list(pr.values()))
    }

stats_rows = [compute_graph_stats(name, data) for name, data in datasets.items()]
pd.DataFrame(stats_rows).to_csv("exploration_outputs/statistics.csv", index=False)

# ========== Class Distribution Plots ==========
for name, data in datasets.items():
    counts = torch.bincount(data.y)
    plt.figure()
    plt.bar(torch.arange(len(counts)).numpy(), counts.numpy())
    plt.title(f"{name} - Class Distribution")
    plt.xlabel("Class")
    plt.ylabel("Count")
    plt.savefig(f"exploration_outputs/class_distribution_{name.lower()}.png")
    plt.close()
