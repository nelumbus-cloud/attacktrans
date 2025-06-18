# explore_graphs.py
import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_networkx
from torch.nn.functional import cosine_similarity
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
    """
    Computes a comprehensive set of statistics for a given graph.
    """
    G = to_networkx(data, to_undirected=True)
    degrees = [d for _, d in G.degree()]
    clustering = nx.clustering(G)
    triangles = sum(nx.triangles(G).values()) // 3
    components = list(nx.connected_components(G))
    largest_cc = max(components, key=len)
    density = nx.density(G)

    try:
        # These can be slow on large graphs and fail on disconnected ones
        sp_length = nx.average_shortest_path_length(G.subgraph(largest_cc))
        diameter = nx.diameter(G.subgraph(largest_cc))
    except:
        sp_length = float('nan')
        diameter = float('nan')

    pr = nx.pagerank(G)
    bc = nx.betweenness_centrality(G)

    # ---  Homophily (Edge-based) ---
    row, col = data.edge_index
    y = data.y
    edge_homophily = (y[row] == y[col]).sum().item() / row.size(0)

    # --- Count nodes with degree > 1 ---
    nodes_deg_gt_1 = sum(1 for d in degrees if d > 1)

    # ---  Calculate Node Homophily (Feature-based) ---
    # This implements hi = CosSim(Xi, sum_{j in Ni} (1/sqrt(dj*di)) * Xj)
    # It measures the similarity between a node's features and the
    # aggregated, normalized features of its neighbors.
    x = data.x
    deg_tensor = torch.tensor(degrees, dtype=torch.float)
    
    # Handle division by zero for isolated nodes if any exist
    deg_inv_sqrt = deg_tensor.pow(-0.5)
    deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0


    row, col = data.edge_index
    norm_vals = deg_inv_sqrt[row] * deg_inv_sqrt[col]
    adj_norm = torch.sparse_coo_tensor(data.edge_index, norm_vals, data.size())
    
    # Get the aggregated neighbor features for all nodes at once
    aggregated_neighbor_features = torch.sparse.mm(adj_norm, x)

    # Calculate the cosine similarity between each node's features and its aggregated neighbor features
    node_homophilies = cosine_similarity(x, aggregated_neighbor_features, dim=1)
    avg_node_homophily = node_homophilies.mean().item()


    return {
        "Dataset": name,
        "Num Nodes": G.number_of_nodes(),
        "Num Edges": G.number_of_edges(),
        "Avg Degree": np.mean(degrees),
        "Max Degree": np.max(degrees),
        "Nodes with Degree > 1": nodes_deg_gt_1, # New Metric
        "Clustering Coef (avg)": np.mean(list(clustering.values())),
        "Triangles": triangles,
        "Connected Components": len(components),
        "Largest Component Size": len(largest_cc),
        "Density": density,
        "Avg Shortest Path": sp_length,
        "Diameter": diameter,
        "Edge Homophily (Label-based)": edge_homophily, # Renamed for clarity
        "Avg Node Homophily (Feature-based)": avg_node_homophily, # New Metric
        "Avg Betweenness": np.mean(list(bc.values())),
        "Avg PageRank": np.mean(list(pr.values()))
    }

# Compute stats and save to CSV
stats_rows = [compute_graph_stats(name, data) for name, data in datasets.items()]
pd.DataFrame(stats_rows).to_csv("exploration_outputs/statistics.csv", index=False)

print("Graph statistics saved to exploration_outputs/statistics.csv")

# ========== Class Distribution Plots ==========
for name, data in datasets.items():
    counts = torch.bincount(data.y)
    plt.figure(figsize=(10, 6))
    plt.bar(torch.arange(len(counts)).numpy(), counts.numpy())
    plt.title(f"{name} - Class Distribution", fontsize=16)
    plt.xlabel("Class", fontsize=12)
    plt.ylabel("Number of Nodes", fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.savefig(f"exploration_outputs/class_distribution_{name.lower()}.png")
    plt.close()

print("Class distribution plots saved to exploration_outputs/")
