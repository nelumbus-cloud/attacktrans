import torch
from torch_geometric.datasets import Planetoid, WikipediaNetwork
from torch_geometric.utils import to_undirected
import os
import torch_geometric.transforms as T
import numpy as np
import random
import argparse
import sys
import csv

def calculate_homophily(edge_index, y, num_edges):
    same_label_edges = 0
    for i in range(edge_index.shape[1]):
        u, v = edge_index[0, i].item(), edge_index[1, i].item()
        if y[u] == y[v]:
            same_label_edges += 1
    return same_label_edges / num_edges

def add_heterophilous_edges(data, K, V_c, D_c):
    num_nodes = data.num_nodes
    y = data.y
    
    original_edges = set()
    for i in range(data.edge_index.shape[1]):
        u, v = data.edge_index[0, i].item(), data.edge_index[1, i].item()
        original_edges.add(tuple(sorted((u, v))))

    new_edges_list = []
    class_labels = list(D_c.keys())

    edges_to_add_count = 0
    while edges_to_add_count < K:
        i = random.randint(0, num_nodes - 1)
        yi = y[i].item()

        target_c = np.random.choice(class_labels, p=D_c[yi])

        if not V_c[target_c]:
            continue
        
        j = random.choice(V_c[target_c])

        if i == j or tuple(sorted((i, j))) in original_edges:
            continue
            
        new_edges_list.append([i, j])
        new_edges_list.append([j, i])
        original_edges.add(tuple(sorted((i, j))))
        edges_to_add_count += 1
        #debug for large edges
        if edges_to_add_count > 0 and edges_to_add_count % 2000 == 0:
            print(f"  ... {edges_to_add_count} / {K} edges generated.")

    print(f"\nSuccessfully generated {edges_to_add_count} new unique edges.")
    
    new_edges_tensor = torch.tensor(new_edges_list, dtype=torch.long).t()
    new_total_edge_index = torch.cat([data.edge_index, new_edges_tensor], dim=1)
    
    return new_total_edge_index

#transform to undirected and LC
def transform_graph(data):
    data.edge_index = to_undirected(data.edge_index)
    return data


def save_graph(data, dataset_name, k, save_dir='generated_datasets', save=True):
    """
    Saves the generated graph data to a compressed .npz file.
    """
    # Create the directory if it doesn't exist
    

    #Save all relevant graph attributes as numpy arrays

    homophily = calculate_homophily(data.edge_index, data.y, data.num_edges)
    print(f"Homophily level {homophily}")

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        file_path = os.path.join(save_dir, f"{dataset_name}_{k}.npz")
        np.savez_compressed(
            file_path,
            x=data.x.cpu().numpy(),
            y=data.y.cpu().numpy(),
            edge_index=data.edge_index.cpu().numpy(),
            train_mask=data.train_mask.cpu().numpy(),
            val_mask=data.val_mask.cpu().numpy(),
            test_mask=data.test_mask.cpu().numpy(),
        )
        print(f"Graph saved to {file_path}")    
        #write header if file doesn't exist
        csv_path = os.path.join(save_dir, "stats.csv")
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a", newline="") as f:
            writer = csv.writer(f)
            if write_header:
                writer.writerow(["dataset", "K", "total_edges", "homophily"])
            writer.writerow([dataset_name, k, data.num_edges/2, homophily])
   
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate heterophilous graphs based on the paper 'Is Homophily a Necessity for Graph Neural Networks?'.")
    parser.add_argument("--dataset", required=True, help="The name of the dataset to use.", choices=["Cora", "Citeseer", "Chameleon", "Squirrel"])
    args = parser.parse_args()

    dataset_name = args.dataset
    
    random.seed(22)

    transform = T.Compose([T.ToUndirected(), T.LargestConnectedComponents()])
    
    if dataset_name in ["Cora", "Citeseer"]:
        dataset = Planetoid(root=f'data/{dataset_name}', name=dataset_name, transform=transform)
    else:
        dataset = WikipediaNetwork(root=f'data/{dataset_name}', name=dataset_name, transform=transform)
        
    data = dataset[0]

    print(f"--- Original {dataset_name} Dataset Stats ---")
    print(f"Nodes: {data.num_nodes}")
    print(f"Edges: {data.num_edges}")
    print(f"Classes: {dataset.num_classes}")
    
    original_h = calculate_homophily(data.edge_index, data.y, data.num_edges)
    print(f"Calculated Homophily (h): {original_h:.4f}")
    print("-" * 35)

    K, D_c = [], None

    if dataset_name == "Cora":
        K = [400,500,600,700,800,1000,1400,1900,3000,5000]
        D_c = {
            0: [0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5],
            1: [0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0],
            2: [0.0, 0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
            3: [0.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
            4: [0.0, 0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
            5: [0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
            6: [0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
        }
    elif dataset_name == "Chameleon":
        K = [600,700,800,1000,1400,1900,3000,5000, 6000, 7000]
        D_c = {
            0: [0.0, 0.5, 0.0, 0.0, 0.5],
            1: [0.5, 0.0, 0.5, 0.0, 0.0],
            2: [0.0, 0.5, 0.0, 0.5, 0.0],
            3: [0.0, 0.0, 0.5, 0.0, 0.5],
            4: [0.5, 0.0, 0.0, 0.5, 0.0],
        }

    elif dataset_name == "Squirrel":
        K = [600,700,800,1000,1400,1900,3000,5000, 6000, 7000]
        D_c = {
            0: [0.0, 0.5, 0.0, 0.0, 0.5],
            1: [0.5, 0.0, 0.5, 0.0, 0.0],
            2: [0.0, 0.5, 0.0, 0.5, 0.0],
            3: [0.0, 0.0, 0.5, 0.0, 0.5],
            4: [0.5, 0.0, 0.0, 0.5, 0.0],
        }

    elif dataset_name == "Citeseer":
        K = [400,500,600,700,800,1000,1400,1900,3000,5000]
        D_c = {
            0: [0.0, 0.5, 0.0, 0.0, 0.0, 0.5],
            1: [0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
            2: [0.0, 0.5, 0.0, 0.5, 0.0, 0.0],
            3: [0.0, 0.0, 0.5, 0.0, 0.5, 0.0],
            4: [0.0, 0.0, 0.0, 0.5, 0.0, 0.5],
            5: [0.5, 0.0, 0.0, 0.0, 0.5, 0.0],
        }

    else:
        print("No Dc known")
        sys.exit(0)
    
    #class to nodes
    V_c = {i: [] for i in range(dataset.num_classes)}
    for node_idx, label in enumerate(data.y):
        V_c[label.item()].append(node_idx)

    save_graph(data,dataset_name, 0)
    #sys.exit(0)
    
    for k in K:
        print(f"\nStarting graph generation: Adding K={k} edges to {dataset_name}...")
        generated_edge_index = add_heterophilous_edges(data, k, V_c, D_c)
        data.edge_index = generated_edge_index
        save_graph(data,dataset_name, k)
       
