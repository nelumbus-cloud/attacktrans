import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from collections import defaultdict
import torch
from torch_geometric.datasets import Planetoid
from torch_geometric.utils import to_undirected

# Output directory
output_dir = "tsne_outputs"
os.makedirs(output_dir, exist_ok=True)

# Use CUDA if available
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def generate_tsne_plot(dataset_name, perplexity=30, metric='cosine'):
    print(f"Processing {dataset_name}...")
    
    # Load dataset and move to device
    dataset = Planetoid(root=f'data/{dataset_name}', name=dataset_name)
    data = dataset[0].to(device)
    data.edge_index = to_undirected(data.edge_index)

    labels = data.y.cpu().numpy()  # Convert to numpy on CPU
    num_nodes = data.num_nodes
    num_classes = dataset.num_classes

    # Build adjacency list
    adj_list = defaultdict(list)
    edge_list = data.edge_index.cpu().T.tolist()
    for i, j in edge_list:
        adj_list[i].append(j)

    # Compute neighborhood distributions
    all_classes = sorted(list(np.unique(labels)))
    class_to_idx = {cls: i for i, cls in enumerate(all_classes)}
    neighborhood_distributions = []

    for i in range(num_nodes):
        neighbors = adj_list[i]
        dist_vec = np.zeros(len(all_classes))

        if not neighbors:
            dist_vec[:] = 1.0 / len(all_classes)
        else:
            for j in neighbors:
                dist_vec[class_to_idx[labels[j]]] += 1
            dist_vec /= dist_vec.sum() if dist_vec.sum() > 0 else 1.0

        neighborhood_distributions.append(dist_vec)

    neighborhood_distributions = np.array(neighborhood_distributions)

    # Apply t-SNE
    tsne = TSNE(
        n_components=2,
        metric=metric,
        perplexity=perplexity,
        init='pca',
        random_state=42
    )
    tsne_results = tsne.fit_transform(neighborhood_distributions)

    # Plot

        # Plot with color + shape for accessibility
    colors = plt.cm.get_cmap('tab10', num_classes)
    markers = ['o', 'x', '^', 's', 'P', 'D', '*', 'v', '>', '<']  # Up to 10 classes

    plt.figure(figsize=(10, 8))
    for i, cls in enumerate(all_classes):
        idx = labels == cls
        marker = markers[i % len(markers)]
        plt.scatter(tsne_results[idx, 0], tsne_results[idx, 1],
                    s=30, alpha=0.7, color=colors(i), marker=marker, label=f'Class {cls}')

    plt.title(f'{dataset_name}: t-SNE of Neighborhood Distributions\nPerplexity={perplexity}, Metric={metric}')
    plt.xticks([])
    plt.yticks([])
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(markerscale=1.5, fontsize=8, loc='best')

    save_path = os.path.join(output_dir, f'tsne_{dataset_name.lower()}.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved to {save_path}")

# Run for all three datasets
for name in ['Cora', 'CiteSeer', 'PubMed']:
    generate_tsne_plot(name)
