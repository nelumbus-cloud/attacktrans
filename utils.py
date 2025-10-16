import os
import sys
import csv
import argparse
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import scipy.sparse as sp
from scipy.sparse import csr_matrix

import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.utils import to_undirected

from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from deeprobust.graph.utils import get_train_val_test
from deeprobust.graph.targeted_attack import Nettack, FGA
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
from model import GAT, H2GCN





def make_model(model_name, nfeat, nclass, device, **kwargs):
    model_name = model_name.lower()
    if model_name == "gcn":
        return GCN(nfeat=nfeat, nhid=16, nclass=nclass, dropout=0.5, device=device).to(device)
    elif model_name == "gat":
        return GAT(nfeat=nfeat, nclass=nclass, device=device, **kwargs)
    elif model_name == "h2gcn": 
        return H2GCN(nfeat=nfeat, nclass=nclass, device=device,
                     hidden=64, k=2, dropout=0.5, use_relu=True, lr=0.01, weight_decay=5e-4)
    else:
        raise ValueError(f"Unsupported model: {model_name}")




def load_graph(dataset_name, k, save_dir='generated_datasets'):
    file_path = os.path.join(save_dir, f"{dataset_name}_{k}.npz")
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"No saved graph found at {file_path}")

    with np.load(file_path) as loaded:
        x = torch.from_numpy(loaded['x']).float()
        y = torch.from_numpy(loaded['y']).long()
        edge_index = torch.from_numpy(loaded['edge_index']).long()
        train_mask = torch.from_numpy(loaded['train_mask']).bool()
        val_mask = torch.from_numpy(loaded['val_mask']).bool()
        test_mask = torch.from_numpy(loaded['test_mask']).bool()

    data = Data(x=x, y=y, edge_index=edge_index,
                train_mask=train_mask, val_mask=val_mask, test_mask=test_mask)
    return data


    
def _to_scipy_coo(A):
    """Coerce adjacency to SciPy COO regardless of whether it's SciPy, NumPy, or torch."""
    if sp.issparse(A):
        return A.tocoo()
    if isinstance(A, np.ndarray):
        return csr_matrix(A).tocoo()
    if isinstance(A, torch.Tensor):
        # supports dense or sparse torch tensors on cpu/gpu
        if A.is_sparse:
            A = A.coalesce()
            idx = A.indices().cpu().numpy()
            data = A.values().cpu().numpy()
            n = A.size(0)
            return sp.coo_matrix((data, (idx[0], idx[1])), shape=(n, n))
        else:
            return csr_matrix(A.detach().cpu().numpy()).tocoo()
    raise TypeError(f"Unsupported adjacency type: {type(A)}")



def calculate_homophily(adj, labels):
    """
    Homophily = fraction of non-self-loop edges that connect same-class nodes.
    Accepts adj as SciPy sparse, NumPy array, or torch.Tensor (dense or sparse).
    Accepts labels as NumPy array or torch.Tensor.
    """
    if isinstance(labels, torch.Tensor):
        labels = labels.detach().cpu().numpy()

    adj = _to_scipy_coo(adj)
    row, col = adj.row, adj.col

    # remove self-loops
    mask = row != col
    row, col = row[mask], col[mask]

    total_edges = len(row)
    if total_edges == 0:
        return 0.0

    same_class_edges = np.sum(labels[row] == labels[col])
    return float(same_class_edges) / float(total_edges)


# split data to same as attack papers 20/60/60
def split_data(data, seed):
    train_idx, val_idx, test_idx = get_train_val_test(
        data.num_nodes, stratify=data.y, seed=seed)

    # Create boolean masks
    data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)


    data.train_mask[train_idx] = True
    data.val_mask[val_idx] = True
    data.test_mask[test_idx] = True
        
    return data



def classification_margin(output, true_label):
    '''probs_true_label - probs_best_second_class'''
    probs = torch.exp(output)
    probs_true_label = probs[true_label].clone()
    probs[true_label] = 0
    probs_best_second_class = probs[probs.argmax()]
    return (probs_true_label - probs_best_second_class).item()


def select_nodes(target_gcn, idx_test, labels):
        '''
        
        selecting nodes as reported in nettack paper:
        (i) the 10 nodes with highest margin of classification, i.e. they are clearly correctly classified,
        (ii) the 10 nodes with lowest margin (but still correctly classified) and
        (iii) 20 more nodes randomly
        '''
    
        output = target_gcn.predict()
    
        margin_dict = {}
        for idx in idx_test:
            margin = classification_margin(output[idx], labels[idx])
            if margin < 0: # only keep the nodes correctly classified
                continue
            margin_dict[idx] = margin
        sorted_margins = sorted(margin_dict.items(), key=lambda x:x[1], reverse=True)
        high = [x for x, y in sorted_margins[: 10]]
        low = [x for x, y in sorted_margins[-10: ]]
        other = [x for x, y in sorted_margins[10: -10]]
        other = np.random.choice(other, 20, replace=False).tolist()
    
        return high + low + other

def check_and_resample( degrees, targets, idx_test, verbose=False):

        

        new_targets = []
        
        candidates = list(set(idx_test) - set(targets))
        
        for target in targets:
            t = target
            while degrees[t] <= 1:
                if verbose:
                    print(f"{target} is removed")
                new_target = np.random.choice(candidates)
                candidates.remove(new_target)
                t = new_target
                
            new_targets.append(t)

        if verbose:
            print("Target Nodes", targets)
            print("New targets Nodes",new_targets)
            
        return new_targets



def write_result_csv(filename, dataset_name, seed, K, model, attack_model, budget, misclassification_rate, homophily):
    file_exists = os.path.isfile(filename)
    current_datetime = datetime.now()
    with open(filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        # write header if file doesn't exist
        if not file_exists:
            writer.writerow(["date", "dataset_name", "seed", "K", "model", "attack_model", "budget",  "misclassification_rate", "homophily"])
        writer.writerow([current_datetime, dataset_name, seed, K, model, attack_model, budget, misclassification_rate, homophily])
 




def write_result_csv2(filename, dataset, seed, model, attack, budget,
                     mis_rate_before, mis_rate_after,
                     homophily_before, homophily_after):
    """
    Append experiment results to a CSV file.
    """

    header = [
        "timestamp", "dataset", "seed", "model", "attack", "budget",
        "mis_rate_before", "mis_rate_after",
        "homophily_before", "homophily_after"
    ]

    # Current timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Define the row
    row = [
        timestamp, dataset, seed, model, attack, budget,
        mis_rate_before, mis_rate_after,
        homophily_before, homophily_after
    ]

    file_exists = os.path.isfile(filename)
    
    with open(filename, mode="a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow(row)


if __name__ == '__main__':
    load_graph('Cora', 400)