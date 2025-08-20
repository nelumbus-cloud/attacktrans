import argparse
import os
import numpy as np
import torch


print(torch.__version__)
print(torch.version.cuda)

from utils import load_graph, select_nodes, split_data, check_and_resample, write_result_csv
import torch_geometric.transforms as T
from deeprobust.graph.data import Pyg2Dpr
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from deeprobust.graph.targeted_attack import Nettack, FGA
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
import sys

def undirected_edge_count(adj_csr):
    # counts unique edges even if self-loops exist
    diag = adj_csr.diagonal().sum()
    return int((adj_csr.nnz - diag)//2 + diag)



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


@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    test_correct = pred[data.test_mask] == data.y[data.test_mask]
    test_acc = int(test_correct.sum()) / int(data.test_mask.sum())
    return test_acc

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Attack on graphdata.")
    parser.add_argument("--dataset",required=True,  help="The name of the dataset (e.g., Cora).")
    parser.add_argument("--seed", type = int,required=True, help="Random seed for reproducibility e.g 22")
    parser.add_argument("--output",required=True, help="Output csv file name")
    args = parser.parse_args()
    filename=args.output


    Kcora = [0,400,500,600,700,800,1000,1400,1900,3000,5000]
    #KChameleon = [0, 600,700,800,1000,1400,1900,3000,5000, 6000,7000]
    K = Kcora

    attack = 'Metattack'
    #attack = 'Netattack'
    
    seed = args.seed
    budgets = [5,6,10]
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    print("Using device: ", device)


    #seed is for different initialization
    transform = T.Compose([lambda x : split_data(x, seed)])


    #loop through each generated graph data
    num_edges = 5069
    for k in K:
        print(f"Loading dataset {args.dataset} with K = {k}")
        data = load_graph(args.dataset, k)
        print("Data is un directed", data.is_directed())
        #split
        data = transform(data)
        #to dpr
        data = Pyg2Dpr(data)
        
        features, labels, adj = data.features, data.labels, data.adj

        homophily_before = calculate_homophily(adj, labels)
        

    # Convert to CSR format once before the loop for efficiency
        adj_csr = csr_matrix(adj)
        features_csr = csr_matrix(features)

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"
        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"
        
    
        if attack == "Netattack":

            print("\n--- Performing Netattack ---")
                        
            gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device).to(device)
            
            surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item() + 1,
                nhid=16, dropout=0,
                with_relu=False, 
                with_bias=False,
                device=device)
            
            surrogate.fit(features, adj, labels, data.idx_train, data.idx_val, patience=30)
            #pick target nodes from test set folloiwng netattack/gottack paper
            gcn.fit(features, adj, labels, data.idx_train, data.idx_val, patience=30)
            target_nodes = select_nodes(gcn, data.idx_test, labels)
            
            for d in budgets:
                nettack_correct = 0
                for t in target_nodes:
                    gcn.initialize()
                    attacker = Nettack(surrogate,nnodes=adj.shape[0],attack_structure=True,attack_features=True,device=device).to(device)
                    attacker.attack(features_csr, adj_csr, labels, t, n_perturbations=d, verbose=False)
                    mod_adj, mod_feat = attacker.modified_adj, attacker.modified_features
                    gcn.fit(mod_feat, mod_adj, labels, data.idx_train, data.idx_val, patience=30)
                    gcn.eval()
                    logits = gcn.predict()
                    nettack_correct += (torch.argmax(logits[t]) == labels[t]).item()
                    attacker.reset()
                misclassification_rate = 1-nettack_correct/len(target_nodes)
                write_result_csv(filename, args.dataset, seed, k, "gcn", attack, d, misclassification_rate)


        elif attack == "Metattack":
            #run global attack metattackk
            print("Performing Metttack")
            idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
            idx_unlabeled = np.union1d(idx_val, idx_test)
            idx_unlabeled = np.union1d(idx_val, idx_test)

            gcn = GCN(nfeat=features.shape[1], nhid=16, nclass=labels.max().item() + 1, dropout=0.5, device=device).to(device)
            
            #save results of clean graph
            gcn.fit(features, adj, labels, data.idx_train, data.idx_val, patience=30)
            clean_acc = gcn.test(idx_test)
            
            write_result_csv(filename, args.dataset, seed, k, "gcn", "clean", 0, 1 - clean_acc, homophily_before)
            
            #setup surrogate
            
            surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
            surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
            
            num_edges+=k
            budget = int(0.05*num_edges)
            gcn.initialize()
            #attack 
            attacker = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                    attack_structure=True, attack_features=False, device=device, lambda_=0).to(device)
            attacker.attack(features, adj, labels, idx_train, idx_unlabeled, n_perturbations=budget, ll_constraint=False)
            mod_adj = attacker.modified_adj
            #issue with library
            features = torch.tensor(features)
            labels = torch.tensor(labels)
            gcn.fit(features, mod_adj, labels, data.idx_train, data.idx_val, patience=30)
            gcn.eval()
            
            a = gcn.test(idx_test)
            
            misclassification_rate = 1-a

            homophily_after = calculate_homophily(mod_adj,labels)
            
            write_result_csv(filename, args.dataset, seed, k, "gcn", attack, budget, misclassification_rate, homophily_after)

    