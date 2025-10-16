import argparse
import os
import numpy as np
import torch
import torch_geometric.transforms as T
from deeprobust.graph.data import Pyg2Dpr
import scipy.sparse as sp
from scipy.sparse import csr_matrix
from deeprobust.graph.targeted_attack import Nettack, FGA
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack
import sys
import pandas as pd


print(torch.__version__)
print(torch.version.cuda)

#utility functions
from utils import load_graph, select_nodes, split_data, check_and_resample, write_result_csv
from model import GAT, H2GCN


def undirected_edge_count(adj_csr):
    # counts unique edges even if self-loops exist
    diag = adj_csr.diagonal().sum()
    return int((adj_csr.nnz - diag)//2 + diag)



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
    parser.add_argument("--model", choices=["gcn", "gat", "h2gcn"], default="gcn",
                    help="Classifier to train/evaluate (default: gcn).")

    parser.add_argument("--attack", choices=["clean", "metattack"], default="clean",
                    help="Attack type to run (default: clean).")
    
    args = parser.parse_args()

    #save results under following filename
    filename=args.output


    # Kcora = [0,400,500,600,700,800,1000,1400,1900,3000,5000]
    # KChameleon = [0, 600,700,800,1000,1400,1900,3000,5000, 6000,7000]

    # if args.dataset == 'Cora':
    #     K = Kcora
    #     num_edges = 5069 #initial no of edges
    # elif args.dataset == 'Chameleon':
    #     K = KChameleon
    # else:
    #     print(f"No rewiring level found for {args.dataset}")
    #     sys.exit(1)

    #read rewiring level
    data_stat = pd.read_csv('generated_datasets/stats.csv')
    current_dataset = data_stat[data_stat['dataset'] == args.dataset]
    K = current_dataset['K']

    #get total no of edges of current dataset
    num_edges = current_dataset[current_dataset['K']==0]['total_edges'].iloc[0]
    
    attack = args.attack    
    seed = args.seed

    
    #budgets = [0.10,0.15,0.20, 0.25]
    #budget = 5% of total edges
    budgets = [0.05]
    
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    
    print("Using device: ", device)


    #seed is for different initialization
    transform = T.Compose([lambda x : split_data(x, seed)])


    #loop through each generated graph data
    
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

        #test attacks assumptions on data
        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"
        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"
                
        if args.attack == "clean":
            print("Evaluating clean accuracy only...")
            idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        
            clf = make_model(args.model, nfeat=features.shape[1],
                             nclass=labels.max().item() + 1, device=device)
            clf.fit(features, adj, labels, idx_train, idx_val, patience=30)
            clean_acc = clf.test(idx_test)
        
            write_result_csv(filename, args.dataset, seed, k,
                             args.model, "clean", 0, 1 - clean_acc, homophily_before)

        elif args.attack == "metattack":
            print("Performing Metattack...")
            idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
            idx_unlabeled = np.union1d(idx_val, idx_test)
        
            clf = make_model(args.model, nfeat=features.shape[1],
                             nclass=labels.max().item() + 1, device=device)
        
            # surrogate stays GCN
            surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                            nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
            surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
        
            num_edges += k

            for budgetp in budgets:
                
                budget = int(budgetp * num_edges)

                print(f"Here budget is {budget}")
            
                clf.initialize()
                attacker = Metattack(surrogate, nnodes=adj.shape[0], feature_shape=features.shape,
                                     attack_structure=True, attack_features=False, device=device, lambda_=0).to(device)
                attacker.attack(features, adj, labels, idx_train, idx_unlabeled,
                                n_perturbations=budget, ll_constraint=False)
                mod_adj = attacker.modified_adj
            
                features_t = torch.tensor(features) if not isinstance(features, torch.Tensor) else features
                labels_t = torch.tensor(labels) if not isinstance(labels, torch.Tensor) else labels
            
                clf.fit(features_t, mod_adj, labels_t, idx_train, idx_val, patience=30)
                attacked_acc = clf.test(idx_test)
            
                misclassification_rate = 1 - attacked_acc
                homophily_after = calculate_homophily(mod_adj, labels_t)
            
                write_result_csv(filename, args.dataset, seed, k,
                                 args.model, "metattack", budgetp, misclassification_rate, homophily_after)
