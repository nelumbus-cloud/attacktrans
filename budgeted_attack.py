from utils import split_data, calculate_homophily, make_model, write_result_csv2
import torch_geometric.transforms as T
from torch_geometric.utils import subgraph
from deeprobust.graph.data import Pyg2Dpr
import numpy as np
from collections import Counter
import torch
from deeprobust.graph.defense import GCN
from deeprobust.graph.global_attack import Metattack, MinMax




'''
1. Get datasets


Cornell Wisconsin Texas Film Chameleon Squirrel Cora CiteSeer PubMed

2. Models : GAT, GCN, H2GCN

3. budgets: 5%, 10%, 15%, 20%


'''

#should return features, labels, adj 
#setups
datasets = ["Cornell", "Wisconsin", "Texas", "Film", "Chameleon", "Squirrel", "Cora", "CiteSeer", "PubMed"]
#mask can be used to select the dataset you want to run

mask = [False,False,False, False, True,False,False,False,False]

assert len(mask) == len(datasets)

budgets = [0.05, 0.1, 0.15, 0.20]

def get_data(dataset_name, seed):
    """
    Load dataset based on name and return the data object.
    """
    print(f"Loading dataset: {dataset_name}")
    
    if dataset_name in ["Cora", "CiteSeer", "PubMed"]:
        from torch_geometric.datasets import Planetoid
        dataset = Planetoid(root=f"./data/{dataset_name}", name=dataset_name)
    
    elif dataset_name in ["Cornell", "Wisconsin", "Texas"]:
        from torch_geometric.datasets import WebKB
        dataset = WebKB(root=f"./data/{dataset_name}", name=dataset_name)
    
    elif dataset_name in ["Chameleon", "Squirrel", "Film"]:
        from torch_geometric.datasets import WikipediaNetwork, Actor
        if dataset_name in ["Chameleon", "Squirrel"]:
            dataset = WikipediaNetwork(root=f"./data/{dataset_name}", name=dataset_name)
        else:  # Film
            dataset = Actor(root="./data/Film")

    
    else:
        raise ValueError(f"Dataset {dataset_name} not recognized!")

    data = dataset[0]

    

    #remove underpopulated classes

    counts = torch.bincount(data.y)

    rare_classes = (counts < 2).nonzero(as_tuple=True)[0]
    
    if rare_classes.numel() != 0:
        print("Rare classes:", rare_classes.tolist())
        mask = ~torch.isin(data.y, rare_classes)
        new_edge_index, _ = subgraph(mask.nonzero(as_tuple=True)[0], data.edge_index, relabel_nodes=True)
        data.edge_index = new_edge_index
        data.y = data.y[mask]
        data.x = data.x[mask]
        data.num_nodes = mask.sum().item()

            
    transform = T.Compose([T.ToUndirected(), T.LargestConnectedComponents(), lambda x : split_data(x, seed)])

    data = transform(data)
    data = Pyg2Dpr(data)

    return data


def main():
    seed = 22
    models = ['GCN', 'GAT', 'H2GCN']

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    filename = "result_minmax_realdataset.csv"

    
    for dataset in np.array(datasets)[mask]:
        data = get_data(dataset, seed)
        
        features, labels, adj = data.features, data.labels, data.adj
        homophily_before = calculate_homophily(adj, labels)


        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"
        assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
        idx_unlabeled = np.union1d(idx_val, idx_test)

         # surrogate stays GCN
        surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                        nhid=16, dropout=0, with_relu=False, with_bias=False, device=device).to(device)
        surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)

        num_edges = int(adj.sum() // 2) 
        
        for model in models:
            
            clf = make_model(model, nfeat=features.shape[1],
                             nclass=labels.max().item() + 1, device=device)
    
            #add clean misclass rate.
            clf.fit(features, adj, labels, idx_train, idx_val, patience=30)
            clean_acc = clf.test(idx_test)
            mis_rateb = 1 - clean_acc
            
            for b in budgets:
                    budget = int(b * num_edges)
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
                
                    mis_ratea = 1 - attacked_acc
                    homophily_after = calculate_homophily(mod_adj, labels_t)
                
                    write_result_csv2(filename, dataset, seed, model, "metattack", b, mis_rateb, mis_ratea, homophily_before, homophily_after)


main()