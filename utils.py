import os
import numpy as np
import torch
from torch_geometric.data import Data
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr
from scipy.sparse import csr_matrix
import torch_geometric.transforms as T
import torch
import numpy as np
from deeprobust.graph.utils import get_train_val_test
from torch_geometric.utils import to_undirected
from datetime import datetime
import csv

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
 

if __name__ == '__main__':
    load_graph('Cora', 400)