from deeprobust.graph.targeted_attack import Nettack
from datasets import CoraDataset, ChameleonDataset

from deeprobust.graph.defense import GCN


def setup():
    data = CoraDataset()
    data.load()
    data = data.get_dpr()
    adj, features, labels = data.adj, data.features, data.labels
    return adj, features, labels


# idx_train, idx_val, idx_test = data.idx_train, data.idx_val, data.idx_test
# >>> # Setup Surrogate model
# >>> surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
#                 nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
# >>> surrogate.fit(features, adj, labels, idx_train, idx_val, patience=30)
# >>> # Setup Attack Model
# >>> target_node = 0
# >>> model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device='cpu').to('cpu')
# >>> # Attack
# >>> model.attack(features, adj, labels, target_node, n_perturbations=5)
# >>> modified_adj = model.modified_adj
# >>> modified_features = model.modified_features

def run_attack():
    adj, features, labels = setup()
    # Setup Surrogate model
    surrogate = GCN(nfeat=features.shape[1], nclass=labels.max().item()+1,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device='cpu').to('cpu')
    surrogate.fit(features, adj, labels, patience=30)
    # Setup Attack Model
    target_node = 0  # You can change this to any node index you want to attack
    model = Nettack(surrogate, nnodes=adj.shape[0], attack_structure=True, attack_features=True, device='cpu').to('cpu')
    # Attack
    model.attack(features, adj, labels, target_node, n_perturbations=5)
    modified_adj = model.modified_adj
    modified_features = model.modified_features
    return modified_adj, modified_features

if __name__ == "__main__":
    modified_adj, modified_features = run_attack()
    print("Modified adjacency matrix:", modified_adj)
    print("Modified features matrix:", modified_features)
    # You can add further analysis or save the results as needed