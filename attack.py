from deeprobust.graph.targeted_attack import Nettack
from datasets.cora import CoraDataset
from datasets.chameleon import ChameleonDataset
from deeprobust.graph.defense import GCN
import torch

import torch
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA version (from PyTorch):", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0))

DEVICE='cuda:1'

def run_attack():
    
    #cora = CoraDataset() 
    cora = ChameleonDataset()

    cora.dataset.print_summary()
    
    target_node = 0

    print(cora.num_node_features)
    
    
    surrogate = GCN(nfeat=cora.num_node_features, nclass=cora.num_classes,
                    nhid=16, dropout=0, with_relu=False, with_bias=False, device=DEVICE).to(DEVICE)

    model = Nettack(surrogate, nnodes=cora.num_nodes, attack_structure=True, attack_features=True, device=DEVICE).to(DEVICE)
    cora.attack("NETATTACK", model, target_node, surrogate)

    print(cora.modified_adj, cora.modified_features)

run_attack()