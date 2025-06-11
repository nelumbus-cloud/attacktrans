# datasets/base.py
from deeprobust.graph.data import Dataset, Dpr2Pyg, Pyg2Dpr

class BaseDataset:
    def __init__(self, root, name):
        self.root = root
        self.name = name

        self.hidden_channels = 8
        self.num_heads = 8
        self.dropout = 0.6
        self.lr = 0.005
        self.weight_decay = 5e-4
        self.epochs = 200
        self.data = None

    def load(self):
        """
        Should return (data, num_node_features, num_classes)
        """
        raise NotImplementedError

    def process_mask(self, mask, split_idx=0):
        """
        Handles mask format. Child class can override if needed.
        """
        return mask
    
    def get_dpr(self):
        self.dpr_data = Pyg2Dpr(self.data)
        return self.dpr_data
    
    
