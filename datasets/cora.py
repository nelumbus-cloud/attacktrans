from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from .base import BaseDataset

class CoraDataset(BaseDataset):
    def __init__(self):
        super().__init__(root='data/Cora', name='Cora')

    def load(self, split_idx=0):
        dataset = Planetoid(root=self.root, name=self.name, transform=NormalizeFeatures())
        data = dataset[0]
        return data, dataset.num_node_features, dataset.num_classes

    # No process_mask override needed; Cora's masks are 1D
