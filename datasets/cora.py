from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from .base import BaseDataset

class CoraDataset(BaseDataset):
    
    def __init__(self, seed, adj=None, features=None):
        super().__init__(root='data/Cora', name='Cora')

        dataset = Planetoid(root=self.root, name=self.name, transform=self.transform(seed))
        
        self.dataset = dataset
        self.num_node_features, self.num_classes = dataset.num_node_features, dataset.num_classes
        self.num_nodes = dataset[0].x.shape[0]
        self.adj = None
        self.feature = None
    
