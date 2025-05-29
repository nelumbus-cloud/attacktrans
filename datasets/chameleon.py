from torch_geometric.datasets import WikipediaNetwork
from torch_geometric.transforms import NormalizeFeatures
from .base import BaseDataset

class ChameleonDataset(BaseDataset):
    def __init__(self):
        super().__init__(root='data/Chameleon', name='chameleon')

    def load(self, split_idx=0):
        dataset = WikipediaNetwork(root=self.root, name=self.name, transform=NormalizeFeatures())
        data = dataset[0]
        data.train_mask = self.process_mask(data.train_mask, split_idx)
        data.val_mask = self.process_mask(data.val_mask, split_idx)
        data.test_mask = self.process_mask(data.test_mask, split_idx)
        return data, dataset.num_node_features, dataset.num_classes

    def process_mask(self, mask, split_idx=0):
        if mask.dim() == 1:
            return mask
        return mask[:, split_idx]
