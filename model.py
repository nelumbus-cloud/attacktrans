# model.py

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv
from torch_geometric.utils import from_networkx, from_scipy_sparse_matrix
import networkx as nx
import numpy as np
import scipy.sparse as sp
import torch.nn as nn
import torch_sparse
from torch import FloatTensor

# the following h2gcn module is implementation of https://github.com/GitEventhandler/H2GCN-PyTorch.

class _H2GCNCore(nn.Module):
    """
    Lightweight H2GCN-style encoder/classifier.

    Builds k-hop representations using (A\I) and (A^2 \ A \ I) channels:
      - a1 = indicator(A - I), normalized
      - a2 = indicator(A^2 - A - I), normalized
    Propagates hidden states via spmm on a1 and a2 for k steps and concatenates:
      r0 = act(X W_embed)
      for t in 1..k:
          r1 = a1 @ r_{t-1}
          r2 = a2 @ r_{t-1}
          rt = act([r1 || r2])
      H = [r0 || r1 || ... || rk]
      logits = softmax(H W_classify)
    """
    def __init__(self, feat_dim: int, hidden_dim: int, class_dim: int,
                 k: int = 2, dropout: float = 0.5, use_relu: bool = True):
        super().__init__()
        self.k = k
        self.dropout = dropout
        self.act = F.relu if use_relu else (lambda x: x)

        # r0 has size hidden_dim; each subsequent rt doubles (concat r1||r2) => hidden_dim * (2**t)
        total_blocks = (2 ** (k + 1) - 1)  # 1 + 2 + 4 + ... + 2^k
        self.w_embed = nn.Parameter(torch.empty(feat_dim, hidden_dim))
        self.w_classify = nn.Parameter(torch.empty(hidden_dim * total_blocks, class_dim))
        self.reset_parameters()

        # these are prepared per-adj (set during fit)
        self._a1 = None
        self._a2 = None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.w_embed)
        nn.init.xavier_uniform_(self.w_classify)

    @staticmethod
    def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
        csp = sp_tensor.coalesce()
        vals = torch.where(csp.values() > 0, torch.ones_like(csp.values()), torch.zeros_like(csp.values()))
        return torch.sparse_coo_tensor(csp.indices(), vals, csp.size(), dtype=torch.float, device=csp.device)

    @staticmethod
    def _eye_like(n: int, device) -> torch.sparse.Tensor:
        idx = torch.arange(n, device=device)
        i = torch.stack([idx, idx], dim=0)
        v = torch.ones(n, dtype=torch.float, device=device)
        return torch.sparse_coo_tensor(i, v, (n, n)).coalesce()

    @staticmethod
    def _to_sparse_coo(adj, device):
        # Accept scipy.sparse matrix **or** array types
        if sp.issparse(adj):  # works for spmatrix and sparray
            adj = adj.tocoo()
            i = torch.tensor(np.vstack([adj.row, adj.col]), dtype=torch.long, device=device)
            v = torch.tensor(adj.data, dtype=torch.float32, device=device)
            return torch.sparse_coo_tensor(i, v, (adj.shape[0], adj.shape[1])).coalesce()
    
        if isinstance(adj, np.ndarray):
            return _H2GCNCore._to_sparse_coo(sp.coo_matrix(adj), device)
    
        if isinstance(adj, torch.Tensor):
            # if dense torch tensor, convert via NumPy
            if not adj.is_sparse:
                adj = adj.detach().cpu().numpy()
                return _H2GCNCore._to_sparse_coo(sp.coo_matrix(adj), device)
            return adj.coalesce().to(device)
    
        raise TypeError(f"Unsupported adjacency type: {type(adj)}")


    @staticmethod
    def _spspmm(a: torch.sparse.Tensor, b: torch.sparse.Tensor) -> torch.sparse.Tensor:
        a = a.coalesce(); b = b.coalesce()
        assert a.size(1) == b.size(0), f"spspmm shape mismatch {a.size()} x {b.size()}"
        idx, val = torch_sparse.spspmm(a.indices(), a.values(), b.indices(), b.values(),
                                       a.size(0), a.size(1), b.size(1))
        return torch.sparse_coo_tensor(idx, val, (a.size(0), b.size(1)), device=a.device).coalesce()

    @staticmethod
    def _sym_norm(adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
        # D^{-1/2} A D^{-1/2}
        adj = adj.coalesce()
        n = adj.size(0)
        deg = torch.sparse.sum(adj, dim=1).to_dense()
        d = torch.pow(deg.clamp(min=1e-12), -0.5)
        i = torch.arange(n, device=adj.device)
        d_mat = torch.sparse_coo_tensor(torch.stack([i, i]), d, (n, n), device=adj.device).coalesce()
        return _H2GCNCore._spspmm(_H2GCNCore._spspmm(d_mat, adj), d_mat)

    def _prepare_a1_a2(self, adj_sp: torch.sparse.Tensor):
        # a1 = indicator(A - I)
        n = adj_sp.size(0)
        eye = self._eye_like(n, adj_sp.device)
        a_minus_i = adj_sp - eye
        a1 = self._indicator(a_minus_i)

        # a2 = indicator(A^2 - A - I)
        a2_raw = self._spspmm(adj_sp, adj_sp) - adj_sp - eye
        a2 = self._indicator(a2_raw)

        # sym-normalize
        self._a1 = self._sym_norm(a1)
        self._a2 = self._sym_norm(a2)

    def forward(self, x: torch.Tensor, adj_sp: torch.sparse.Tensor) -> torch.Tensor:
        if self._a1 is None or self._a2 is None:
            self._prepare_a1_a2(adj_sp)

        # r0
        r0 = self.act(torch.matmul(x, self.w_embed))
        reps = [r0]

        # k steps
        r_last = r0
        for _ in range(self.k):
            r1 = torch.sparse.mm(self._a1, r_last)
            r2 = torch.sparse.mm(self._a2, r_last)
            r = torch.cat([r1, r2], dim=1)
            r = self.act(r)
            reps.append(r)
            r_last = r

        H = torch.cat(reps, dim=1)
        H = F.dropout(H, p=self.dropout, training=self.training)
        logits = torch.matmul(H, self.w_classify)
        return logits


class H2GCN:
    """
    DeepRobust-style wrapper:
      - initialize()
      - fit(features, adj, labels, idx_train, idx_val, patience=30, max_epochs=1000)
      - test(idx_test) -> accuracy
    Accepts features: (np.ndarray | scipy.sparse | torch.Tensor)
            adj:      (scipy.sparse preferred | np.ndarray | torch Tensor dense/sparse)
            labels:   (np.ndarray | torch.Tensor of ints)
    """
    def __init__(self, nfeat, nclass, hidden=64, k=2, dropout=0.5,
                 use_relu=True, lr=0.01, weight_decay=5e-4, device=torch.device("cpu")):
        self.device = device
        self.model = _H2GCNCore(
            feat_dim=nfeat, hidden_dim=hidden, class_dim=nclass,
            k=k, dropout=dropout, use_relu=use_relu
        ).to(self.device)
        self.lr = lr
        self.weight_decay = weight_decay

        # cache tensors for predict/test
        self._cache = {}

    # ------- utilities
    @staticmethod
    def _to_dense_x(x, device):
        if sp.issparse(x):
            x = x.todense()
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float().to(device)
        if isinstance(x, torch.Tensor):
            return x.float().to(device)
        raise TypeError(f"Unsupported features type: {type(x)}")

    @staticmethod
    def _to_labels(y, device):
        if isinstance(y, np.ndarray):
            return torch.from_numpy(y).long().to(device)
        if isinstance(y, torch.Tensor):
            return y.long().to(device)
        raise TypeError(f"Unsupported labels type: {type(y)}")

    def _to_sparse_adj(self, adj):
        return _H2GCNCore._to_sparse_coo(adj, self.device)

    @staticmethod
    def _make_mask(n, idx):
        mask = torch.zeros(n, dtype=torch.bool)
        mask[idx] = True
        return mask

    def initialize(self):
        self.model.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val, patience=30, max_epochs=1000):
        x = self._to_dense_x(features, self.device)
        y = self._to_labels(labels, self.device)
        adj_sp = self._to_sparse_adj(adj)  # builds COO on device

        # ensure a1/a2 recomputed for this adjacency
        self.model._a1 = None
        self.model._a2 = None

        n = x.size(0)
        train_mask = self._make_mask(n, idx_train).to(self.device)
        val_mask = self._make_mask(n, idx_val).to(self.device)

        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        best_val = -1.0
        best_state = None
        bad = 0

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            opt.zero_grad()
            logits = self.model(x, adj_sp)
            loss = F.cross_entropy(logits[train_mask], y[train_mask])
            loss.backward()
            opt.step()

            # validation
            self.model.eval()
            with torch.no_grad():
                vlogits = self.model(x, adj_sp)
                vpred = vlogits[val_mask].argmax(dim=1)
                vacc = float((vpred == y[val_mask]).sum().item()) / int(val_mask.sum())
            if vacc > best_val:
                best_val = vacc
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                bad = 0
            else:
                bad += 1
                if bad >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # cache tensors for predict/test
        self._cache = {"x": x, "y": y, "adj_sp": adj_sp}

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        x = self._cache["x"]; adj_sp = self._cache["adj_sp"]
        logits = self.model(x, adj_sp).detach().cpu()
        return logits

    @torch.no_grad()
    def test(self, idx_test):
        self.model.eval()
        x = self._cache["x"]; y = self._cache["y"]; adj_sp = self._cache["adj_sp"]
        logits = self.model(x, adj_sp)
        pred = logits.argmax(dim=1)
        mask = self._make_mask(x.size(0), idx_test).to(self.device)
        correct = (pred[mask] == y[mask]).sum().item()
        total = int(mask.sum())
        return correct / total if total > 0 else 0.0


class _GATBackbone(torch.nn.Module):
    """
    Two-layer GAT following the common PyG example.
    """
    def __init__(self, in_channels, hidden_channels, out_channels, heads=8, dropout=0.6):
        super().__init__()
        self.gat1 = GATConv(in_channels, hidden_channels, heads=heads, dropout=dropout)
        self.gat2 = GATConv(hidden_channels * heads, out_channels, heads=1, concat=False, dropout=dropout)
        self.dropout = dropout

    def reset_parameters(self):
        self.gat1.reset_parameters()
        self.gat2.reset_parameters()

    def forward(self, x, edge_index):
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.gat2(x, edge_index)
        return x

class GAT:
    def __init__(self, nfeat, nclass, nhid=8, heads=8, dropout=0.6,
                 lr=0.01, weight_decay=5e-4, device=torch.device("cpu")):
        self.device = device
        self.model = _GATBackbone(
            in_channels=nfeat,
            hidden_channels=nhid,
            out_channels=nclass,
            heads=heads,
            dropout=dropout,
        ).to(self.device)

        self.lr = lr
        self.weight_decay = weight_decay
        self.dropout = dropout

        self._cached = {}  # will hold tensors/masks for predict/test

    @staticmethod
    def _to_tensors(features, adj, labels, device):
        # features: np.ndarray or scipy.spmatrix
        if sp.issparse(features):
            features = features.todense()
        if isinstance(features, np.ndarray):
            x = torch.from_numpy(features).float().to(device)
        elif isinstance(features, torch.Tensor):
            x = features.float().to(device)
        else:
            x = torch.tensor(features, dtype=torch.float32, device=device)

        # labels: np.ndarray or torch.Tensor
        if isinstance(labels, np.ndarray):
            y = torch.from_numpy(labels).long().to(device)
        elif isinstance(labels, torch.Tensor):
            y = labels.long().to(device)
        else:
            y = torch.tensor(labels, dtype=torch.long, device=device)

        # adj: scipy sparse preferred
        if sp.issparse(adj):
            coo = adj.tocoo()
        elif isinstance(adj, np.ndarray):
            coo = sp.coo_matrix(adj)
        elif isinstance(adj, torch.Tensor):
            # assume dense; move to CPU first
            coo = sp.coo_matrix(adj.detach().cpu().numpy())
        else:
            raise TypeError(f"Unsupported adjacency type: {type(adj)}")

        edge_index, edge_weight = from_scipy_sparse_matrix(coo)
        edge_index = edge_index.to(device)

        return x, y, edge_index

    @staticmethod
    def _make_mask(n_nodes, idx):
        mask = torch.zeros(n_nodes, dtype=torch.bool)
        mask[idx] = True
        return mask

    def initialize(self):
        self.model.reset_parameters()

    def fit(self, features, adj, labels, idx_train, idx_val, patience=30, max_epochs=200):
        x, y, edge_index = self._to_tensors(features, adj, labels, self.device)
        n_nodes = x.size(0)
        train_mask = self._make_mask(n_nodes, idx_train).to(self.device)
        val_mask = self._make_mask(n_nodes, idx_val).to(self.device)

        self.model.train()
        optimizer = torch.optim.Adam(self.model.parameters(),
                                     lr=self.lr, weight_decay=self.weight_decay)

        best_loss = float("inf")
        best_state = None
        wait = 0

        for epoch in range(1, max_epochs + 1):
            self.model.train()
            optimizer.zero_grad()
            out = self.model(x, edge_index)
            loss = F.cross_entropy(out[train_mask], y[train_mask])
            loss.backward()
            optimizer.step()

            # validation
            self.model.eval()
            with torch.no_grad():
                logits = self.model(x, edge_index)
                val_loss = F.cross_entropy(logits[val_mask], y[val_mask]).item()

            if val_loss < best_loss:
                best_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    break

        if best_state is not None:
            self.model.load_state_dict({k: v.to(self.device) for k, v in best_state.items()})

        # cache tensors for predict/test without recomputing conversions
        self._cached = {"x": x, "y": y, "edge_index": edge_index}

    @torch.no_grad()
    def predict(self):
        self.model.eval()
        x = self._cached["x"]; edge_index = self._cached["edge_index"]
        logits = self.model(x, edge_index).detach().cpu()
        return logits

    @torch.no_grad()
    def test(self, idx_test):
        self.model.eval()
        x = self._cached["x"]; y = self._cached["y"]; edge_index = self._cached["edge_index"]
        logits = self.model(x, edge_index)
        pred = logits.argmax(dim=1)
        mask = self._make_mask(x.size(0), idx_test).to(self.device)
        correct = (pred[mask] == y[mask]).sum().item()
        total = int(mask.sum())
        return correct / total if total > 0 else 0.0

if __name__ == "__main__":
    import numpy as np
    import torch
    import networkx as nx
    import scipy.sparse as sp

    torch.manual_seed(0)
    np.random.seed(0)

    # tiny synthetic setup
    num_nodes   = 40
    in_features = 16
    num_classes = 4

    # features & labels
    X = np.random.randn(num_nodes, in_features).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_nodes,)).astype(np.int64)

    # simple graph (cycle) -> SciPy COO adjacency
    G = nx.cycle_graph(num_nodes)
    A = nx.to_scipy_sparse_array(G, format="coo", dtype=np.float32)

    # train/val/test split (indices)
    idx_train = np.arange(0, 15)
    idx_val   = np.arange(15, 25)
    idx_test  = np.arange(25, num_nodes)

    device = torch.device("cpu")

    # ---- GAT wrapper ----
    gat = GAT(nfeat=in_features, nclass=num_classes, nhid=8, heads=2, dropout=0.6, device=device)
    gat.initialize()
    gat.fit(X, A, y, idx_train, idx_val, patience=10, max_epochs=200)
    acc_gat = gat.test(idx_test)
    logits_gat = gat.predict()
    print(f"GAT: test acc = {acc_gat:.3f}, logits shape = {tuple(logits_gat.shape)}")

    # ---- H2GCN wrapper ----
    h2 = H2GCN(nfeat=in_features, nclass=num_classes, hidden=16, k=2, dropout=0.5,
               use_relu=True, lr=0.01, weight_decay=5e-4, device=device)
    h2.initialize()
    h2.fit(X, A, y, idx_train, idx_val, patience=10, max_epochs=200)
    acc_h2 = h2.test(idx_test)
    logits_h2 = h2.predict()
    print(f"H2GCN: test acc = {acc_h2:.3f}, logits shape = {tuple(logits_h2.shape)}")

