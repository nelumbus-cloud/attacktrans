import json
import torch
import torch.nn.functional as F
from torch.optim import Adam

from config import CONFIG
from datasets.cora import CoraDataset
from datasets.chameleon import ChameleonDataset
from model import GAT

def select_dataset(name):
    if name == "cora":
        return CoraDataset()
    elif name == "chameleon":
        return ChameleonDataset()
    else:
        raise ValueError(f"Unknown dataset: {name}")

def train_one_epoch(model, data, optimizer, criterion):
    model.train()
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = criterion(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()
    return loss.item()

@torch.no_grad()
def test(model, data):
    model.eval()
    out = model(data.x, data.edge_index)
    pred = out.argmax(dim=1)
    def acc(mask): return int((pred[mask] == data.y[mask]).sum()) / int(mask.sum())
    return acc(data.train_mask), acc(data.val_mask), acc(data.test_mask)

def run_experiment(dataset_name, cfg):
    device = torch.device(cfg.get("device", "cpu") if torch.cuda.is_available() else "cpu") #useful when running jupyter lab
    ds = select_dataset(dataset_name)
    data, num_features, num_classes = ds.load(split_idx=cfg.get("split_idx", 0))
    data = data.to(device)

    model = GAT(
        in_channels=num_features,
        hidden_channels=cfg['hidden_channels'],
        out_channels=num_classes,
        heads=cfg['num_heads'],
        dropout=cfg['dropout']
    ).to(device)
    optimizer = Adam(model.parameters(), lr=cfg['lr'], weight_decay=cfg['weight_decay'])
    criterion = torch.nn.CrossEntropyLoss()

    best_val_acc = 0
    test_acc_at_best_val = 0

    logs = []
    for epoch in range(1, cfg['epochs'] + 1):
        loss = train_one_epoch(model, data, optimizer, criterion)
        train_acc, val_acc, test_acc = test(model, data)
        logs.append({
            "epoch": epoch,
            "loss": float(loss),
            "train_acc": float(train_acc),
            "val_acc": float(val_acc),
            "test_acc": float(test_acc)
        })
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            test_acc_at_best_val = test_acc
        if epoch % 10 == 0 or epoch == 1 or epoch == cfg['epochs']:
            print(f"[{dataset_name}] Epoch: {epoch:03d}, Loss: {loss:.4f}, "
                  f"Train: {train_acc:.4f}, Val: {val_acc:.4f}, Test: {test_acc:.4f}")

    result = {
        "dataset": dataset_name,
        "best_val_acc": float(best_val_acc),
        "test_acc_at_best_val": float(test_acc_at_best_val),
        "final_train_acc": float(logs[-1]["train_acc"]),
        "final_val_acc": float(logs[-1]["val_acc"]),
        "final_test_acc": float(logs[-1]["test_acc"]),
        "all_logs": logs  
    }
    return result

if __name__ == "__main__":
    results = {}

    for dataset_name in CONFIG.keys():
        print(f"\n=== Running GAT on {dataset_name.upper()} ===")
        cfg = CONFIG[dataset_name]
        result = run_experiment(dataset_name, cfg)
        results[dataset_name] = result

    # Save to JSON
    output_file = "gat_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nAll results saved to {output_file}")
