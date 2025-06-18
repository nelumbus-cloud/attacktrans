import torch
import os
import argparse
import json

from deeprobust.graph.defense import GCN
from deeprobust.graph.data import Pyg2Dpr

from datasets.cora import CoraDataset
from datasets.amazon_rating import AmazonRating

DEVICE = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'

CONFIG = {
    "dataset_name": "Cora",
    "model_name": "gcn"
}


def get_dataset(name, seed):
    name = name.lower()
    if name == "cora":
        return CoraDataset(seed)
    elif name == "amazon-rating":
        return AmazonRating(seed)
    else:
        raise ValueError(f"Unknown dataset: {name}")


def load_clean_dataset(seed):
    ds = get_dataset(CONFIG["dataset_name"], seed)
    pyg_data = ds.dataset
    dpr_data = Pyg2Dpr(pyg_data)
    return ds, dpr_data


def get_model(name, nfeat, nclass):
    name = name.lower()
    if name == "gcn":
        return GCN(nfeat=nfeat, nclass=nclass, nhid=16, device=DEVICE).to(DEVICE)
    elif name == "gsage":
        raise NotImplementedError("GraphSAGE model not implemented.")
    else:
        raise ValueError(f"Unknown model: {name}")




def evaluate_misclassification(model, base_data, pt_file):
    if not os.path.exists(pt_file):
        raise FileNotFoundError(f"Attack result file not found: {pt_file}")

    attack_data = torch.load(pt_file)
    results = []

    for entry in attack_data:
        n_pert = entry["n_perturbations"]
        print(f"\n== Perturbation Budget: {n_pert} ==")

        total, misclassified = 0, 0

        for perturb in entry["perturbations"]:
            target = perturb["target_node"]
            adj = perturb["modified_adj"]
            feat = perturb["modified_features"]


            model = get_model(CONFIG["model_name"], feat.shape[1], base_data.labels.max().item() + 1)
            
            model.fit(
                feat, adj, base_data.labels,
                base_data.idx_train, base_data.idx_val,
                initialize=True,
                patience=30
            )

            logits = model.predict()
            
            pred = logits[target].argmax(dim=0).item()

            true = base_data.labels[target].item()

            if pred != true:
                print(f"  Node {target} misclassified  pred: {pred}, true: {true}")
                misclassified += 1
            else:
                print(f"  Node {target} correct")

            total += 1

        rate = misclassified / total
        results.append({
            "budget": n_pert,
            "misclassification_rate": round(rate, 4)
        })

    # Save JSON
    os.makedirs("attack_results", exist_ok=True)
    json_file = f"attack_results/{CONFIG['dataset_name']}-{seed}_misclassification.json"
    with open(json_file, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\nSaved misclassification report to {json_file}")


def run(seed):
    dataset, data = load_clean_dataset(seed)

    print("Training on clean graph...")
    model = get_model(CONFIG["model_name"], dataset.num_node_features, dataset.num_classes)

    
    model.fit(
        data.features, data.adj, data.labels,
        data.idx_train, data.idx_val,
        initialize=True,
        patience=30
        )

    acc = model.test(data.idx_test)
    
    print(f"\nClean Graph Test Accuracy: {acc:.2%}")

    pt_file = f"attack_results/{CONFIG['dataset_name']}-{seed}_netattack.pt"
    print(f"\n--- Evaluating on Attacked Graphs from: {pt_file} ---")
    evaluate_misclassification(model, data, pt_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    seed = args.seed
    run(seed)
