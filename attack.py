import torch
import os
import argparse

from deeprobust.graph.targeted_attack import Nettack
from deeprobust.graph.defense import GCN

from datasets.cora import CoraDataset
from datasets.amazon_rating import AmazonRating

# ---- Device Setup ----
print("Torch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("CUDA device count:", torch.cuda.device_count())
print("CUDA version (from PyTorch):", torch.version.cuda)
print("Device name:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

DEVICE = 'cuda:1' if torch.cuda.device_count() > 1 else 'cuda:0'
os.makedirs("attack_results", exist_ok=True)

# ---- Configuration ----
CONFIG = {
    "dataset": "Cora",  # Options: "Cora", "Amazon-Rating"
    "perturbation_budgets": [1, 2, 3, 4, 5]
}


def get_dataset(name, seed):
    name = name.lower()
    if name == "cora":
        return CoraDataset(seed)
   
    else:
        raise ValueError(f"Unknown dataset: {name}")


def setup_dataset(seed):
    ds = get_dataset(CONFIG["dataset"], seed)
    data = ds.dataset[0]
    ds.dataset.print_summary()
    print("Num node features:", ds.num_node_features)
    return ds, data


def setup_surrogate(ds):
    model = GCN(
        nfeat=ds.num_node_features,
        nclass=ds.num_classes,
        nhid=16,
        dropout=0,
        with_relu=False,
        with_bias=False,
        device=DEVICE
    ).to(DEVICE)
    return model


def perform_attack(ds, surrogate, data):
    results_by_budget = []

    target_nodes = ds.fit_surrogate(surrogate)
    print(f"Attacking {len(target_nodes)} candidate nodes...")

    for n_pert in CONFIG["perturbation_budgets"]:
        print(f"==== Perturbation Budget: {n_pert} ====")
        pert_results = []

        for node_id in target_nodes:
            print(f" -> Attacking node {node_id} with {n_pert} perturbations")

            attacker = Nettack(
                surrogate,
                nnodes=data.num_nodes,
                attack_structure=True,
                attack_features=True,
                device=DEVICE
            ).to(DEVICE)

            modified_adj, modified_features = ds.attack(attacker, node_id, n_pert)

            pert_results.append({
                "target_node": node_id,
                "modified_adj": modified_adj,
                "modified_features": modified_features
            })

        results_by_budget.append({
            "n_perturbations": n_pert,
            "perturbations": pert_results
        })

    return results_by_budget


def run(seed):
    ds, data = setup_dataset(seed)
    surrogate = setup_surrogate(ds)
    attack_results = perform_attack(ds, surrogate, data)

    save_path = f'attack_results/{CONFIG["dataset"]}-{seed}_netattack.pt'
    torch.save(attack_results, save_path)
    print(f"\nSaved attack results to {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    run(args.seed)
