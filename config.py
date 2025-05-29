# configs/config.py

CONFIG = {
    "cora": {
        "hidden_channels": 8,
        "num_heads": 8,
        "dropout": 0.6,
        "lr": 0.005,
        "weight_decay": 5e-4,
        "epochs": 200,
        "split_idx": 0,
        "device": "cuda"
    },
    "chameleon": {
        "hidden_channels": 16,     
        "num_heads": 8,
        "dropout": 0.5,
        "lr": 0.01,               
        "weight_decay": 5e-4,
        "epochs": 300,
        "split_idx": 0,
        "device": "cuda"
    }
}
