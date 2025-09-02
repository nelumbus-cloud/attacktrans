# attacktrans

Code for the paper **"Robustness of GNNs under Different Homophily Levels"**

---

## Setup

### 1. System Dependencies

Update your system and install the required compilers:

```bash
sudo apt update
sudo apt install g++-10 gcc-10
export CC=gcc-10
export CXX=g++-10
```

### 2. Python Dependencies

Install all required Python packages:

```bash
pip install -r requirements.txt
```

---

## Running Experiments

The main script is `attack.py`.
You can run an experiment with the following command:

```bash
python attack.py \
  --seed "$seed" \
  --dataset "$dataset" \
  --output "$filename" \
  --attack "$attack" \
  --model "$model"
```

### Arguments

* `--seed` : Random seed for reproducibility
* `--dataset` : Dataset name (e.g., `cora`, `citeseer`, `pubmed`)
* `--output` : Output filename for logs/results
* `--attack` : Attack method (e.g., `pgd`, `nettack`, etc.)
* `--model` : GNN model type (e.g., `gcn`, `graphsage`, `gin`)

---

## Project Structure

```
attacktrans/
│── attack.py         # Main entry script for running attacks
│── requirements.txt  # Python dependencies
│── README.md         # Project documentation
└── ...               # Other source files, datasets, configs
```

---

## Citation

If you use this code in your research, please cite our paper:

```
@inproceedings{baral2025,
  title={Robustness of GNNs under Different Homophily Levels},
  author={Saroj Baral, Khem Poudel, Jorge Vargas, Jaishree Ranganath},
  booktitle={Proceedings of ...},
  year={2025}
}
```

---
