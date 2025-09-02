import os
import math
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

FIGSIZE = (4.8, 3.2)
my_custom_style = {
    'lines.linewidth': 0.7,
    'axes.prop_cycle': plt.cycler(color=plt.style.library['tableau-colorblind10']['axes.prop_cycle'].by_key()['color']),
    'legend.frameon':False,
}

paper_style = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 9,
    "font.family": "serif",
    "axes.labelsize": 9,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "axes.grid": True,
    "grid.linestyle": ":",
    "grid.alpha": 0.4,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}



combined_style = my_custom_style
combined_style.update(paper_style)

plt.rcParams.update(combined_style)

def _clean_homophily_lookup(df: pd.DataFrame) -> pd.DataFrame:
    clean_df = df[df["attack_model"].str.lower() == "clean"].copy()
    if "homophily" not in clean_df.columns:
        raise ValueError("CSV must contain 'homophily' for clean runs.")
    return (
        clean_df.groupby("K", as_index=False)[["homophily"]]
        .mean()
        .rename(columns={"homophily": "h_clean"})
    )

def plot_line_misclf_vs_homophily(df: pd.DataFrame, out_dir="results_figs", out_name="fig1_line"):
    """Figure 1: Line plot of mean misclassification vs clean homophily.
    """
    os.makedirs(out_dir, exist_ok=True)
    clean_h_df = _clean_homophily_lookup(df)

    # Normalize text columns for robust filtering
    df = df.copy()
    if "attack_model" in df.columns:
        df["attack_model"] = df["attack_model"].astype(str).str.lower()
    if "model" in df.columns:
        df["model"] = df["model"].astype(str).str.lower()

    # 1) Clean curve
    clean_mean = (
        df[df["attack_model"] == "clean"]
        .groupby("K", as_index=False)["misclassification_rate"].mean()
        .merge(clean_h_df, on="K", how="left")
        .sort_values("h_clean")
    )

    # 2) Attacked curve evaluated with GCN
    meta_gcn_mean = (
        df[(df["attack_model"] == "metattack") & (df["model"] == "gcn")]
        .groupby("K", as_index=False)["misclassification_rate"].mean()
        .merge(clean_h_df, on="K", how="left")
        .sort_values("h_clean")
    )

    # 3) Attacked curve evaluated with GAT
    meta_gat_mean = (
        df[(df["attack_model"] == "metattack") & (df["model"] == "gat")]
        .groupby("K", as_index=False)["misclassification_rate"].mean()
        .merge(clean_h_df, on="K", how="left")
        .sort_values("h_clean")
    )

    meta_h2gcn_mean = (
        df[(df["attack_model"] == "metattack") & (df["model"] == "h2gcn")]
        .groupby("K", as_index=False)["misclassification_rate"].mean()
        .merge(clean_h_df, on="K", how="left")
        .sort_values("h_clean")
    )

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.plot(clean_mean["h_clean"], clean_mean["misclassification_rate"],
            marker="o", linestyle="--", label="Clean")

    ax.plot(meta_gcn_mean["h_clean"], meta_gcn_mean["misclassification_rate"],
            marker="s", linestyle="-", label="Metattack (GCN)")

    ax.plot(meta_gat_mean["h_clean"], meta_gat_mean["misclassification_rate"],
            marker="^", linestyle="-", label="Metattack (GAT)")

    ax.plot(meta_gat_mean["h_clean"], meta_h2gcn_mean["misclassification_rate"],
            marker="x", linestyle="-", label="Metattack (H2GCN)")
    

    ax.set_xlabel("Homophily $h$ (clean graph)")
    ax.set_ylabel("Misclassification rate")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{out_name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")

def plot_bar_homophily_before_after(
    df: pd.DataFrame,
    out_dir: str = "results_figs",
    out_name: str = "fig_homophily_bar_added_edges"
):
    """
    Bar chart of homophily before (clean) vs after (Metattack).
    X-axis: cumulative number of added edges (cumsum of K).
    Y-axis: homophily.
    Bars are grouped (before vs after) per cumulative step.
    """
    os.makedirs(out_dir, exist_ok=True)

    # Clean
    clean = (
        df[df["attack_model"].str.lower() == "clean"]
        .groupby("K", as_index=False)["homophily"]
        .mean()
        .rename(columns={"homophily": "h_before"})
    )

    # Metattack
    attacked = (
        df[df["attack_model"].str.lower() == "metattack"]
        .groupby("K", as_index=False)["homophily"]
        .mean()
        .rename(columns={"homophily": "h_after"})
    )

    merged = clean.merge(attacked, on="K", how="inner").sort_values("K")
    if merged.empty:
        raise ValueError("No overlapping K values between clean and metattack to plot homophily.")

    merged["num_added_edges"] = merged["K"].cumsum()

    x = np.arange(len(merged))
    width = 0.35

    fig, ax = plt.subplots(figsize=FIGSIZE)

    ax.bar(x - width/2, merged["h_before"], width, label="Before (clean)")
    ax.bar(x + width/2, merged["h_after"],  width, label="After (Metattack)")

    ax.set_xticks(x)
    ax.set_xticklabels([str(int(val)) for val in merged["num_added_edges"]])
    ax.set_xlabel("Number of added edges")
    ax.set_ylabel("Homophily $h$")
    ax.set_ylim(0, 1)
    ax.legend(frameon=False)
    fig.tight_layout()

    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"{out_name}.{ext}")
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved: {path}")

def prepare_budget_h2gcn():
    budget_file = 'budget_cora.csv'
    df1 = pd.read_csv(budget_file)

    # we have 5% and clean classification here
    clean_df = pd.read_csv('result_h2gcn.csv')

    clean_df = clean_df[clean_df['seed']==33]

    clean_df['budget'] = 0.05

    print(clean_df.shape)
    print(df1.shape)

    df = pd.concat([df1, clean_df], ignore_index = True)

    df.to_csv('h2gcn_budget.csv')

def plot_h2gcn():    
    file = 'h2gcn_budget.csv'
    df = pd.read_csv(file)
    
    x = df[df['attack_model'] == 'clean'].sort_values(by='K')['homophily']
    fig, ax = plt.subplots(figsize=FIGSIZE)
    
    ax.set_ylim(0, 1)
    ax.set_ylabel("Misclassification rate")
    
    for budget in sorted(df['budget'].unique()):
        y = df[df['budget'] == budget].sort_values(by='K')['misclassification_rate']
        # No need to specify linewidth anymore
        ax.plot(x, y, label=f'{budget}')
    
    ax.legend(title='Budget')
    plt.savefig('h2gcn_budget.pdf', bbox_inches='tight')


def plot_citeseer():
    file = 'Citeseer.csv'
    df = pd.read_csv(file)
    model = 'h2gcn'
    x = df[(df['attack_model'] == 'clean') & (df['model'] == model)].sort_values(by='K')['homophily']
    yclean = df[(df['attack_model'] == 'clean') & (df['model'] == model)].sort_values(by='K')['misclassification_rate']
    ymettattack = df[(df['attack_model'] == 'metattack') & (df['model'] == model)].sort_values(by='K')['misclassification_rate']
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Misclassification rate")
    ax.plot(x,yclean, label='Clean graph')
    ax.plot(x,ymettattack, label='Metattacked graph')
    plt.savefig('results_figs/h2gch_citeseer.pdf', bbox_inches='tight')

    

    
    

    
    
    
    

plot_citeseer()
