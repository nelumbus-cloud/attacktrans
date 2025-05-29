import json
import os
import plotly.graph_objects as go
import plotly.express as px

# 1. Load results
with open('gat_results.json', 'r') as f:
    results = json.load(f)

# 2. Make output directory
output_dir = "plots"
os.makedirs(output_dir, exist_ok=True)

# 3. Bar Chart: Best Validation/Test Accuracy Per Dataset
datasets = []
best_val_acc = []
test_acc_at_best_val = []

for name, result in results.items():
    datasets.append(name.capitalize())
    best_val_acc.append(result["best_val_acc"])
    test_acc_at_best_val.append(result["test_acc_at_best_val"])

fig_bar = go.Figure(data=[
    go.Bar(name='Best Val Acc', x=datasets, y=best_val_acc, marker_color='royalblue'),
    go.Bar(name='Test Acc at Best Val', x=datasets, y=test_acc_at_best_val, marker_color='seagreen')
])
fig_bar.update_layout(
    barmode='group',
    title='GAT: Best Validation/Test Accuracy per Dataset',
    yaxis_title='Accuracy',
    xaxis_title='Dataset',
    font=dict(size=18),
    legend=dict(font=dict(size=16))
)
fig_bar.write_image(os.path.join(output_dir, "gat_best_acc_bar.png"), scale=3)
fig_bar.write_html(os.path.join(output_dir, "gat_best_acc_bar.html"))

# 4. Line Plot: Accuracy Curves per Dataset
for name, result in results.items():
    logs = result["all_logs"]
    epochs = [entry["epoch"] for entry in logs]
    train_acc = [entry["train_acc"] for entry in logs]
    val_acc = [entry["val_acc"] for entry in logs]
    test_acc = [entry["test_acc"] for entry in logs]

    fig_line = go.Figure()
    fig_line.add_trace(go.Scatter(x=epochs, y=train_acc, mode='lines', name='Train Acc', line=dict(width=3)))
    fig_line.add_trace(go.Scatter(x=epochs, y=val_acc, mode='lines', name='Val Acc', line=dict(width=3, dash='dot')))
    fig_line.add_trace(go.Scatter(x=epochs, y=test_acc, mode='lines', name='Test Acc', line=dict(width=3, dash='dash')))
    fig_line.update_layout(
        title=f"GAT Accuracy Curves - {name.capitalize()}",
        xaxis_title="Epoch",
        yaxis_title="Accuracy",
        font=dict(size=18),
        legend=dict(font=dict(size=16)),
        width=800, height=500
    )
    fig_line.write_image(os.path.join(output_dir, f"gat_acc_curve_{name}.png"), scale=3)
    fig_line.write_html(os.path.join(output_dir, f"gat_acc_curve_{name}.html"))

print(f"Plots saved to: {output_dir}/")
