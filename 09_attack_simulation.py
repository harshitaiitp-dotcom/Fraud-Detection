import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
print("--- ADVERSARIAL ATTACK & DEFENSE SIMULATION FOR GNN ---")
features_df = pd.read_csv("data/elliptic_txs_features.csv", header=None)
classes_df = pd.read_csv("data/elliptic_txs_classes.csv")
edges_df = pd.read_csv("data/elliptic_txs_edgelist.csv")
classes_df["numeric_class"] = classes_df["class"].map({"1": 0, "2": 1})
labeled_df = classes_df[classes_df["class"] != "unknown"].copy()
merged_data = pd.merge(labeled_df, features_df, left_on="txId", right_on=0)
unique_txs = merged_data["txId"].values
tx_to_idx = {tx: idx for idx, tx in enumerate(unique_txs)}
filtered_edges = edges_df[
    edges_df["txId1"].isin(tx_to_idx) & edges_df["txId2"].isin(tx_to_idx)
]
src = filtered_edges["txId1"].map(tx_to_idx).values
dst = filtered_edges["txId2"].map(tx_to_idx).values
edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
x_orig = torch.tensor(merged_data.iloc[:, 3:97].values, dtype=torch.float)
y = torch.tensor(merged_data["numeric_class"].values, dtype=torch.long)
indices = np.arange(len(y))
train_idx, test_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=y.numpy()
)
class FraudGCN(nn.Module):
  def __init__(self, in_channels, hidden_channels, out_channels):
    super(FraudGCN, self).__init__()
    self.conv1 = GCNConv(in_channels, hidden_channels)
    self.conv2 = GCNConv(hidden_channels, out_channels)
  def forward(self, x, edge_index):
    x = self.conv1(x, edge_index)
    x = F.relu(x)
    x = F.dropout(x, p=0.5, training=self.training)
    x = self.conv2(x, edge_index)
    return x
def evaluate(model, x_tensor, edge_tensor):
  model.eval()
  with torch.no_grad():
    pred = model(x_tensor, edge_tensor).argmax(dim=1)
    acc = (pred[test_idx] == y[test_idx]).sum().item() / len(test_idx)
  return acc
model = FraudGCN(in_channels=x_orig.shape[1], hidden_channels=32, out_channels=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()
model.train()
for epoch in range(30):
  optimizer.zero_grad()
  out = model(x_orig, edge_index)
  loss = criterion(out[train_idx], y[train_idx])
  loss.backward()
  optimizer.step()
clean_acc = evaluate(model, x_orig, edge_index)
print(f"\n1. Clean Model Test Accuracy: {clean_acc * 100:.2f}%")
x_attacked = x_orig.clone()
test_illicit_mask = (y == 1) & (
    torch.tensor([i in test_idx for i in range(len(y))], dtype=torch.bool)
)
noise = torch.randn_like(x_attacked[test_illicit_mask]) * 2.5
x_attacked[test_illicit_mask] += noise
attacked_acc = evaluate(model, x_attacked, edge_index)
print(f"2. Model Accuracy Under Adversarial Attack: {attacked_acc * 100:.2f}%")
print(
    f"   --> Performance drop due to evasion: {(clean_acc - attacked_acc) * 100:.2f}%"
)
print("\n3. Training Robust Model with Adversarial Training Defense...")
robust_model = FraudGCN(
    in_channels=x_orig.shape[1], hidden_channels=32, out_channels=2
)
robust_opt = torch.optim.Adam(robust_model.parameters(), lr=0.01)
robust_model.train()
for epoch in range(40):
  x_train_adv = x_orig.clone()
  train_illicit_mask = (y == 1) & (
      torch.tensor([i in train_idx for i in range(len(y))], dtype=torch.bool)
  )
  x_train_adv[train_illicit_mask] += (
      torch.randn_like(x_train_adv[train_illicit_mask]) * 1.5
  )
  robust_opt.zero_grad()
  out = robust_model(x_train_adv, edge_index)
  loss = criterion(out[train_idx], y[train_idx])
  loss.backward()
  robust_opt.step()
defended_acc = evaluate(robust_model, x_attacked, edge_index)
print(
    f"4. Robust Model Accuracy Under Adversarial Attack:"
    f" {defended_acc * 100:.2f}%"
)
print(
    f"   --> Recovered accuracy via Defense:"
    f" +{(defended_acc - attacked_acc) * 100:.2f}%"
)