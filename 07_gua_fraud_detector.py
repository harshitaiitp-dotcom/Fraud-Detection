import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
print("Loading Real and Synthetic Data for GUA Pipeline...")
features_df = pd.read_csv("data/elliptic_txs_features.csv", header=None)
classes_df = pd.read_csv("data/elliptic_txs_classes.csv")
edges_df = pd.read_csv("data/elliptic_txs_edgelist.csv")
classes_df["numeric_class"] = classes_df["class"].map({"1": 0, "2": 1})
labeled_df = classes_df[classes_df["class"] != "unknown"].copy()
merged_data = pd.merge(labeled_df, features_df, left_on="txId", right_on=0)
synthetic_df = pd.read_csv("data/synthetic_illicit_txs.csv")
if "txId" in synthetic_df.columns:
    synthetic_df = synthetic_df.drop(columns=["txId"])
min_feat = min(94, synthetic_df.shape[1])
syn_features = synthetic_df.values[:, :min_feat]
synthetic_formatted = pd.DataFrame(syn_features)
synthetic_formatted["numeric_class"] = 1  
synthetic_formatted["txId"] = [
    9000000 + i for i in range(len(synthetic_formatted))
]
print(
    f"Loaded {len(merged_data)} real labeled nodes and {len(synthetic_formatted)} synthetic fraud nodes."
)
unique_txs = merged_data["txId"].values
tx_to_idx = {tx: idx for idx, tx in enumerate(unique_txs)}
filtered_edges = edges_df[
    edges_df["txId1"].isin(tx_to_idx) & edges_df["txId2"].isin(tx_to_idx)
]
src = filtered_edges["txId1"].map(tx_to_idx).values
dst = filtered_edges["txId2"].map(tx_to_idx).values
edge_index = torch.tensor(np.array([src, dst]), dtype=torch.long)
feature_cols = [c for c in merged_data.columns if isinstance(c, int)][
    :min_feat
]
x = torch.tensor(merged_data[feature_cols].values, dtype=torch.float)
y = torch.tensor(merged_data["numeric_class"].values, dtype=torch.long)
data = Data(x=x, edge_index=edge_index, y=y)
indices = np.arange(data.num_nodes)
train_idx, test_idx = train_test_split(
    indices, test_size=0.3, random_state=42, stratify=y.numpy()
)
data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
data.train_mask[train_idx] = True
data.test_mask[test_idx] = True
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
model = FraudGCN(
    in_channels=data.num_features, hidden_channels=32, out_channels=2
)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
criterion = nn.CrossEntropyLoss()
print("\nTraining GNN Fraud Detector...")
model.train()
for epoch in range(50):
  optimizer.zero_grad()
  out = model(data.x, data.edge_index)
  loss = criterion(out[data.train_mask], data.y[data.train_mask])
  loss.backward()
  optimizer.step()
  if (epoch + 1) % 10 == 0:
    print(f"Epoch [{epoch+1}/50] | Loss: {loss.item():.4f}")
model.eval()
with torch.no_grad():
  pred = model(data.x, data.edge_index).argmax(dim=1)
  correct = (pred[data.test_mask] == data.y[data.test_mask]).sum()
  acc = int(correct) / int(data.test_mask.sum())
  print(f"\n--- GNN FRAUD DETECTION RESULTS ---")
  print(f"Test Set Accuracy in Detecting Real vs Fraud: {acc * 100:.2f}%")