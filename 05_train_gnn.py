import os
import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import SAGEConv
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
def train_gnn(data_dir="data"):
    print("Loading data for Graph Neural Network...")
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    edges_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    features_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    classes_df = pd.read_csv(classes_path)
    features_df = pd.read_csv(features_path, header=None)
    features_df.rename(columns={0: 'txId', 1: 'timestep'}, inplace=True)
    merged = pd.merge(features_df, classes_df, on='txId', how='inner')
    merged['node_idx'] = range(len(merged))
    id_to_idx = dict(zip(merged['txId'], merged['node_idx']))
    labeled_df = merged[merged['class'].isin(['1', '2'])].copy()    
    x = torch.tensor(merged.iloc[:, 2:96].values, dtype=torch.float)
    y_raw = (merged['class'] == '2').astype(int).values
    y = torch.tensor(y_raw, dtype=torch.long)
    edges_df = pd.read_csv(edges_path)
    edges_df = edges_df[edges_df['txId1'].isin(id_to_idx) & edges_df['txId2'].isin(id_to_idx)]   
    src = edges_df['txId1'].map(id_to_idx).values
    dst = edges_df['txId2'].map(id_to_idx).values
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, y=y)
    labeled_indices = labeled_df['node_idx'].values
    train_idx, test_idx = train_test_split(labeled_indices, test_size=0.3, random_state=42, stratify=y_raw[labeled_indices])
    train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
    train_mask[train_idx] = True
    test_mask[test_idx] = True
    data.train_mask = train_mask
    data.test_mask = test_mask
    print(f"Graph loaded successfully!")
    print(f"  - Total Nodes: {data.num_nodes:,}")
    print(f"  - Total Edges: {data.num_edges:,}")
    print(f"  - Training Nodes: {train_mask.sum().item()}")
    print(f"  - Test Nodes: {test_mask.sum().item()}")
    class FraudGNN(torch.nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = SAGEConv(in_channels, hidden_channels)
            self.conv2 = SAGEConv(hidden_channels, out_channels)
        def forward(self, x, edge_index):
            x = self.conv1(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
            x = self.conv2(x, edge_index)
            return x
    model = FraudGNN(in_channels=data.num_features, hidden_channels=64, out_channels=2)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
    criterion = torch.nn.CrossEntropyLoss()
    print("\nTraining GNN Model...")
    model.train()
    for epoch in range(1, 41):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()    
        if epoch % 10 == 0:
            print(f"Epoch {epoch:02d} | Loss: {loss.item():.4f}")
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)       
        test_pred = pred[data.test_mask].numpy()
        test_true = data.y[data.test_mask].numpy()
        f1 = f1_score(test_true, test_pred, pos_label=1, zero_division=0)
        print("\n" + "=" * 50)
        print(f"GNN Test Set Fraud F1-Score: {f1:.4f}")
        print("=" * 50)
if __name__ == "__main__":
    train_gnn()