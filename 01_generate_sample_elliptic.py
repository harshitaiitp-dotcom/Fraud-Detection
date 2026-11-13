import os
import numpy as np
import pandas as pd
def generate_mini_elliptic(data_dir="data", num_nodes=500):
    os.makedirs(data_dir, exist_ok=True)
    np.random.seed(42)  
    print(f"Creating mini Elliptic dataset with {num_nodes} transactions...")
    tx_ids = np.arange(100000, 100000 + num_nodes)
    timesteps = np.random.choice([1, 2, 3, 4, 5], size=num_nodes, p=[0.25, 0.25, 0.2, 0.15, 0.15])
    classes = np.random.choice(['1', '2', 'unknown'], size=num_nodes, p=[0.25, 0.10, 0.65])
    classes_df = pd.DataFrame({'txId': tx_ids, 'class': classes})
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    classes_df.to_csv(classes_path, index=False)
    print(f" Saved: {classes_path} ({len(classes_df)} rows)")
    edges = []
    illicit_indices = np.where(classes == '2')[0]
    if len(illicit_indices) > 5:
        src = tx_ids[illicit_indices[0]]
        sink = tx_ids[illicit_indices[-1]]
        for mid_idx in illicit_indices[1:-1]:
            edges.append({'txId1': src, 'txId2': tx_ids[mid_idx]})
            edges.append({'txId1': tx_ids[mid_idx], 'txId2': sink})
    for i in range(num_nodes - 1):
        if np.random.rand() > 0.6:
            valid_targets = np.where(timesteps >= timesteps[i])[0]
            valid_targets = valid_targets[valid_targets != i]
            if len(valid_targets) > 0:
                target_idx = np.random.choice(valid_targets)
                edges.append({'txId1': tx_ids[i], 'txId2': tx_ids[target_idx]})
    edges_df = pd.DataFrame(edges).drop_duplicates()
    edges_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    edges_df.to_csv(edges_path, index=False)
    print(f" Saved: {edges_path} ({len(edges_df)} edges)")
    local_feats = np.zeros((num_nodes, 94))
    for i in range(num_nodes):
        if classes[i] == '2': 
            local_feats[i] = np.random.normal(loc=1.5, scale=0.8, size=94)
        else: 
            local_feats[i] = np.random.normal(loc=-0.5, scale=1.0, size=94)
    agg_feats = np.random.normal(loc=0.0, scale=1.0, size=(num_nodes, 72))
    all_features = np.hstack([
        tx_ids.reshape(-1, 1),
        timesteps.reshape(-1, 1),
        local_feats,
        agg_feats
    ])
    features_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    pd.DataFrame(all_features).to_csv(features_path, header=False, index=False)
    print(f" Saved: {features_path} ({all_features.shape[0]} rows x {all_features.shape[1]} columns)")
    print("\n Sample dataset ready! You can now run '02_explore_graph.py'.")
if __name__ == "__main__":
    generate_mini_elliptic()