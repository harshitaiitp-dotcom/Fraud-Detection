import os
import pandas as pd
import numpy as np
def explore_data(data_dir="data"):
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    edges_path = os.path.join(data_dir, "elliptic_txs_edgelist.csv")
    features_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    if not os.path.exists(classes_path):
        print(f"Error: Could not find files in '{data_dir}/'. Please run '01_generate_sample_elliptic.py' first!")
        return
    print("=" * 60)
    print("STEP 1: INSPECTING TRANSACTION CLASSES (LABELS)")
    print("=" * 60)
    classes_df = pd.read_csv(classes_path)
    print(f"Total Transactions: {len(classes_df):,}")
    counts = classes_df['class'].value_counts()
    for label, count in counts.items():
        meaning = "Licit (Legitimate)" if str(label) == '1' else ("Illicit (Fraud/Crime)" if str(label) == '2' else "Unknown/Unlabeled")
        pct = (count / len(classes_df)) * 100
        print(f"  Class '{label}' ({meaning:<22}): {count:>6} ({pct:.1f}%)")

    print("\n" + "=" * 60)
    print("STEP 2: INSPECTING EDGES (TRANSACTION PAYMENTS)")
    print("=" * 60)
    edges_df = pd.read_csv(edges_path)
    print(f"Total Payment Edges: {len(edges_df):,}")
    print("Sample Payment Flows (txId1 sends money to txId2):")
    print(edges_df.head(5).to_string(index=False))

    print("\n" + "=" * 60)
    print("STEP 3: INSPECTING FEATURES")
    print("=" * 60)
    features_df = pd.read_csv(features_path, header=None)
    num_rows, num_cols = features_df.shape
    print(f"Features Table Shape: {num_rows} rows x {num_cols} columns")
    print(f"  - Column 0: Transaction ID")
    print(f"  - Column 1: Timestep")
    print(f"  - Columns 2-95: 94 Local Features (Direct transaction attributes)")
    print(f"  - Columns 96-{num_cols-1}: 72 Aggregated Features (1-hop neighbor attributes)")
    features_df.rename(columns={0: 'txId', 1: 'timestep'}, inplace=True)
    merged = pd.merge(features_df, classes_df, on='txId')
    labeled = merged[merged['class'].isin(['1', '2'])]    
    print("\nComparing Feature 2 (e.g. Transaction Fee) across classes:")
    mean_licit = labeled[labeled['class'] == '1'][2].mean()
    mean_illicit = labeled[labeled['class'] == '2'][2].mean()
    print(f"  Mean Feature 2 for Licit (Class 1)   : {mean_licit:.4f}")
    print(f"  Mean Feature 2 for Illicit (Class 2) : {mean_illicit:.4f}")
    print(" Notice how fraudulent transactions have a distinct statistical profile!")
    print("=" * 60)
    print("\nNext: Run '03_train_generative_model.py' to train a GenAI model on these patterns.")
if __name__ == "__main__":
    explore_data()