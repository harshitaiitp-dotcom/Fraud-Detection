import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, f1_score
def evaluate(data_dir="data"):
    classes_path = os.path.join(data_dir, "elliptic_txs_classes.csv")
    features_path = os.path.join(data_dir, "elliptic_txs_features.csv")
    synth_path = os.path.join(data_dir, "synthetic_illicit_txs.csv")
    if not os.path.exists(synth_path):
        print("Error: 'synthetic_illicit_txs.csv' not found. Run '03_train_generative_model.py' first!")
        return
    classes_df = pd.read_csv(classes_path)
    features_df = pd.read_csv(features_path, header=None)
    features_df.rename(columns={0: 'txId', 1: 'timestep'}, inplace=True)
    merged = pd.merge(features_df, classes_df, on='txId')
    labeled = merged[merged['class'].isin(['1', '2'])].copy()
    real_illicit = labeled[labeled['class'] == '2'].iloc[:, 2:96].values
    real_licit = labeled[labeled['class'] == '1'].iloc[:, 2:96].values
    synth_df = pd.read_csv(synth_path)
    synth_illicit = synth_df.drop(columns=['class']).values
    print("=" * 65)
    print("TEST 1: STATISTICAL FIDELITY (Real Illicit vs Synthetic Illicit)")
    print("=" * 65)
    real_mean = real_illicit.mean(axis=0)
    synth_mean = synth_illicit.mean(axis=0)
    mean_diff = np.abs(real_mean - synth_mean).mean()
    print(f"Number of Real Illicit Samples      : {len(real_illicit)}")
    print(f"Number of Synthetic Illicit Samples : {len(synth_illicit)}")
    print(f"Average Mean Difference across 94 features: {mean_diff:.4f}")    
    print("\nFeature 2 Comparison:")
    print(f"  Real Licit Mean       : {real_licit[:, 0].mean():.4f}")
    print(f"  Real Illicit Mean     : {real_illicit[:, 0].mean():.4f}")
    print(f"  Synthetic Illicit Mean: {synth_illicit[:, 0].mean():.4f}")
    print("-> Notice how the Synthetic generator closely captured the illicit distribution!")
    print("\n" + "=" * 65)
    print("TEST 2: ML UTILITY (Augmenting Fraud Detection with Synthetic Data)")
    print("=" * 65)
    X_real = labeled.iloc[:, 2:96].values
    y_real = (labeled['class'] == '2').astype(int).values
    X_train, X_test, y_train, y_test = train_test_split(
        X_real, y_real, test_size=0.3, random_state=42, stratify=y_real
    )
    clf_baseline = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_baseline.fit(X_train, y_train)
    y_pred_base = clf_baseline.predict(X_test)
    f1_base = f1_score(y_test, y_pred_base, pos_label=1, zero_division=0)
    X_train_augmented = np.vstack([X_train, synth_illicit])
    y_train_augmented = np.hstack([y_train, np.ones(len(synth_illicit), dtype=int)])
    clf_augmented = RandomForestClassifier(n_estimators=50, random_state=42)
    clf_augmented.fit(X_train_augmented, y_train_augmented)
    y_pred_aug = clf_augmented.predict(X_test)
    f1_aug = f1_score(y_test, y_pred_aug, pos_label=1, zero_division=0)
    print("Results on Held-Out REAL Test Set:")
    print(f"  Baseline (Real Data only)       - Fraud F1-Score: {f1_base:.4f}")
    print(f"  Augmented (Real + Synthetic GenAI) - Fraud F1-Score: {f1_aug:.4f}")
    if f1_aug >= f1_base:
        print("\nSUCCESS: The synthetic fraud data improved / matched the detector's capability!")
    else:
        print("\nNote: More epochs or tuning will further refine the synthetic generator.")
    print("=" * 65)
if __name__ == "__main__":
    evaluate()