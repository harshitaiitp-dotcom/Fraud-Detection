import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier

print("--- EXPLAINABLE AI (XAI) MODULE FOR FRAUD DETECTION ---")

# 1. Load Data
print("Loading transaction dataset...")
features_df = pd.read_csv("data/elliptic_txs_features.csv", header=None)
classes_df = pd.read_csv("data/elliptic_txs_classes.csv")

# Extract labeled data (Licit = 1, Illicit = 2)
classes_df["label"] = classes_df["class"].map({"1": 0, "2": 1})
labeled_df = classes_df[classes_df["class"] != "unknown"].copy()
merged_df = pd.merge(labeled_df, features_df, left_on="txId", right_on=0)

# Extract Local Features (Columns 3 to 97)
X = merged_df.iloc[:, 3:97].values
y = merged_df["label"].values

feature_names = [f"Local_Feature_{i}" for i in range(X.shape[1])]

# 2. Train Surrogate Model for Global/Local XAI
print("Training explainable surrogate model...")
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X, y)

# 3. Global Feature Importance (Top Fraud Drivers)
importances = clf.feature_importances_
indices = np.argsort(importances)[::-1]

print("\n--- TOP 10 GLOBAL FRAUD DRIVERS ---")
for r in range(10):
  idx = indices[r]
  print(
      f"Rank {r+1}: Feature {idx} ({feature_names[idx]}) -> Importance:"
      f" {importances[idx]:.4f}"
  )

# 4. SHAP (SHapley Additive exPlanations) for Local Decision Explanation
print("\nCalculating SHAP values for local instance explanation...")
explainer = shap.TreeExplainer(clf)
sample_idx = np.where(y == 1)[0][0]  # Select a sample illicit transaction
sample_x = X[sample_idx : sample_idx + 1]

shap_values = explainer.shap_values(sample_x)

print(
    f"\n--- LOCAL EXPLANATION FOR FRAUD TRANSACTION (ID:"
    f" {merged_df.iloc[sample_idx]['txId']}) ---"
)
print("Top features pushing decision toward FRAUD classification:")

# Flatten or unpack shap_values safely regardless of shape
if isinstance(shap_values, list):
  node_shap = np.array(shap_values[1]).flatten()
elif len(np.array(shap_values).shape) == 3:
  node_shap = np.array(shap_values)[0, :, 1]
else:
  node_shap = np.array(shap_values).flatten()

top_shap_idx = np.argsort(np.abs(node_shap))[::-1][:5]
for idx in top_shap_idx:
  val = sample_x[0, idx]
  impact = node_shap[idx]
  print(
      f"  * {feature_names[idx]}: Value = {val:.4f} | SHAP Impact ="
      f" {impact:+.4f}"
  )