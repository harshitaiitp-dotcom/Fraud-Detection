import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
print("Loading data...")
features_df = pd.read_csv("data/elliptic_txs_features.csv", header=None)
classes_df = pd.read_csv("data/elliptic_txs_classes.csv")
illicit_ids = classes_df[classes_df["class"] == "2"]["txId"]
real_illicit = features_df[features_df[0].isin(illicit_ids)].iloc[
    :, 2:96
]
synthetic_illicit = pd.read_csv("data/synthetic_illicit_txs.csv")
if "txId" in synthetic_illicit.columns:
    synthetic_illicit = synthetic_illicit.drop(columns=["txId"])
min_features = min(real_illicit.shape[1], synthetic_illicit.shape[1])
X_real = real_illicit.values[:, :min_features]
X_syn = synthetic_illicit.values[:, :min_features]
y_real = np.zeros(len(X_real))
y_syn = np.ones(len(X_syn))
X = np.vstack([X_real, X_syn])
y = np.hstack([y_real, y_syn])
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\n--- RESULTS: REAL VS SYNTHETIC CLASSIFICATION ---")
print(
    classification_report(y_test, y_pred, target_names=["Real", "Synthetic"])
)
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))