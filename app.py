import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
import streamlit as st
st.set_page_config(
    page_title="Blockchain Fraud Detection System", layout="wide"
)
st.title("Blockchain Fraud Detection System (Surrogate GNN & XAI)")
st.markdown(
    "Upload Elliptic-formatted transaction features (94 local features) to"
    " detect illicit transactions, inspect SHAP explanations, and test"
    " adversarial robustness."
)
@st.cache_resource
def load_or_train_models():
  try:
    features_df = pd.read_csv("data/elliptic_txs_features.csv", header=None)
    classes_df = pd.read_csv("data/elliptic_txs_classes.csv")
    classes_df["label"] = classes_df["class"].map({"1": 0, "2": 1})
    labeled = classes_df[classes_df["class"] != "unknown"]
    merged = pd.merge(labeled, features_df, left_on="txId", right_on=0)
    X_train = merged.iloc[:, 3:97].values
    y_train = merged["label"].values
  except Exception:
    np.random.seed(42)
    X_licit = np.random.normal(0, 1, (400, 94))
    X_illicit = np.random.normal(1.2, 1.2, (100, 94))
    X_train = np.vstack([X_licit, X_illicit])
    y_train = np.array([0] * 400 + [1] * 100)
  rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
  rf_model.fit(X_train, y_train)
  explainer = shap.TreeExplainer(rf_model)
  return rf_model, explainer
rf_model, explainer = load_or_train_models()
st.sidebar.header("Simulation Settings")
adv_noise = st.sidebar.slider(
    "Adversarial Perturbation Noise (Evasion Test)", 0.0, 3.0, 0.0, 0.1
)
use_sample = st.sidebar.checkbox(
    "Use built-in sample transactions (if no CSV uploaded)", value=True
)
uploaded_file = st.file_uploader(
    "Upload Transaction Features CSV (94 columns)", type=["csv"]
)
df = None
if uploaded_file is not None:
  df = pd.read_csv(uploaded_file)
elif use_sample:
  np.random.seed(99)
  sample_data = np.vstack(
      [np.random.normal(0, 1, (8, 94)), np.random.normal(1.5, 1.0, (4, 94))]
  )
  df = pd.DataFrame(
      sample_data, columns=[f"Local_Feature_{i}" for i in range(94)]
  )
if df is not None:
  st.subheader("Transaction Data Preview")
  st.dataframe(df.head())
  numeric_df = df.select_dtypes(include=[np.number]).iloc[:, :94]
  if numeric_df.shape[1] < 94:
    st.error(
        f"Expected 94 numeric feature columns, but found {numeric_df.shape[1]}."
    )
  else:
    if st.button("Run Fraud Detection & XAI Analysis", type="primary"):
      X_input = numeric_df.values.copy()

      if adv_noise > 0:
        np.random.seed(7)
        X_input += np.random.normal(0, adv_noise, X_input.shape)
        st.warning(
            f"Applied Gaussian evasion noise (std = {adv_noise}) to test model"
            " robustness."
        )
      probs = rf_model.predict_proba(X_input)[:, 1]
      preds = (probs > 0.5).astype(int)
      results_df = pd.DataFrame({
          "Transaction_Row": np.arange(len(X_input)),
          "Prediction": np.where(preds == 1, "ILLICIT (FRAUD)", "LICIT"),
          "Fraud_Probability": np.round(probs, 4),
      })
      col1, col2 = st.columns([1, 1])
      with col1:
        st.subheader("Detection Results")
        st.dataframe(results_df, use_container_width=True)
        st.metric("Flagged Illicit Transactions", int(preds.sum()))
      with col2:
        st.subheader("SHAP Local Explanation (Top Flagged Node)")
        target_idx = (
            int(np.argmax(probs)) if preds.sum() > 0 else 0
        )
        shap_vals = explainer.shap_values(X_input[target_idx : target_idx + 1])
        if isinstance(shap_vals, list):
          node_shap = np.array(shap_vals[1]).flatten()
        elif len(np.array(shap_vals).shape) == 3:
          node_shap = np.array(shap_vals)[0, :, 1]
        else:
          node_shap = np.array(shap_vals).flatten()
        top_idx = np.argsort(np.abs(node_shap))[::-1][:5]
        xai_df = pd.DataFrame({
            "Feature": [f"Local_Feature_{i}" for i in top_idx],
            "Input_Value": np.round(X_input[target_idx, top_idx], 4),
            "SHAP_Impact": np.round(node_shap[top_idx], 4),
        })
        st.write(f"Explaining **Transaction Row {target_idx}**:")
        st.dataframe(xai_df, use_container_width=True)
