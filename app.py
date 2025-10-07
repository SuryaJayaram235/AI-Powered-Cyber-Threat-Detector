import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, precision_recall_fscore_support
)
from sklearn.ensemble import IsolationForest
from xgboost import XGBClassifier
from tensorflow import keras
import seaborn as sns
import matplotlib.pyplot as plt

from preprocess import preprocess_generic
from features import build_features

st.set_page_config(page_title="AI Cybersecurity Threat Detection", layout="wide")

st.title("🛡️ AI-Powered Cybersecurity Threat Detection")
st.write("Run anomaly detection & classification models.")

# --- Sidebar Controls ---
st.sidebar.header("⚙️ Settings")
contamination = st.sidebar.slider("Contamination (IsolationForest)", 0.001, 0.5, 0.01)
n_estimators = st.sidebar.slider("n_estimators (XGBoost)", 50, 500, 100, step=50)
threshold = st.sidebar.slider("Anomaly Threshold", 0.0, 1.0, 0.5, step=0.01)
random_state = st.sidebar.number_input("Random State", value=42, step=1)
max_rows = st.sidebar.number_input("Max Rows to Load", value=50000, step=1000)

# Auto-run all models
st.sidebar.info("⚡ All models run automatically once a dataset is uploaded.")

# --- File Upload ---
uploaded = st.file_uploader("Upload a CSV dataset", type="csv")
if uploaded is not None:
    df_raw = pd.read_csv(uploaded, low_memory=False)
    dataset_name = uploaded.name.replace(".csv", "")
    st.session_state["dataset_name"] = dataset_name
else:
    dataset_name = "unknown_dataset"

# --- Processing Pipeline ---
if uploaded is not None:
    st.subheader("1) Load & Preprocess Data")
    st.write(f"Dataset: `{dataset_name}` with {df_raw.shape[0]} rows, {df_raw.shape[1]} columns.")

    df = preprocess_generic(df_raw)
    if len(df) > max_rows:
        df = df.sample(n=max_rows, random_state=random_state)
        st.info(f"Sampled {max_rows} rows for faster training.")

    st.dataframe(df.head())

    # --- Feature Engineering ---
    st.subheader("2) Feature Engineering")

    X, cols = build_features(df)
    st.write("Feature matrix shape:", X.shape)

    engineered_only = [c for c in X.columns if c not in df.columns]
    if engineered_only:
        st.write("🛠️ Engineered-only features:")
        st.dataframe(X[engineered_only].head())
    else:
        st.info("No extra engineered features detected (only scaling applied).")

    # --- Check for labels ---
    has_labels = "is_malicious" in df.columns and df["is_malicious"].notna().any()
    if has_labels:
        unique_labels = np.unique(df["is_malicious"].dropna())
        eval_mode = "binary" if len(unique_labels) == 2 else "macro"
    else:
        eval_mode = None

    # --- Model Training ---
    st.subheader("3) Train and Compare Models")
    results = []

    progress_bar = st.progress(0)
    progress_text = st.empty()
    total_models = 3
    completed = 0  # global counter

    def update_progress():
        global completed
        completed += 1
        percent = int((completed / total_models) * 100)
        progress_bar.progress(percent)
        progress_text.text(f"Progress: {percent}% ({completed}/{total_models} models finished)")

    # ---- Run models sequentially ----
    # IsolationForest
    st.info("🔍 Running IsolationForest...")
    iso = IsolationForest(contamination=contamination, random_state=random_state)
    iso.fit(X)
    iso_score = -iso.decision_function(X)
    df["iso_score"] = iso_score
    if has_labels:
        if eval_mode == "binary":
            auc = roc_auc_score(df["is_malicious"], iso_score)
            p, r, f1, _ = precision_recall_fscore_support(
                df["is_malicious"], (iso_score >= threshold).astype(int),
                average="binary", zero_division=0
            )
        else:
            y_pred = (iso_score >= threshold).astype(int)
            auc = 0
            p, r, f1, _ = precision_recall_fscore_support(
                df["is_malicious"], y_pred, average="macro", zero_division=0
            )
        results.append(("IsolationForest", auc, p, r, f1))
    st.success("✅ IsolationForest finished")
    update_progress()

    # XGBoost
    if has_labels:
        st.info("⚡ Training XGBoost...")
        xgb = XGBClassifier(n_estimators=n_estimators, random_state=random_state,
                            eval_metric="logloss", use_label_encoder=False)
        xgb.fit(X, df["is_malicious"])
        y_pred = xgb.predict(X)
        y_score = xgb.predict_proba(X)
        if eval_mode == "binary":
            auc = roc_auc_score(df["is_malicious"], y_score[:, 1])
            p, r, f1, _ = precision_recall_fscore_support(
                df["is_malicious"], y_pred, average="binary", zero_division=0
            )
        else:
            auc = roc_auc_score(df["is_malicious"], y_score, multi_class="ovr", average="macro")
            p, r, f1, _ = precision_recall_fscore_support(
                df["is_malicious"], y_pred, average="macro", zero_division=0
            )
        df["xgb_score"] = y_score[:, 1] if eval_mode == "binary" else np.max(y_score, axis=1)
        results.append(("XGBoost", auc, p, r, f1))
        st.success("✅ XGBoost finished")
    else:
        st.warning("⚠️ Skipped XGBoost (no labels).")
    update_progress()

    # Autoencoder
    st.info("🧠 Training Autoencoder...")
    input_dim = X.shape[1]
    ae = keras.Sequential([
        keras.layers.InputLayer(shape=(input_dim,)),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(16, activation="relu"),
        keras.layers.Dense(32, activation="relu"),
        keras.layers.Dense(input_dim, activation="linear")
    ])
    ae.compile(optimizer="adam", loss="mse")
    ae.fit(X, X, epochs=5, batch_size=64, verbose=0)
    recon = ae.predict(X)
    ae_score = np.mean(np.square(X - recon), axis=1)
    df["ae_score"] = ae_score
    if has_labels:
        if eval_mode == "binary":
            auc = roc_auc_score(df["is_malicious"], ae_score)
            p, r, f1, _ = precision_recall_fscore_support(
                df["is_malicious"], (ae_score >= threshold).astype(int),
                average="binary", zero_division=0
            )
        else:
            y_pred = (ae_score >= threshold).astype(int)
            auc = 0
            p, r, f1, _ = precision_recall_fscore_support(
                df["is_malicious"], y_pred, average="macro", zero_division=0
            )
        results.append(("Autoencoder", auc, p, r, f1))
    st.success("✅ Autoencoder finished")
    update_progress()

    progress_text.success("🎉 All models finished!")

    # --- Leaderboard ---
    st.subheader("📊 Model Leaderboard")
    if has_labels and results:
        res_df = pd.DataFrame(results, columns=["Model", "ROC-AUC", "Precision", "Recall", "F1"])
        res_df.insert(0, "Dataset", dataset_name)
        st.dataframe(res_df, use_container_width=True)

    # --- Risk & Exposure Analysis ---
    st.subheader("4) Risk Analysis & Exposure")

    if any(c in df.columns for c in ["iso_score", "xgb_score", "ae_score"]):
        def risk_level(score):
            if score < 0.33:
                return "Low"
            elif score < 0.66:
                return "Medium"
            else:
                return "High"

        score_col = "xgb_score" if "xgb_score" in df else ("iso_score" if "iso_score" in df else "ae_score")
        df["risk_level"] = df[score_col].apply(risk_level)

        risk_counts = df["risk_level"].value_counts(normalize=True) * 100
        st.write("📊 Risk distribution (% of records):")
        st.bar_chart(risk_counts)

        st.markdown("**Legend:** 🟢 Low | 🟡 Medium | 🔴 High")

        st.write("⚠️ Exposure Summary")
        st.markdown(f"""
        - **High Risk:** {risk_counts.get('High', 0):.2f}% of dataset  
        - **Medium Risk:** {risk_counts.get('Medium', 0):.2f}% of dataset  
        - **Low Risk:** {risk_counts.get('Low', 0):.2f}% of dataset  
        """)

        st.write("🔥 Heatmap of Features vs Risk Level")
        try:
            risk_map = {"Low": 0, "Medium": 1, "High": 2}
            df["risk_level_num"] = df["risk_level"].map(risk_map)
            corr = df.corr(numeric_only=True)
            if "risk_level_num" in corr.columns:
                fig, ax = plt.subplots(figsize=(10, 6))
                sns.heatmap(
                    corr[["risk_level_num"]].sort_values(by="risk_level_num", ascending=False),
                    annot=True, cmap="coolwarm", ax=ax
                )
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Could not generate heatmap: {e}")

        if "Timestamp" in df.columns:
            try:
                st.write("⏱️ Risk Over Time")
                df["Timestamp"] = pd.to_datetime(df["Timestamp"], errors="coerce")
                time_risk = df.groupby(pd.Grouper(key="Timestamp", freq="H"))["risk_level"].value_counts(normalize=True).unstack().fillna(0)
                st.line_chart(time_risk)
            except Exception as e:
                st.warning(f"Time-based exposure could not be plotted: {e}")

    # --- Feature Importance (XGBoost) ---
    if "xgb_score" in df.columns:
        st.subheader("5) XGBoost Feature Importance")
        try:
            importances = xgb.feature_importances_
            feat_df = pd.DataFrame({
                "Feature": cols,
                "Importance": importances
            }).sort_values(by="Importance", ascending=False)

            st.write("Top 15 most important features driving XGBoost decisions:")
            st.bar_chart(feat_df.set_index("Feature").head(15))

            with st.expander("🔎 View full feature importance table"):
                st.dataframe(feat_df)

        except Exception as e:
            st.warning(f"Could not calculate feature importance: {e}")

    # --- SHAP Explainability ---
    if "xgb_score" in df.columns:
        st.subheader("6) SHAP Explainability (XGBoost)")
        try:
            import shap

            # Sample to speed up SHAP
            shap_sample = X.sample(min(2000, len(X)), random_state=random_state)
            explainer = shap.TreeExplainer(xgb)
            shap_values = explainer.shap_values(shap_sample)

            # --- Global summary ---
            st.write("🌍 Global Feature Impact (Top Predictors Across Dataset)")
            fig, ax = plt.subplots(figsize=(10, 6))
            shap.summary_plot(shap_values, shap_sample, feature_names=cols, plot_type="bar", show=False)
            st.pyplot(fig)

            # --- Instance-level explanation ---
            st.write("🔎 Explain Individual Predictions")
            row_idx = st.slider("Select row to explain", 0, len(shap_sample) - 1, 0)
            st.write("Instance Data (sampled):", shap_sample.iloc[row_idx])

            # Fix for SHAP v0.20+ force_plot
            base_val = explainer.expected_value[0] if isinstance(explainer.expected_value, (list, np.ndarray)) else explainer.expected_value

            fig2, ax2 = plt.subplots(figsize=(10, 6))
            shap.force_plot(
                base_val,
                shap_values[row_idx],
                shap_sample.iloc[row_idx, :],
                matplotlib=True,
                show=False
            )
            st.pyplot(fig2)

            st.caption("⚡ SHAP explanations run on a sample of up to 2000 rows for performance.")

        except Exception as e:
            st.warning(f"Could not generate SHAP explanations: {e}")
