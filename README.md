# 🛡️ AI-Powered Cybersecurity Threat Detection App

This is a **Streamlit web application** for detecting cybersecurity threats using **machine learning**.  
automatically runs anomaly detection & classification models.  

## 🚀 Features
- Upload **CSV dataset** 
- Automatic **preprocessing & feature engineering**.
- Multiple ML models:
  - **IsolationForest** (unsupervised anomaly detection)
  - **XGBoost** (supervised classification if labels available)
  - **Autoencoder** (deep learning anomaly detection)
- 📊 Leaderboard (ROC-AUC, Precision, Recall, F1)
- 🔥 Risk analysis (low/medium/high)
- 🌍 Heatmap of feature correlation with risk
- ⏱️ Risk over time (if timestamps available)
- 🔎 SHAP explainability (global + instance-level)