import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def build_features(df: pd.DataFrame):
    """
    Build ML-ready feature matrix from dataframe.
    - Keeps numeric columns
    - One-hot encodes categorical columns (top 20 levels per column)
    - Drops label column `is_malicious`
    - Applies StandardScaler to normalize features
    """

    df = df.copy()

    # --- Numeric features ---
    num_df = df.select_dtypes(include=["number"]).copy()

    # --- Categorical features ---
    cat_df = df.select_dtypes(include=["object", "category"]).copy()

    if not cat_df.empty:
        # Clean strings
        cat_df = cat_df.apply(lambda col: col.astype(str).str.lower().str.strip())
        
        # Keep only top 20 categories per column
        top_cats = {
            col: cat_df[col].value_counts().index[:20]
            for col in cat_df.columns
        }
        for col, top in top_cats.items():
            cat_df[col] = cat_df[col].apply(lambda x: x if x in top else "other")

        # ✅ Use sparse_output for sklearn >= 1.2
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        cat_encoded = ohe.fit_transform(cat_df)
        cat_feature_names = ohe.get_feature_names_out(cat_df.columns)
        cat_df = pd.DataFrame(cat_encoded, columns=cat_feature_names, index=cat_df.index)
    else:
        cat_df = pd.DataFrame()

    # --- Combine numeric + categorical ---
    X = pd.concat([num_df, cat_df], axis=1)

    # Drop labels if present
    if "is_malicious" in X.columns:
        X = X.drop(columns=["is_malicious"])

    # --- Scale features ---
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)

    return X_scaled, X.columns.tolist()
