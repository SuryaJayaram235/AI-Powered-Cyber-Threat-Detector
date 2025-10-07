import pandas as pd
import numpy as np
import ipaddress
import streamlit as st

def ip_to_int(ip: str) -> int:
    """Convert IP address string to integer. Handles invalid gracefully."""
    try:
        return int(ipaddress.ip_address(ip))
    except Exception:
        return 0

def preprocess_generic(df: pd.DataFrame) -> pd.DataFrame:
    """
    Universal preprocessing for cybersecurity datasets:
    - Detects and encodes label column (binary or multi-class)
    - Handles timestamps (hour, dayofweek)
    - Converts IP addresses to numeric
    - Factorizes other categorical string columns
    - Fills missing values with 0
    """

    df = df.copy()

    # --- Detect label column ---
    label_candidates = [c for c in df.columns if c.lower() in
                        ["label", "status", "attack", "target", "malicious", "class", "y"]]

    if label_candidates:
        label_col = label_candidates[0]
        labels = df[label_col].astype(str).str.strip()
        unique_labels = labels.unique()

        if len(unique_labels) <= 2:
            # Binary case
            df["is_malicious"] = labels.apply(
                lambda x: 0 if x.lower() in ["benign", "normal", "legit", "good"] else 1
            )
            st.session_state["label_map"] = {0: "Benign/Normal", 1: "Malicious"}
        else:
            # Multi-class case
            df["is_malicious"], uniques = pd.factorize(labels)
            st.session_state["label_map"] = {i: label for i, label in enumerate(uniques)}
    else:
        # No label column found
        df["is_malicious"] = None
        st.session_state["label_map"] = {}

    # --- Handle timestamp columns ---
    ts_candidates = [c for c in df.columns if "time" in c.lower() or "date" in c.lower()]
    if ts_candidates:
        ts_col = ts_candidates[0]
        df["timestamp"] = pd.to_datetime(df[ts_col], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour.fillna(0).astype(int)
        df["dayofweek"] = df["timestamp"].dt.dayofweek.fillna(0).astype(int)

    # --- Handle IP address columns ---
    ip_candidates = [c for c in df.columns if "ip" in c.lower()]
    for col in ip_candidates:
        df[col + "_int"] = df[col].astype(str).apply(ip_to_int)

    # --- Encode other categoricals (factorize to integers) ---
    for col in df.columns:
        if df[col].dtype == object and col not in ip_candidates:
            df[col] = pd.factorize(df[col].astype(str))[0]

    # --- Fill NaNs ---
    df = df.fillna(0)

    return df
