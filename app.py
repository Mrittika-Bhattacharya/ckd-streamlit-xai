
import os
import json
import numpy as np
import pandas as pd
import joblib
import streamlit as st
import shap
import matplotlib.pyplot as plt

st.set_page_config(page_title="CKD Risk Prediction (Calibrated + XAI)", layout="wide")
st.title("CKD Risk Prediction — RF + Isotonic Calibration + Tuned Threshold + SHAP")

# -------------------------
# Load artifacts (cached)
# -------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("final_calibrated_model_isotonic.joblib")
    preprocess = joblib.load("preprocess_pipeline_aug_v2.joblib")
    with open("final_threshold.json", "r") as f:
        thr_info = json.load(f)
    final_thr = float(thr_info["final_threshold"])
    X_bg = np.load("shap_background.npy")

    return model, preprocess, final_thr, X_bg

final_model, preprocess, FINAL_THR, X_bg = load_artifacts()

# -------------------------
# Utilities
# -------------------------
def unwrap_estimator(calibrated_model):
    """
    For CalibratedClassifierCV, extract the underlying estimator (RF) for SHAP.
    Handles newer sklearn FrozenEstimator wrappers too.
    """
    def _unwrap(obj):
        if "FrozenEstimator" in str(type(obj)) and hasattr(obj, "estimator"):
            return obj.estimator
        return obj

    base_est = None
    if hasattr(calibrated_model, "calibrated_classifiers_"):
        cc0 = calibrated_model.calibrated_classifiers_[0]
        for attr in ["estimator", "base_estimator"]:
            if hasattr(cc0, attr):
                base_est = _unwrap(getattr(cc0, attr))
                break
    if base_est is None:
        for attr in ["estimator", "base_estimator"]:
            if hasattr(calibrated_model, attr):
                base_est = _unwrap(getattr(calibrated_model, attr))
                break
    if base_est is None:
        raise ValueError("Could not unwrap base estimator from calibrated model.")
    return base_est

@st.cache_resource
def get_shap_explainer():
    base_est = unwrap_estimator(final_model)
    explainer = shap.TreeExplainer(base_est, data=X_bg)
    return explainer

explainer = get_shap_explainer()

def transform_input(df_raw: pd.DataFrame) -> np.ndarray:
    # ---- Clean column names ----
    df_raw = df_raw.copy()
    df_raw.columns = [str(c).strip() for c in df_raw.columns]

    # Remove common junk columns from CSV export
    for junk in ["Unnamed: 0", "index", "Index"]:
        if junk in df_raw.columns:
            df_raw = df_raw.drop(columns=[junk])

    # Drop columns that shouldn't be learned
    drop_cols = []
    if "DoctorInCharge" in df_raw.columns:
        drop_cols.append("DoctorInCharge")
    if "Diagnosis" in df_raw.columns:
        df_raw = df_raw.drop(columns=["Diagnosis"])

    # ---- Check schema expected by preprocess ----
    # ColumnTransformer fitted on a pandas DataFrame usually stores expected input columns here:
    if hasattr(preprocess, "feature_names_in_"):
        required = list(preprocess.feature_names_in_)
    else:
        # fallback (less ideal): assume current columns are correct
        required = list(df_raw.columns)

    missing = sorted(list(set(required) - set(df_raw.columns)))

    if missing:
        st.error("❌ Uploaded CSV does not match the expected input schema.")
        st.write("### Missing required columns:")
        st.write(missing)

        st.write("### Expected columns (template):")
        st.write(required)

        st.write("### Your uploaded columns:")
        st.write(list(df_raw.columns))

        st.info(
            "Upload a RAW patient CSV like `demo_patients_raw_val.csv` from your notebook "
            "(contains all features). Prediction-export files will NOT work."
        )
        st.stop()

    # ---- Reorder columns to match training-time order ----
    df_raw = df_raw[required]

    # Transform
    X = preprocess.transform(df_raw.drop(columns=drop_cols, errors="ignore"))
    return X

def predict_proba_and_label(X: np.ndarray):
    proba_ckd = final_model.predict_proba(X)[:, 1]
    pred = (proba_ckd >= FINAL_THR).astype(int)
    return proba_ckd, pred

def group_shap_by_original_feature(shap_vals_row: np.ndarray, feature_names: np.ndarray):
    """
    Streamlit-friendly: group one-hot encoded features back to original feature names.
    feature_names look like: "num__Age", "cat__Ethnicity_3", etc.
    We'll group:
      - num__Age -> Age
      - cat__Ethnicity_3 -> Ethnicity
    """
    groups = {}
    for val, name in zip(shap_vals_row, feature_names):
        name = str(name)
        if name.startswith("num__"):
            base = name.replace("num__", "")
        elif name.startswith("cat__"):
            raw = name.replace("cat__", "")
            # take part before first "_" as original categorical column
            base = raw.split("_")[0] if "_" in raw else raw
        else:
            base = name

        groups[base] = groups.get(base, 0.0) + float(val)

    # return sorted dataframe
    df_g = pd.DataFrame({"feature": list(groups.keys()), "shap_sum": list(groups.values())})
    df_g["abs"] = df_g["shap_sum"].abs()
    df_g = df_g.sort_values("abs", ascending=False).drop(columns=["abs"]).reset_index(drop=True)
    return df_g

def plot_bar(df_g, title="Top SHAP contributions (grouped)"):
    top = df_g.head(12).iloc[::-1]
    plt.figure(figsize=(7, 4))
    plt.barh(top["feature"], top["shap_sum"])
    plt.title(title)
    plt.xlabel("SHAP contribution ( + increases CKD risk, - decreases )")
    plt.tight_layout()
    return plt.gcf()

# -------------------------
# UI — CSV upload
# -------------------------
st.markdown("""
### How to use
Upload a **CSV** containing one or more patient rows with the **same feature columns** you used for training  
(you can include `DoctorInCharge` and/or `Diagnosis` — the app will ignore them).
""")

uploaded = st.file_uploader("Upload patient CSV", type=["csv"])

if uploaded is None:
    st.info("Upload a CSV to get predictions + explainability.")
    st.stop()

df_in = pd.read_csv(uploaded)
st.write("Preview of uploaded data:", df_in.head())

# Transform + predict
X = transform_input(df_in)
proba, pred = predict_proba_and_label(X)

out = df_in.copy()
out["ckd_risk_proba"] = proba
out["decision_thr"] = FINAL_THR
out["ckd_pred_label"] = pred

st.subheader("Predictions")
st.dataframe(out)

# Choose one row for explanation
st.subheader("Explainability (SHAP) for a selected prediction")
row_idx = st.number_input("Select row index to explain", min_value=0, max_value=len(out)-1, value=0, step=1)

risk = float(out.loc[row_idx, "ckd_risk_proba"])
label = int(out.loc[row_idx, "ckd_pred_label"])

col1, col2, col3 = st.columns(3)
col1.metric("Calibrated CKD risk", f"{risk:.3f}")
col2.metric("Threshold used", f"{FINAL_THR:.3f}")
col3.metric("Decision", "CKD (High Risk)" if label == 1 else "Non-CKD (Lower Risk)")

# Compute SHAP for that one row
feature_names = np.array(preprocess.get_feature_names_out(), dtype=object)
sv = explainer.shap_values(X[row_idx:row_idx+1])

# binary trees often return [class0, class1]
if isinstance(sv, list) and len(sv) == 2:
    sv_row = sv[1][0]  # class-1 contributions
else:
    sv_row = np.array(sv)[0]

df_g = group_shap_by_original_feature(sv_row, feature_names)

# Doctor-friendly summary
st.markdown("#### Doctor-friendly summary (Top drivers)")
inc = df_g[df_g["shap_sum"] > 0].head(8)
dec = df_g[df_g["shap_sum"] < 0].head(8)

cA, cB = st.columns(2)
with cA:
    st.markdown("**Factors increasing CKD risk ( + )**")
    st.dataframe(inc)
with cB:
    st.markdown("**Factors decreasing CKD risk ( − )**")
    st.dataframe(dec)

# Plot
fig = plot_bar(df_g, title="Top SHAP contributions (grouped to original features)")
st.pyplot(fig)

st.caption("Note: SHAP shows which features pushed the model toward CKD vs Non-CKD for this specific patient, based on the Random Forest base estimator.")
