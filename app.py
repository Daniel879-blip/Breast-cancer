# app.py
"""
Medical-style Breast Cancer Prediction Streamlit App
- Single-file app
- Synthetic demo dataset (1000 patients) with demographic & medical features
- Upload your own CSV (must contain a target column you select)
- Sidebar: BAT parameters, classifier (Naive Bayes), train button, save artifacts
- Auto-generated patient form (categorical dropdowns, numeric sliders)
- Charts: distribution, correlation, BAT convergence, ROC, confusion matrix, feature importance
- Plain-English write-up and explanation on prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from io import StringIO

# plotting
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import matplotlib.pyplot as plt

# sklearn
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample

# persistence
from joblib import dump, load

st.set_page_config(page_title="Breast Cancer — Medical-style App", layout="wide")
st.title("🧬 Breast Cancer Prediction — Medical-style (BAT + Naive Bayes)")
st.markdown(
    "Upload your CSV (or use demo). Choose the target column in the sidebar. "
    "Adjust BAT parameters and press **Run BAT & Train**. After training, use the patient form to predict."
)

# ---------------------------
# Helpers
# ---------------------------
def make_onehot_encoder_compat(**kwargs):
    """Return OneHotEncoder compatible with different sklearn versions."""
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)

def explain_prediction_text(prob, label):
    pct = f"{prob*100:.1f}%" if prob is not None else "N/A"
    if label == 1:
        return (
            f"**Prediction:** POSITIVE (likely malignant). Confidence: **{pct}**.\n\n"
            "Interpretation: The model found the entered features more similar to malignant cases in the training data.\n\n"
            "Recommended next steps:\n"
            "1. Clinical consultation with imaging (mammogram/ultrasound/MRI).\n"
            "2. Biopsy if clinically indicated.\n"
            "3. Use this output as a screening aid only — not a diagnosis."
        )
    else:
        return (
            f"**Prediction:** NEGATIVE (likely benign). Confidence: **{pct}**.\n\n"
            "Interpretation: The model found the entered features more similar to benign cases in the training data.\n\n"
            "Recommended next steps:\n"
            "1. Continue routine screening.\n"
            "2. Consult clinician if symptoms persist.\n"
        )

# ---------------------------
# Synthetic demo dataset (1000 rows)
# ---------------------------
@st.cache_data
def generate_demo_df(n=1000, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 85, n)
    sex = rng.choice(['Female', 'Male'], size=n, p=[0.98, 0.02])
    tumor_size = np.round(rng.normal(12.0, 8.0, n).clip(1, 80), 1)  # mm
    family_history = rng.choice(['Yes', 'No'], size=n, p=[0.12, 0.88])
    bmi = np.round(rng.normal(27, 5, n).clip(15, 45), 1)
    blood_pressure = np.round(rng.normal(120, 15, n).clip(80, 200), 0)
    glucose = np.round(rng.normal(95, 15, n).clip(50, 250), 0)
    # base risk score linear combination -> probability via logistic
    score = (
        0.03 * (age - 40)
        + 0.08 * (tumor_size - 10)
        + 0.6 * (family_history == 'Yes').astype(float)
        + 0.02 * (bmi - 25)
        + 0.01 * (blood_pressure - 120)
    )
    prob = 1 / (1 + np.exp(-score))
    target = (rng.random(n) < prob).astype(int)
    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'tumor_size_mm': tumor_size,
        'family_history': family_history,
        'bmi': bmi,
        'blood_pressure': blood_pressure,
        'glucose_level': glucose,
        'target': target
    })
    return df

# ---------------------------
# Sidebar controls
# ---------------------------
st.sidebar.header("Controls & Data")
uploaded = st.sidebar.file_uploader("Upload CSV dataset (optional).", type=['csv'])
use_demo = st.sidebar.checkbox("Use demo dataset (if no upload)", value=True if uploaded is None else False)

st.sidebar.markdown("---")
st.sidebar.subheader("BAT Feature Selection (Adjust)")
num_bats = st.sidebar.slider("Number of bats", 8, 60, 24)
max_gen = st.sidebar.slider("Generations", 5, 100, 40)
loudness = st.sidebar.slider("Loudness (A)", 0.05, 1.0, 0.5, step=0.05)
pulse_rate = st.sidebar.slider("Pulse rate (r)", 0.0, 1.0, 0.5, step=0.05)
sleep_per_gen = st.sidebar.slider("UI speed (sec per generation)", 0.0, 0.05, 0.01, step=0.005)

st.sidebar.markdown("---")
st.sidebar.subheader("Classifier")
classifier_choice = st.sidebar.selectbox("Classifier (for now)", ["Naive Bayes (Gaussian)"])

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🔁 Run BAT & Train")
save_artifacts = st.sidebar.checkbox("Save trained model & metadata to ./models", value=False)

# ---------------------------
# Load dataset (uploaded or demo)
# ---------------------------
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()
elif use_demo:
    df_raw = generate_demo_df()
else:
    st.info("Please upload a CSV or enable demo dataset in the sidebar.")
    st.stop()

# dataset preview
with st.expander("Dataset preview (first 10 rows)"):
    st.dataframe(df_raw.head(10))

# choose target column
st.sidebar.markdown("### Choose target (label) column")
target_col = st.sidebar.selectbox("Target column", options=list(df_raw.columns), index=list(df_raw.columns).index('target') if 'target' in df_raw.columns else len(df_raw.columns)-1)

# ---------------------------
# Clean & Prepare
# ---------------------------
def clean_prepare(df, tgt_col):
    dfc = df.copy()
    dfc = dfc.dropna(how='all')
    # If target is non-numeric with M/B or malignant/benign convert to 1/0
    if dfc[tgt_col].dtype == object or str(dfc[tgt_col].dtype).startswith('category'):
        vals = [str(x).lower() for x in dfc[tgt_col].dropna().unique()]
        if any(v.startswith('m') for v in vals) and any(v.startswith('b') for v in vals):
            dfc[tgt_col] = dfc[tgt_col].map(lambda x: 1 if str(x).lower().startswith('m') else 0)
        else:
            try:
                dfc[tgt_col] = pd.to_numeric(dfc[tgt_col])
            except Exception:
                pass
    dfc = dfc.loc[dfc[tgt_col].notna()].reset_index(drop=True)
    y = dfc[tgt_col].astype(int)
    X = dfc.drop(columns=[tgt_col])
    # drop id-like columns
    for c in list(X.columns):
        if c.lower() in ("id", "patient_id", "pid"):
            X = X.drop(columns=[c])
    return X.reset_index(drop=True), y.reset_index(drop=True)

X_df, y_ser = clean_prepare(df_raw, target_col)

st.sidebar.markdown("---")
st.sidebar.write(f"Rows: {X_df.shape[0]}  |  Features: {X_df.shape[1]}  |  Target: {target_col}")

# show class distribution
st.subheader("Target class distribution")
counts = y_ser.value_counts().to_dict()
fig_pie = px.pie(values=list(counts.values()), names=list(counts.keys()), title="Class distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# show missing value summary
missing = X_df.isnull().sum()
if missing.sum() > 0:
    with st.expander("Missing values per column"):
        st.dataframe(missing[missing > 0])

# detect feature types
numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
st.markdown(f"**Detected** numeric: {len(numeric_cols)} — categorical: {len(cat_cols)}")

# ---------------------------
# Preprocessor builder
# ---------------------------
def build_preprocessor(X):
    num_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_local = X.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers = []
    if len(cat_cols_local) > 0:
        ohe = make_onehot_encoder_compat(handle_unknown='ignore')
        transformers.append(('cat', ohe, cat_cols_local))
    if len(num_cols) > 0:
        transformers.append(('num', StandardScaler(), num_cols))
    col_trans = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)
    return col_trans, num_cols, cat_cols_local

preproc, num_cols, cat_cols = build_preprocessor(X_df)

# placeholders / state
conv_placeholder = st.empty()
progress_place = st.sidebar.empty()
status_place = st.sidebar.empty()
metrics_place = st.empty()

trained_model = None
trained_preproc = None
selected_indices = []
transformed_feature_names = []
convergence_curve = []

# ---------------------------
# BAT + Train (when button pressed)
# ---------------------------
def ui_progress(gen_idx, best_score, conv_so_far, mx):
    frac = int((gen_idx + 1)/mx * 100)
    progress_place.progress(frac)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=conv_so_far, mode='lines+markers', name='best_cv'))
    fig.update_layout(title=f"BAT Convergence (gen {gen_idx+1}/{mx})", xaxis_title="Generation", yaxis_title="Best CV Score", height=350)
    conv_placeholder.plotly_chart(fig, use_container_width=True)
    status_place.info(f"BAT running: generation {gen_idx+1}/{mx}")

if run_btn:
    status_place.info("Starting preprocessing and BAT...")
    work = pd.concat([X_df, y_ser], axis=1).dropna().reset_index(drop=True)
    if work.shape[0] < 10:
        st.warning("After dropping missing rows there are less than 10 samples — results may be unreliable.")
    X_clean = work.drop(columns=[target_col])
    y_clean = work[target_col].astype(int)

    # rebuild preprocessor
    preproc, num_cols, cat_cols = build_preprocessor(X_clean)
    try:
        preproc.fit(X_clean)
    except Exception as e:
        st.error(f"Preprocessor fit failed: {e}")
        st.stop()

    # transformed names
    try:
        transformed_feature_names = list(preproc.get_feature_names_out())
    except Exception:
        # manual build
        transformed_feature_names = []
        if cat_cols:
            ohe = preproc.named_transformers_['cat']
            cats = ohe.categories_
            for col, levels in zip(cat_cols, cats):
                for lvl in levels:
                    transformed_feature_names.append(f"{col}__{lvl}")
        if num_cols:
            transformed_feature_names.extend(num_cols)

    # transform
    X_trans = preproc.transform(X_clean)
    if not isinstance(X_trans, np.ndarray):
        try:
            X_trans = X_trans.toarray()
        except Exception:
            X_trans = np.array(X_trans)

    y_vals = y_clean.values.astype(int)

    # run BAT
    try:
        sel_idx, conv = bat_feature_selection(
            features=X_trans,
            labels=y_vals,
            num_bats=num_bats,
            max_gen=max_gen,
            loudness=loudness,
            pulse_rate=pulse_rate,
            progress_updater=ui_progress,
            sleep_per_gen=sleep_per_gen
        )
    except Exception as e:
        st.error(f"BAT failed: {e}")
        st.stop()

    progress_place.empty()
    status_place.success("BAT finished.")

    if len(sel_idx) == 0:
        st.warning("BAT selected zero features — falling back to mutual information top-k.")
        mi = mutual_info_classif(X_trans, y_vals)
        topk = min(10, X_trans.shape[1])
        sel_idx = list(np.argsort(mi)[-topk:])
        conv = conv if len(conv) > 0 else [0.0]

    selected_indices = sel_idx
    convergence_curve = conv
    selected_names = [transformed_feature_names[i] for i in selected_indices] if transformed_feature_names else [f"f{i}" for i in selected_indices]
    st.success(f"Selected {len(selected_names)} features.")
    st.write(selected_names)

    # final convergence plot
    try:
        fig_fin = px.line(y=convergence_curve, labels={'value':'Best CV Score','index':'Generation'}, title="BAT Convergence (final)")
        fig_fin.update_traces(mode='lines+markers')
        conv_placeholder.plotly_chart(fig_fin, use_container_width=True)
    except Exception:
        pass

    # Train GaussianNB on selected
    try:
        X_sel = X_trans[:, selected_indices]
        X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_vals, test_size=0.20, random_state=42, stratify=y_vals)
        clf = GaussianNB()
        clf.fit(X_tr, y_tr)
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # metrics
    y_pred = clf.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    acc = float((y_pred == y_te).mean())
    metrics_place.metric("Test accuracy", f"{acc:.3f}")
    st.subheader("Model evaluation (test set)")
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))

    cm = confusion_matrix(y_te, y_pred)
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], colorscale="Blues"))
    cm_fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)

    try:
        y_prob = clf.predict_proba(X_te)[:,1]
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC={roc_auc:.3f}"))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Chance'))
        roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(roc_fig, use_container_width=True)
    except Exception:
        pass

    # feature importance (mutual info) on selected
    try:
        if X_sel.shape[1] > 0:
            mi = mutual_info_classif(X_sel, y_vals)
            mi_df = pd.DataFrame({'feature': selected_names, 'mi': mi}).sort_values('mi', ascending=False)
            st.subheader("Feature importance (Mutual Information on selected features)")
            st.dataframe(mi_df.style.background_gradient(cmap='Oranges'))
    except Exception:
        pass

    # Save artifacts if requested
    if save_artifacts:
        Path("models").mkdir(parents=True, exist_ok=True)
        dump(clf, "models/model.joblib")
        dump(preproc, "models/preprocessor.joblib")
        np.save("models/selected_indices.npy", np.array(selected_indices, dtype=int))
        with open("models/feature_names.json", "w") as f:
            json.dump(transformed_feature_names, f)
        np.save("models/convergence.npy", np.array(convergence_curve, dtype=float))
        metadata = {
            "original_columns": X_df.columns.tolist(),
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "numeric_stats": {c: {'min': float(X_df[c].min()), 'max': float(X_df[c].max()), 'mean': float(X_df[c].mean())} for c in num_cols}
        }
        with open("models/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        st.sidebar.success("Saved to ./models/")

    # store trained artifacts in session state (for immediate prediction)
    st.session_state['clf'] = clf
    st.session_state['preproc'] = preproc
    st.session_state['selected_indices'] = selected_indices
    st.session_state['transformed_feature_names'] = transformed_feature_names
    st.session_state['convergence_curve'] = convergence_curve

# ---------------------------
# Patient form & prediction
# ---------------------------
st.markdown("---")
st.header("🧾 Patient Prediction Form")

# Determine model and preprocessor availability
if 'clf' in st.session_state:
    model_for_pred = st.session_state['clf']
    preproc_for_pred = st.session_state['preproc']
    sel_idx_for_pred = st.session_state['selected_indices']
    trans_feat_names_local = st.session_state.get('transformed_feature_names', [])
else:
    # try to load saved artifacts if available
    model_for_pred = None
    preproc_for_pred = None
    sel_idx_for_pred = []
    if Path("models/model.joblib").exists() and Path("models/preprocessor.joblib").exists() and Path("models/selected_indices.npy").exists():
        try:
            model_for_pred = load("models/model.joblib")
            preproc_for_pred = load("models/preprocessor.joblib")
            sel_idx_for_pred = list(np.load("models/selected_indices.npy").astype(int))
            with open("models/feature_names.json", "r") as f:
                trans_feat_names_local = json.load(f)
            st.sidebar.info("Loaded saved model from ./models/")
        except Exception:
            model_for_pred = None
            preproc_for_pred = None
            sel_idx_for_pred = []

if model_for_pred is None:
    st.info("Model not available yet. Train using 'Run BAT & Train' in the sidebar or save artifacts to ./models and reload.")
else:
    # Build patient form using original feature columns (X_df)
    with st.form("patient_form"):
        user_inputs = {}
        for col in X_df.columns:
            if col in X_df.select_dtypes(include=[np.number]).columns:
                vmin = float(X_df[col].min())
                vmax = float(X_df[col].max())
                vmean = float(X_df[col].mean())
                user_inputs[col] = st.number_input(col, min_value=vmin, max_value=vmax, value=vmean, format="%.5f")
            else:
                uniques = X_df[col].dropna().unique().tolist()
                if len(uniques) <= 12:
                    user_inputs[col] = st.selectbox(col, options=uniques, index=0)
                else:
                    user_inputs[col] = st.text_input(col, value=str(uniques[0]))
        submitted = st.form_submit_button("Predict Patient")

    if submitted:
        sample_df = pd.DataFrame([user_inputs])[X_df.columns]
        try:
            sample_trans = preproc_for_pred.transform(sample_df)
            if not isinstance(sample_trans, np.ndarray):
                try:
                    sample_trans = sample_trans.toarray()
                except Exception:
                    sample_trans = np.array(sample_trans)
        except Exception as e:
            st.error(f"Preprocessing failed: {e}")
            sample_trans = None

        if sample_trans is not None:
            try:
                sample_sel = sample_trans[:, sel_idx_for_pred]
            except Exception as e:
                st.error(f"Selecting trained features failed: {e}")
                sample_sel = None

            if sample_sel is not None:
                try:
                    pred = model_for_pred.predict(sample_sel)[0]
                    prob = model_for_pred.predict_proba(sample_sel)[0][int(pred)] if hasattr(model_for_pred, "predict_proba") else None
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred = None
                    prob = None

                if pred is not None:
                    label = "Positive (Malignant)" if int(pred) == 1 else "Negative (Benign)"
                    if int(pred) == 1:
                        st.error(f"🔴 {label}")
                    else:
                        st.success(f"🟢 {label}")
                    if prob is not None:
                        st.metric("Model confidence", f"{prob:.2%}")
                    st.markdown("### Explanation & Next Steps")
                    st.write(explain_prediction_text(prob if prob is not None else 0, int(pred)))
                    outdf = sample_df.copy()
                    outdf['prediction'] = int(pred)
                    outdf['confidence'] = float(prob) if prob is not None else None
                    st.subheader("Patient input summary")
                    st.dataframe(outdf.T)
                    st.download_button("Download patient result (CSV)", outdf.to_csv(index=False).encode('utf-8'), "patient_result.csv", "text/csv")

# ---------------------------
# Additional analysis charts & write-up
# ---------------------------
st.markdown("---")
st.header("Charts, Analysis & Write-up")

# Correlation heatmap (numeric)
if len(numeric_cols) > 1:
    st.subheader("Correlation heatmap (numeric features)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(X_df[numeric_cols].corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Top selected features display if available
if 'selected_indices' in locals() and len(selected_indices) > 0:
    st.subheader("Top selected transformed features (sample)")
    try:
        names = [transformed_feature_names[i] for i in selected_indices]
        st.write(names[:30])
    except Exception:
        pass

# Simple distributions for demo fields
st.subheader("Feature distributions")
cols_to_plot = (numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols)
for col in cols_to_plot:
    fig = px.histogram(X_df, x=col, color=y_ser, nbins=30, title=f"Distribution: {col} by target", marginal="box")
    st.plotly_chart(fig, use_container_width=True)

# Final write-up
st.markdown("""
## Methodology & Interpretation (Write-up)

**Objective:** Build an interactive app that allows uploading clinical/demographic datasets and training a classifier that uses
a bio-inspired **BAT** algorithm to select features and **Gaussian Naive Bayes** for classification.

**Preprocessing:** Categorical variables are One-Hot encoded (compatible across sklearn versions); numeric features standardized.

**Feature selection (BAT):** BAT (binary variant) searches the transformed feature space and optimizes 5-fold cross-validated accuracy.
The convergence plot shows how best-found accuracy evolves over generations.

**Prediction:** After training, the patient form accepts human-friendly inputs (age, sex, tumor size, BMI, etc.), which are preprocessed
exactly the same way as the training data. The model returns a probabilistic prediction and an explanation with next steps.

**Limitations:** This is an educational/prototyping tool. Not a clinical diagnostic device. Model performance depends heavily on dataset quality and representativeness.
""")

# ---------------------------
# End
# ---------------------------
