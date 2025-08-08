# app.py
"""
Breast Cancer Risk Predictor — medical patient form + BAT feature selection + GaussianNB
Single-file Streamlit app with:
- dataset upload or built-in synthetic dataset
- target selection and column mapping UI
- BAT parameters in sidebar + class balancing options
- training (BAT -> selected features -> GaussianNB)
- charts: BAT convergence, ROC, confusion matrix, class distribution, correlation heatmap, feature importance
- hard-coded patient form fields that map to dataset columns (mapping editable)
- step-by-step written explanations for training & predictions
- added: explicit display of accuracy, precision, F1 and explanatory text for each chart/metric
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from joblib import dump, load

# plotting libs
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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc, precision_score, f1_score, accuracy_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.utils import resample

st.set_page_config(page_title="Breast Cancer Risk — Medical Form", layout="wide")
st.title("🧬 Breast Cancer Risk Predictor — Medical Form (BAT + Naive Bayes)")

# -----------------------
# Helpers
# -----------------------
def make_onehot_encoder_compat(**kwargs):
    """Return a OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)

def explain_prediction_text(prob, label):
    """Return plain-language explanation for a prediction."""
    pct = f"{prob*100:.1f}%" if prob is not None else "N/A"
    if label == 1:
        return (
            f"**Prediction:** POSITIVE (likely malignant). Confidence: **{pct}**.\n\n"
            "Interpretation: The model found the entered features more similar to malignant cases in the training data.\n\n"
            "Recommended next steps:\n"
            "1. Urgent clinical follow-up and specialist consult.\n"
            "2. Diagnostic imaging (mammogram/ultrasound/MRI) and biopsy if indicated.\n"
            "3. Treat this as a screening aid only — not a definitive diagnosis."
        )
    else:
        return (
            f"**Prediction:** NEGATIVE (likely benign). Confidence: **{pct}**.\n\n"
            "Interpretation: The model found the entered features more similar to benign cases in the training data.\n"
            "Recommended next steps: routine surveillance and clinical follow-up if symptoms persist."
        )

def format_pct(x):
    try:
        return f"{x*100:.2f}%"
    except Exception:
        return "N/A"

# -----------------------
# Synthetic builtin dataset (balanced by default)
# -----------------------
@st.cache_data
def generate_balanced_synthetic(n=500, seed=42):
    rng = np.random.default_rng(seed)
    Age = rng.integers(20, 80, n)
    Sex = rng.choice(['Female', 'Male'], n, p=[0.92, 0.08])
    BMI = np.round(rng.normal(26.5, 5.0, n).clip(16, 45), 1)
    Family_History = rng.choice(['Yes', 'No'], n, p=[0.2, 0.8])
    Smoking_Status = rng.choice(['Never', 'Former', 'Current'], n, p=[0.6, 0.25, 0.15])
    Alcohol_Intake = rng.choice(['Low','Moderate','High'], n, p=[0.5,0.35,0.15])
    Physical_Activity = rng.choice(['Low','Moderate','High'], n, p=[0.3,0.5,0.2])
    Hormone_Therapy = rng.choice(['Yes','No'], n, p=[0.12,0.88])
    Breastfeeding_History = rng.choice(['Yes','No'], n, p=[0.6,0.4])
    Pregnancies = rng.integers(0,6,n)

    # Toy risk score to generate labels, then balance classes
    score = (
        0.04 * (Age - 40)
        + 0.6 * (BMI - 25) / 5
        + 1.2 * (Family_History == 'Yes').astype(int)
        + 0.9 * (Hormone_Therapy == 'Yes').astype(int)
        + 0.8 * (Smoking_Status == 'Current').astype(int)
        - 0.4 * (Physical_Activity == 'High').astype(int)
        + 1.5 * ((Pregnancies == 0).astype(int))
    )
    prob = 1 / (1 + np.exp(-score))
    y = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'Age': Age,
        'Sex': Sex,
        'BMI': BMI,
        'Family_History': Family_History,
        'Smoking_Status': Smoking_Status,
        'Alcohol_Intake': Alcohol_Intake,
        'Physical_Activity': Physical_Activity,
        'Hormone_Therapy': Hormone_Therapy,
        'Breastfeeding_History': Breastfeeding_History,
        'Pregnancies': Pregnancies,
        'Target': y
    })

    # balance classes by resampling to equal counts
    pos = df[df['Target'] == 1]
    neg = df[df['Target'] == 0]
    m = max(len(pos), len(neg))
    pos_up = resample(pos, replace=True, n_samples=m, random_state=seed)
    neg_up = resample(neg, replace=True, n_samples=m, random_state=seed+1)
    df_bal = pd.concat([pos_up, neg_up]).sample(frac=1, random_state=seed).reset_index(drop=True)
    return df_bal

# -----------------------
# BAT algorithm (binary)
# -----------------------
def objective_cv_score(features, labels, mask):
    mask = np.array(mask, dtype=bool)
    if mask.sum() == 0:
        return 0.0
    X_sel = features[:, mask]
    clf = GaussianNB()
    try:
        scores = cross_val_score(clf, X_sel, labels, cv=5, error_score='raise')
        return float(np.mean(scores))
    except Exception:
        return 0.0

def bat_feature_selection(features, labels, num_bats=20, max_gen=40, loudness=0.5, pulse_rate=0.5, progress_cb=None, sleep_per_gen=0.0):
    n_feats = features.shape[1]
    positions = (np.random.rand(num_bats, n_feats) > 0.5).astype(int)
    velocities = np.zeros((num_bats, n_feats))
    freq_min, freq_max = 0.0, 2.0

    fitness = np.array([objective_cv_score(features, labels, positions[i]) for i in range(num_bats)])
    best_idx = int(np.argmax(fitness))
    best_pos = positions[best_idx].copy()
    best_score = float(fitness[best_idx])
    convergence = [best_score]

    for gen in range(max_gen):
        for i in range(num_bats):
            freq = freq_min + (freq_max - freq_min) * np.random.rand()
            velocities[i] = velocities[i] + freq * (positions[i] ^ best_pos)
            prob = 1.0 / (1.0 + np.exp(-velocities[i]))
            new_pos = (np.random.rand(n_feats) < prob).astype(int)

            if np.random.rand() > pulse_rate:
                flip_mask = (np.random.rand(n_feats) < 0.05).astype(int)
                tmp = best_pos.copy()
                tmp[flip_mask == 1] = 1 - tmp[flip_mask == 1]
                new_pos = tmp

            new_score = objective_cv_score(features, labels, new_pos)
            if (new_score > fitness[i]) and (np.random.rand() < loudness):
                positions[i] = new_pos
                fitness[i] = new_score
            if new_score > best_score:
                best_pos = new_pos.copy()
                best_score = new_score
        convergence.append(best_score)
        if progress_cb is not None:
            try:
                progress_cb(gen, best_score, convergence.copy(), max_gen)
            except Exception:
                pass
        if sleep_per_gen > 0:
            time.sleep(sleep_per_gen)

    selected_indices = [int(i) for i, v in enumerate(best_pos) if v == 1]
    return selected_indices, convergence

# -----------------------
# Sidebar UI
# -----------------------
st.sidebar.header("Controls & Data")
uploaded = st.sidebar.file_uploader("Upload CSV dataset (optional)", type=["csv"])
use_builtin = st.sidebar.checkbox("Use built-in balanced synthetic dataset", value=True if uploaded is None else False)

st.sidebar.markdown("---")
st.sidebar.subheader("BAT parameters")
num_bats = st.sidebar.slider("Number of bats", 6, 80, 24)
max_gen = st.sidebar.slider("Max generations", 5, 100, 40)
loudness = st.sidebar.slider("Loudness (A)", 0.05, 1.0, 0.5, step=0.05)
pulse_rate = st.sidebar.slider("Pulse rate (r)", 0.0, 1.0, 0.5, step=0.05)
sleep_per_gen = st.sidebar.slider("UI speed (sec/gen)", 0.0, 0.05, 0.01, step=0.005)

st.sidebar.markdown("---")
st.sidebar.subheader("Classifier")
classifier_choice = st.sidebar.selectbox("Classifier (only Gaussian NB implemented)", ["Naive Bayes (Gaussian)"])

st.sidebar.markdown("---")
st.sidebar.subheader("Class balancing (optional)")
balance_option = st.sidebar.selectbox("Balance training data by", ["None", "Oversample minority", "Undersample majority"])

st.sidebar.markdown("---")
run_train = st.sidebar.button("🔁 Run BAT & Train")
save_artifacts = st.sidebar.checkbox("Save artifacts to ./models", value=False)

# -----------------------
# Load dataset safely (fixes 'object has no attribute head')
# -----------------------
def load_uploaded(uploaded_obj):
    if uploaded_obj is None:
        return None
    # uploaded_obj can be UploadedFile, path string, or bytes
    try:
        # If it's a Streamlit UploadedFile
        df = pd.read_csv(uploaded_obj)
        return df
    except Exception:
        # try to decode bytes or read by path
        try:
            if isinstance(uploaded_obj, (str, Path)):
                return pd.read_csv(str(uploaded_obj))
            # bytes-like
            content = uploaded_obj.read()
            return pd.read_csv(pd.io.common.StringIO(content.decode('utf-8')))
        except Exception as e:
            st.error(f"Could not parse uploaded file: {e}")
            return None

if uploaded is not None:
    df_raw = load_uploaded(uploaded)
    if df_raw is None:
        st.stop()
elif use_builtin:
    df_raw = generate_balanced_synthetic(500)
else:
    st.info("Please upload a CSV or enable the built-in dataset.")
    st.stop()

# Preview dataset
with st.expander("Dataset preview (first 8 rows)"):
    st.dataframe(df_raw.head(8))

# -----------------------
# Target selection & mapping UI
# -----------------------
st.sidebar.markdown("---")
st.sidebar.subheader("Target & mapping")

# Ensure df_raw is a DataFrame before listing columns
if not isinstance(df_raw, pd.DataFrame):
    st.error("Loaded data is not a table. Make sure you uploaded a CSV file.")
    st.stop()

columns = list(df_raw.columns)
tgt_col = st.sidebar.selectbox("Select target (label) column", options=columns, index=columns.index("Target") if "Target" in columns else len(columns)-1)
st.sidebar.write(f"Using target column: **{tgt_col}**")

# Patient form fields we want available (hard-coded)
medical_fields = [
    'Age','Sex','BMI','Family_History','Smoking_Status','Alcohol_Intake',
    'Physical_Activity','Hormone_Therapy','Breastfeeding_History','Pregnancies'
]

st.sidebar.markdown("### Map patient-form fields to dataset columns")
mapping = {}
for f in medical_fields:
    # default mapping: if exact column exists, choose it; else choose "Not present"
    default = f if f in columns else "Not present"
    options = ["Not present"] + columns
    mapping[f] = st.sidebar.selectbox(f"{f} ->", options=options, index=options.index(default) if default in options else 0)

# Auto-map helper button
if st.sidebar.button("Auto-map by name similarity"):
    for f in medical_fields:
        # try exact match ignoring case/underscores
        found = None
        key = f.lower().replace("_","")
        for c in columns:
            if c.lower().replace("_","") == key:
                found = c; break
        if found:
            mapping[f] = found
        else:
            # try partial matches
            for c in columns:
                if key in c.lower().replace("_","") or c.lower().replace("_","") in key:
                    found = c; break
            if found:
                mapping[f] = found

# -----------------------
# Prepare X, y for training
# -----------------------
def clean_and_prepare(df, target_col):
    dfc = df.copy()
    dfc = dfc.dropna(how='all')
    # attempt to convert target to numeric 0/1
    if dfc[target_col].dtype == object or str(dfc[target_col].dtype).startswith('category'):
        vals = [str(x).lower() for x in dfc[target_col].dropna().unique()]
        if any(v.startswith('m') for v in vals) and any(v.startswith('b') for v in vals):
            dfc[target_col] = dfc[target_col].map(lambda x: 1 if str(x).lower().startswith('m') else 0)
        else:
            try:
                dfc[target_col] = pd.to_numeric(dfc[target_col])
            except Exception:
                pass
    dfc = dfc.loc[dfc[target_col].notna()].reset_index(drop=True)
    y = dfc[target_col].astype(int)
    X = dfc.drop(columns=[target_col])
    # drop obviously ID-like columns
    for c in list(X.columns):
        if c.lower() in ('id','patient_id','pid'):
            X.drop(columns=[c], inplace=True)
    return X.reset_index(drop=True), y.reset_index(drop=True)

X_df, y_ser = clean_and_prepare(df_raw, tgt_col)

st.subheader("Dataset summary")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{X_df.shape[0]}")
c2.metric("Features", f"{X_df.shape[1]}")
c3.metric("Target", f"{tgt_col}")

# Missing values display
missing = X_df.isna().sum()
if missing.sum() > 0:
    with st.expander("Missing values per column"):
        st.dataframe(missing[missing > 0])

# Class distribution
st.subheader("Class distribution")
counts = y_ser.value_counts().to_dict()
fig_pie = px.pie(values=list(counts.values()), names=list(counts.keys()), title="Class distribution")
st.plotly_chart(fig_pie, use_container_width=True)
st.markdown(
    "**Explanation — Class distribution**  \n"
    "This pie chart shows the proportion of samples labeled with each target class in the dataset. "
    "Counts are calculated by `y_ser.value_counts()`. The model's baseline expectations depend on these class ratios — "
    "if one class dominates, simple accuracy can be misleading and balancing strategies may be used (see 'Class balancing' in the sidebar)."
)

# -----------------------
# Preprocessor builder
# -----------------------
def build_preprocessor(Xdf):
    num_cols = Xdf.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = Xdf.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers = []
    if len(cat_cols) > 0:
        ohe = make_onehot_encoder_compat(handle_unknown='ignore')
        transformers.append(('cat', ohe, cat_cols))
    if len(num_cols) > 0:
        transformers.append(('num', StandardScaler(), num_cols))
    ct = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)
    return ct, num_cols, cat_cols

preproc, num_cols, cat_cols = build_preprocessor(X_df)

# placeholders for UI updates
conv_placeholder = st.empty()
progress_placeholder = st.sidebar.empty()
status_placeholder = st.sidebar.empty()
metrics_placeholder = st.empty()

# trained artifacts (session)
if 'clf' in st.session_state:
    trained_clf = st.session_state['clf']
else:
    trained_clf = None

if 'preproc' in st.session_state:
    trained_preproc = st.session_state['preproc']
else:
    trained_preproc = None

selected_indices = st.session_state.get('selected_indices', [])
transformed_feature_names = st.session_state.get('transformed_feature_names', [])
convergence_curve = st.session_state.get('convergence_curve', [])

# -----------------------
# Training: Run BAT & Train
# -----------------------
def ui_progress(gen_idx, best_score, conv_so_far, max_gen_local):
    frac = int((gen_idx+1)/max_gen_local * 100)
    progress_placeholder.progress(frac)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=conv_so_far, mode='lines+markers', name='best_cv'))
    fig.update_layout(title=f"BAT convergence (gen {gen_idx+1}/{max_gen_local})", xaxis_title="Generation", yaxis_title="Best CV score", height=380)
    conv_placeholder.plotly_chart(fig, use_container_width=True)
    status_placeholder.info(f"BAT running: generation {gen_idx+1}/{max_gen_local}")

if run_train:
    status_placeholder.info("Preparing data for training...")
    # optionally balance dataset
    X_df_train = X_df.copy()
    y_train_ser = y_ser.copy()
    df_work = pd.concat([X_df_train, y_train_ser], axis=1)
    df_work = df_work.dropna().reset_index(drop=True)

    if balance_option != "None":
        # oversample minority
        positives = df_work[df_work[tgt_col] == 1]
        negatives = df_work[df_work[tgt_col] == 0]
        if len(positives) == 0 or len(negatives) == 0:
            st.warning("Cannot balance because one class is missing.")
        else:
            if balance_option == "Oversample minority":
                if len(positives) < len(negatives):
                    positives = resample(positives, replace=True, n_samples=len(negatives), random_state=42)
                else:
                    negatives = resample(negatives, replace=True, n_samples=len(positives), random_state=42)
            elif balance_option == "Undersample majority":
                m = min(len(positives), len(negatives))
                positives = resample(positives, replace=False, n_samples=m, random_state=42)
                negatives = resample(negatives, replace=False, n_samples=m, random_state=42)
            df_work = pd.concat([positives, negatives]).sample(frac=1, random_state=42).reset_index(drop=True)

    X_clean = df_work.drop(columns=[tgt_col])
    y_clean = df_work[tgt_col].astype(int).values

    # rebuild preprocessor for cleaned X
    preproc, num_cols, cat_cols = build_preprocessor(X_clean)
    try:
        preproc.fit(X_clean)
    except Exception as e:
        st.error(f"Preprocessor fit failed: {e}")
        st.stop()

    # get transformed feature names safely
    try:
        transformed_feature_names_local = list(preproc.get_feature_names_out())
    except Exception:
        transformed_feature_names_local = []
        if len(cat_cols) > 0:
            ohe = preproc.named_transformers_['cat']
            cats = ohe.categories_
            for col, levels in zip(cat_cols, cats):
                for lvl in levels:
                    transformed_feature_names_local.append(f"{col}__{lvl}")
        transformed_feature_names_local.extend(num_cols)

    # transform
    X_trans = preproc.transform(X_clean)
    if not isinstance(X_trans, np.ndarray):
        try:
            X_trans = X_trans.toarray()
        except Exception:
            X_trans = np.array(X_trans)

    # run BAT
    try:
        sel_idx, conv = bat_feature_selection(
            features=X_trans,
            labels=y_clean,
            num_bats=num_bats,
            max_gen=max_gen,
            loudness=loudness,
            pulse_rate=pulse_rate,
            progress_cb=ui_progress,
            sleep_per_gen=sleep_per_gen
        )
    except Exception as e:
        st.error(f"BAT failed: {e}")
        st.stop()

    progress_placeholder.empty()
    status_placeholder.success("BAT finished.")

    if len(sel_idx) == 0:
        st.warning("BAT selected zero features; falling back to mutual-info top features.")
        mi = mutual_info_classif(X_trans, y_clean)
        topk = min(10, X_trans.shape[1])
        sel_idx = list(np.argsort(mi)[-topk:])
        conv = conv if len(conv) > 0 else [0.0]

    selected_indices = sel_idx
    convergence_curve = conv
    transformed_feature_names = transformed_feature_names_local

    selected_names = [transformed_feature_names[i] for i in selected_indices] if transformed_feature_names else [f"f{i}" for i in selected_indices]
    st.success(f"Selected {len(selected_names)} features.")
    st.write(selected_names[:60])

    # final convergence
    try:
        fig_fin = px.line(y=convergence_curve, labels={'value':'Best CV score','index':'Generation'}, title="BAT Convergence (final)")
        fig_fin.update_traces(mode='lines+markers')
        conv_placeholder.plotly_chart(fig_fin, use_container_width=True)
    except Exception:
        pass

    # train GaussianNB
    try:
        X_sel = X_trans[:, selected_indices]
        X_tr, X_te, y_tr, y_te = train_test_split(X_sel, y_clean, test_size=0.2, random_state=42, stratify=y_clean)
        clf = GaussianNB()
        clf.fit(X_tr, y_tr)
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # evaluate
    y_pred = clf.predict(X_te)
    report = classification_report(y_te, y_pred, output_dict=True, zero_division=0)
    acc = float(accuracy_score(y_te, y_pred))
    prec = float(precision_score(y_te, y_pred, zero_division=0))
    f1 = float(f1_score(y_te, y_pred, zero_division=0))

    # display metrics prominently
    with metrics_placeholder.container():
        st.subheader("Model performance (hold-out test)")
        m1, m2, m3 = st.columns(3)
        m1.metric("Accuracy", f"{acc:.3f}", delta=None)
        m2.metric("Precision", f"{prec:.3f}")
        m3.metric("F1 score", f"{f1:.3f}")
        st.markdown(
            "**How these metrics are calculated:**  \n"
            "- **Accuracy** = (correct predictions) / (total predictions). Computed with `accuracy_score(y_true, y_pred)`.  \n"
            "- **Precision** = true positives / (true positives + false positives). Computed with `precision_score(y_true, y_pred)`. "
            "It answers: *Of the samples predicted positive, how many actually were positive?*  \n"
            "- **F1 score** = harmonic mean of precision and recall. Computed with `f1_score(y_true, y_pred)`. "
            "It balances precision and recall and is useful with imbalanced classes."
        )
        st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))

    # confusion matrix
    cm = confusion_matrix(y_te, y_pred)
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], colorscale="Blues"))
    cm_fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)
    st.markdown(
        "**Explanation — Confusion matrix**  \n"
        "Rows are the TRUE classes and columns are the PREDICTED classes. "
        "`confusion_matrix(y_true, y_pred)` returns a 2×2 array for binary problems:  \n"
        "- Top-left: True Negatives (TN)  \n"
        "- Top-right: False Positives (FP)  \n"
        "- Bottom-left: False Negatives (FN)  \n"
        "- Bottom-right: True Positives (TP)  \n"
        "Precision = TP / (TP + FP). Recall = TP / (TP + FN). Accuracy = (TP + TN) / total."
    )

    # ROC
    try:
        y_prob = clf.predict_proba(X_te)[:,1]
        fpr, tpr, _ = roc_curve(y_te, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC={roc_auc:.3f}"))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Chance'))
        roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(roc_fig, use_container_width=True)
        st.markdown(
            "**Explanation — ROC & AUC**  \n"
            "ROC (Receiver Operating Characteristic) plots True Positive Rate (Recall) vs False Positive Rate at different probability thresholds. "
            "`roc_curve(y_true, y_score)` computes TPR and FPR at many thresholds.  \n"
            "AUC (Area Under Curve) summarizes the ROC curve: 1.0 is perfect, 0.5 is random. Computed with `auc(fpr, tpr)`."
        )
    except Exception:
        pass

    # feature importance (mutual info) on selected
    try:
        if X_sel.shape[1] > 0:
            mi = mutual_info_classif(X_sel, y_clean)
            mi_df = pd.DataFrame({'feature': selected_names, 'mi': mi}).sort_values('mi', ascending=False)
            st.subheader("Feature importance (Mutual Information on selected features)")
            st.dataframe(mi_df.style.background_gradient(cmap='Oranges'))
            st.markdown(
                "**Explanation — Feature importance (Mutual Information)**  \n"
                "Mutual Information (MI) measures how much information about the target is contained in a feature. "
                "`mutual_info_classif(X, y)` returns a non-negative score per feature; higher means the feature is more informative about the label. "
                "MI does not indicate direction (positive/negative) — it only measures dependence."
            )
    except Exception:
        pass

    # save artifacts
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
        st.sidebar.success("Saved artifacts to ./models/")

    # store trained artifacts in session
    st.session_state['clf'] = clf
    st.session_state['preproc'] = preproc
    st.session_state['selected_indices'] = selected_indices
    st.session_state['transformed_feature_names'] = transformed_feature_names
    st.session_state['convergence_curve'] = convergence_curve

# -----------------------
# Patient form and prediction
# -----------------------
st.markdown("---")
st.header("🧾 Patient Form — medical inputs")

# Build patient form with hard-coded fields; defaults come from dataset where possible
def default_for(col, kind):
    if col in X_df.columns:
        if kind == 'numeric':
            return float(X_df[col].mean())
        else:
            vals = X_df[col].dropna().unique().tolist()
            return vals[0] if vals else None
    else:
        return 40.0 if kind == 'numeric' else None

with st.form("patient_form"):
    patient_inputs = {}
    for f in medical_fields:
        mapped_col = mapping.get(f, "Not present")
        kind = 'numeric' if f in ['Age','BMI','Pregnancies'] else 'categorical'
        # if the mapped column exists in dataset and has sensible stats, use those for min/max
        if kind == 'numeric':
            vmin = float(X_df[mapped_col].min()) if (mapped_col in X_df.columns and pd.api.types.is_numeric_dtype(X_df[mapped_col])) else 0.0
            vmax = float(X_df[mapped_col].max()) if (mapped_col in X_df.columns and pd.api.types.is_numeric_dtype(X_df[mapped_col])) else 100.0
            default = default_for(mapped_col, 'numeric') if mapped_col in X_df.columns else (default_for(f,'numeric'))
            patient_inputs[f] = st.number_input(f.replace('_',' '), value=default, min_value=vmin, max_value=vmax, format="%.2f")
        else:
            opts = []
            if mapped_col in X_df.columns:
                opts = list(X_df[mapped_col].dropna().unique())
            if not opts:
                # fallback to a sensible list
                if f == 'Sex':
                    opts = ['Female','Male']
                elif f == 'Family_History':
                    opts = ['Yes','No']
                elif f == 'Smoking_Status':
                    opts = ['Never','Former','Current']
                elif f == 'Alcohol_Intake' or f == 'Physical_Activity':
                    opts = ['Low','Moderate','High']
                else:
                    opts = ['Yes','No']
            default = default_for(mapped_col,'categorical') if mapped_col in X_df.columns else opts[0]
            patient_inputs[f] = st.selectbox(f.replace('_',' '), options=opts, index=opts.index(default) if default in opts else 0)
    submit_patient = st.form_submit_button("Predict for this patient")

# Build a one-row DataFrame matching dataset original columns using mapping
def build_sample_from_patient(patient_dict, mapping_dict, Xdf):
    row = {}
    for c in Xdf.columns:
        # find which patient field maps to this column
        mapping_keys = [k for k,v in mapping_dict.items() if v == c]
        if mapping_keys:
            # take the first mapping
            val = patient_dict.get(mapping_keys[0], None)
            row[c] = val
        else:
            # No mapping: use default (mean for numeric, mode for categorical)
            if pd.api.types.is_numeric_dtype(Xdf[c]):
                row[c] = float(Xdf[c].mean())
            else:
                vals = Xdf[c].dropna().unique().tolist()
                row[c] = vals[0] if vals else ""
    return pd.DataFrame([row], columns=Xdf.columns)

if submit_patient:
    # make sure model & preproc are available
    if 'clf' in st.session_state and 'preproc' in st.session_state and st.session_state.get('selected_indices', None) is not None:
        model = st.session_state['clf']
        preproc_model = st.session_state['preproc']
        sel_idx = st.session_state['selected_indices']
        try:
            sample_df = build_sample_from_patient(patient_inputs, mapping, X_df)
            sample_trans = preproc_model.transform(sample_df)
            if not isinstance(sample_trans, np.ndarray):
                sample_trans = sample_trans.toarray()
            sample_sel = sample_trans[:, sel_idx]
            pred = model.predict(sample_sel)[0]
            prob = float(model.predict_proba(sample_sel)[0][int(pred)]) if hasattr(model, "predict_proba") else None
        except Exception as e:
            st.error(f"Preprocessing or prediction failed: {e}")
            pred = None
            prob = None

        if pred is not None:
            if int(pred) == 1:
                st.error("🔴 Prediction: Positive (Malignant)")
            else:
                st.success("🟢 Prediction: Negative (Benign)")
            if prob is not None:
                st.metric("Model confidence", f"{prob:.2%}")

            # Show explanation, and show which mapped features were most important
            st.markdown("### Explanation & recommended next steps")
            st.write(explain_prediction_text(prob if prob is not None else 0, int(pred)))

            # show patient input summary and allow download
            outdf = sample_df.copy()
            outdf['prediction'] = int(pred)
            outdf['confidence'] = float(prob) if prob is not None else None
            st.subheader("Patient input summary (mapped to dataset columns)")
            st.dataframe(outdf.T)
            st.download_button("Download patient report (CSV)", outdf.to_csv(index=False).encode('utf-8'), "patient_report.csv", "text/csv")

            # Show which transformed features matched the selected ones and top mutual-info (if available)
            try:
                # map selected indices back to transformed names if available
                feat_names = st.session_state.get('transformed_feature_names', transformed_feature_names)
                if feat_names and len(sel_idx) > 0:
                    chosen = [feat_names[i] for i in sel_idx]
                    st.subheader("Model used transformed features (sample)")
                    st.write(chosen[:40])
            except Exception:
                pass
    else:
        st.warning("No trained model found. Train first with 'Run BAT & Train' in the sidebar or load saved artifacts from ./models.")

# -----------------------
# Charts & Analysis (always available)
# -----------------------
st.markdown("---")
st.header("Charts, Analysis & Step-by-step Write-up")

# Correlation heatmap
numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
if len(numeric_cols) > 1:
    st.subheader("Correlation heatmap (numeric features)")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(X_df[numeric_cols].corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)
    st.markdown(
        "**Explanation — Correlation heatmap**  \n"
        "This heatmap shows Pearson correlation coefficients between numeric features, computed by `X_df[numeric_cols].corr()`. "
        "Values range from -1 (perfect negative correlation) to +1 (perfect positive correlation). "
        "High absolute correlations indicate strong linear relationships; correlated features can lead to redundant information."
    )

# Distributions of numeric features (first 6)
st.subheader("Numeric feature distributions (by target)")
plot_cols = numeric_cols[:6]
for col in plot_cols:
    fig = px.histogram(X_df, x=col, color=y_ser, nbins=30, title=f"{col} distribution by target", marginal="box")
    st.plotly_chart(fig, use_container_width=True)
    st.markdown(
        f"**Explanation — {col} distribution by target**  \n"
        f"This histogram shows how the numeric feature **{col}** is distributed across the dataset, split by the target label. "
        "Counts were computed per bin for each label. The small box plot (if shown) displays median and quartiles. "
        "Look for shifts in the distributions between classes — those indicate potential predictive power."
    )

# Selected features & importance if available
if st.session_state.get('selected_indices'):
    st.subheader("Selected features & importance")
    try:
        feat_names = st.session_state.get('transformed_feature_names', transformed_feature_names)
        sel_idx = st.session_state['selected_indices']
        chosen = [feat_names[i] for i in sel_idx]
        st.write(f"Selected transformed features (showing up to 50):")
        st.write(chosen[:50])

        # compute mutual info on training set if we have X_trans and y
        # (safe attempt: transform whole X_df with trained_preproc if present)
        if 'preproc' in st.session_state:
            try:
                all_trans = st.session_state['preproc'].transform(X_df)
                if not isinstance(all_trans, np.ndarray):
                    all_trans = all_trans.toarray()
                X_sel_all = all_trans[:, sel_idx]
                mi = mutual_info_classif(X_sel_all, y_ser.values)
                mi_df = pd.DataFrame({'feature': chosen, 'mi': mi}).sort_values('mi', ascending=False)
                st.dataframe(mi_df.style.background_gradient(cmap='Oranges'))
                st.markdown(
                    "**Explanation — Selected features & MI on dataset**  \n"
                    "This shows the features chosen by BAT (on transformed feature space). Mutual Information (MI) was computed with `mutual_info_classif` "
                    "on the transformed features across the whole dataset to provide an estimate of how informative each selected feature is about the label. "
                    "MI is non-directional and measures dependency, not causation."
                )
            except Exception:
                pass
    except Exception:
        pass

# Step-by-step write-up (explicit)
st.markdown("""
## Step-by-step: What the app does & how to interpret graphs

**1) Data Upload**: The user uploads a CSV (patient data), and the app processes it. If no file is uploaded, the built-in synthetic dataset is used.
**2) BAT Training**: BAT (Bat Algorithm) selects the best subset of features for classification. It runs through multiple generations, evolving the solution. The objective inside BAT uses 5-fold cross-validated accuracy (`cross_val_score`) on the selected transformed features.
**3) Model Prediction**: Once trained, the Gaussian Naive Bayes model is fitted on the selected features and used for predictions. Probabilities (when available) come from `predict_proba`.
**4) Evaluation**: View performance metrics (accuracy, precision, F1), confusion matrix, ROC/AUC and feature importance (mutual information).
**5) Prediction Explanation**: The app returns a human-readable explanation and shows which dataset columns were used for the patient input.

### Important notes:
- This is a research/educational tool — **not a diagnostic device**.
- Always consult healthcare professionals for diagnosis and treatment.
""")

