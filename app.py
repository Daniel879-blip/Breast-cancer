# app.py
"""
Breast Cancer Predictor — Hard-coded Medical Patient Form + BAT feature selection + GaussianNB
Single-file Streamlit app.

Usage:
- pip install -r requirements.txt
- streamlit run app.py

Requirements:
streamlit, pandas, numpy, scikit-learn, plotly, joblib, seaborn, matplotlib
"""

import streamlit as st
import pandas as pd
import numpy as np
import time
import json
from pathlib import Path
from joblib import dump, load

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

st.set_page_config(page_title="Breast Cancer Medical Form App", layout="wide")
st.title("🧬 Breast Cancer Predictor — Medical-style Patient Form (BAT + Naive Bayes)")
st.markdown(
    "Upload dataset (optional) and select target in the sidebar. "
    "Use BAT parameters to run feature selection then train the model. "
    "Enter patient data in the form below (age, sex, blood type, family history, etc.) to get a prediction."
)

# -------------------------
# Helpers
# -------------------------
def make_onehot_encoder_compat(**kwargs):
    """Return OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)

def explain_prediction_text(prob, label):
    pct = f"{prob*100:.1f}%" if prob is not None else "N/A"
    if label == 1:
        return (
            f"**Prediction:** POSITIVE (likely malignant). Confidence: **{pct}**.\n\n"
            "Interpretation: The model found the input features more similar to malignant cases in the training data.\n\n"
            "Recommended next steps:\n"
            "1. Urgent clinical follow-up and specialist consult.\n"
            "2. Diagnostic imaging (mammogram/ultrasound/MRI) and biopsy if indicated.\n"
            "3. Treat this as a screening aid only — not a definitive diagnosis."
        )
    else:
        return (
            f"**Prediction:** NEGATIVE (likely benign). Confidence: **{pct}**.\n\n"
            "Interpretation: The model found the input features more similar to benign cases in the training data.\n\n"
            "Recommended next steps:\n"
            "1. Continue routine surveillance and clinical follow-up.\n"
            "2. If symptoms persist, pursue diagnostic imaging."
        )

# BAT implementation (binary)
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

# -------------------------
# Sidebar controls
# -------------------------
st.sidebar.header("Controls & Data")
uploaded = st.sidebar.file_uploader("Upload CSV dataset (optional)", type=['csv'])
use_demo = st.sidebar.checkbox("Use demo medical dataset if no upload", value=True if uploaded is None else False)

st.sidebar.markdown("---")
st.sidebar.subheader("BAT Feature Selection")
num_bats = st.sidebar.slider("Number of bats", 6, 80, 24)
max_gen = st.sidebar.slider("Generations", 5, 120, 40)
loudness = st.sidebar.slider("Loudness (A)", 0.05, 1.0, 0.5, step=0.05)
pulse_rate = st.sidebar.slider("Pulse rate (r)", 0.0, 1.0, 0.5, step=0.05)
sleep_per_gen = st.sidebar.slider("UI speed (sec/gen)", 0.0, 0.1, 0.01, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Classifier")
clf_choice = st.sidebar.selectbox("Classifier", ["Naive Bayes (Gaussian)"])

st.sidebar.markdown("---")
run_btn = st.sidebar.button("🔁 Run BAT & Train")
save_artifacts = st.sidebar.checkbox("Save artifacts to ./models", value=False)

# -------------------------
# Synthetic demo dataset (contains the medical fields)
# -------------------------
@st.cache_data
def create_demo_medical(n=1500, seed=42):
    rng = np.random.default_rng(seed)
    age = rng.integers(25, 85, n)
    sex = rng.choice(['Female', 'Male'], size=n, p=[0.98, 0.02])
    blood_type = rng.choice(['A','B','AB','O'], size=n, p=[0.4,0.3,0.1,0.2])
    tumor_size = np.round(rng.normal(12.0, 8.0, n).clip(1, 80), 1)  # mm
    family_history = rng.choice(['Yes','No'], size=n, p=[0.12,0.88])
    smoking = rng.choice(['Never','Former','Current'], size=n, p=[0.6,0.25,0.15])
    alcohol = rng.choice(['None','Low','Moderate','High'], size=n, p=[0.4,0.35,0.2,0.05])
    bmi = np.round(rng.normal(27, 5, n).clip(15, 45), 1)
    bp = np.round(rng.normal(120, 15, n).clip(80, 200), 0)
    glucose = np.round(rng.normal(95, 15, n).clip(50, 250), 0)
    hrt = rng.choice(['Yes','No'], size=n, p=[0.12,0.88])
    phys_act = rng.choice(['Low','Moderate','High'], size=n, p=[0.3,0.5,0.2])
    prior_lump = rng.choice(['Yes','No'], size=n, p=[0.08,0.92])
    breast_density = rng.choice(['A','B','C','D'], size=n, p=[0.2,0.35,0.3,0.15])
    menopausal = rng.choice(['Pre','Peri','Post'], size=n, p=[0.25,0.1,0.65])

    # risk score (toy model) to simulate target
    score = (
        0.03*(age-40)
        + 0.7*(tumor_size-10)/10
        + 1.0*(family_history == 'Yes').astype(float)
        + 0.4*(hrt == 'Yes').astype(float)
        + 0.6*(prior_lump == 'Yes').astype(float)
        + 0.2*(breast_density == 'D').astype(float)
        + 0.15*(bmi-25)/5
    )
    prob = 1/(1+np.exp(-score))
    target = (rng.random(n) < prob).astype(int)

    df = pd.DataFrame({
        'age': age,
        'sex': sex,
        'blood_type': blood_type,
        'tumor_size_mm': tumor_size,
        'family_history': family_history,
        'smoking': smoking,
        'alcohol': alcohol,
        'bmi': bmi,
        'blood_pressure': bp,
        'glucose_level': glucose,
        'hrt_use': hrt,
        'physical_activity': phys_act,
        'prior_breast_lump': prior_lump,
        'breast_density': breast_density,
        'menopausal_status': menopausal,
        'target': target
    })
    return df

# load dataset
if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Unable to read uploaded CSV: {e}")
        st.stop()
elif use_demo:
    df_raw = create_demo_medical()
else:
    st.info("Upload a CSV or enable demo dataset to proceed.")
    st.stop()

# preview
with st.expander("Dataset preview (first 10 rows)"):
    st.dataframe(df_raw.head(10))

# choose target column
st.sidebar.markdown("---")
tgt_col = st.sidebar.selectbox("Choose target column (label)", options=list(df_raw.columns), index=list(df_raw.columns).index('target') if 'target' in df_raw.columns else len(df_raw.columns)-1)
st.sidebar.write(f"Using target column: **{tgt_col}**")

# Clean & prepare function
def clean_prepare(df, target_col):
    dfc = df.copy()
    dfc = dfc.dropna(how='all')
    # map string target to 0/1 if necessary
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
    # drop id-like columns
    for c in list(X.columns):
        if c.lower() in ('id','patient_id','pid'):
            X = X.drop(columns=[c])
    return X.reset_index(drop=True), y.reset_index(drop=True)

X_df, y_ser = clean_prepare(df_raw, tgt_col)

# quick dataset stats
st.subheader("Dataset summary")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{X_df.shape[0]}")
c2.metric("Features", f"{X_df.shape[1]}")
c3.metric("Target", f"{tgt_col}")

# missing values
miss = X_df.isna().sum()
if miss.sum() > 0:
    with st.expander("Missing values (per column)"):
        st.dataframe(miss[miss > 0])

# class distribution chart
st.subheader("Class distribution")
counts = y_ser.value_counts().to_dict()
fig_p = px.pie(values=list(counts.values()), names=list(counts.keys()), title="Class distribution")
st.plotly_chart(fig_p, use_container_width=True)

# detect numeric/categorical
numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
st.markdown(f"Detected features — numeric: {len(numeric_cols)}, categorical: {len(cat_cols)}")
if len(cat_cols) > 0:
    with st.expander("Categorical columns detected"):
        st.write(cat_cols)

# build preprocessor from dataset columns (we will align patient form inputs to these later)
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

# placeholders for UI update during BAT
conv_placeholder = st.empty()
progress_placeholder = st.sidebar.empty()
status_placeholder = st.sidebar.empty()
metrics_placeholder = st.empty()

trained_clf = None
trained_preproc = None
selected_indices = []
transformed_feature_names = []
convergence_curve = []

# progress UI callback
def ui_progress_cb(gen_idx, best_score, conv_so_far, mx):
    frac = int((gen_idx+1)/mx * 100)
    progress_placeholder.progress(frac)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=conv_so_far, mode='lines+markers', name='best_cv'))
    fig.update_layout(title=f"BAT convergence (gen {gen_idx+1}/{mx})", xaxis_title="Generation", yaxis_title="Best CV score", height=380)
    conv_placeholder.plotly_chart(fig, use_container_width=True)
    status_placeholder.info(f"BAT running: generation {gen_idx+1}/{mx}")

# -------------------------
# Run BAT & Train (sidebar button)
# -------------------------
if run_btn:
    status_placeholder.info("Preparing data and running BAT...")
    work = pd.concat([X_df, y_ser], axis=1).dropna().reset_index(drop=True)
    if work.shape[0] < 20:
        st.warning("Few samples after dropping missing rows — results may be unstable.")
    X_clean = work.drop(columns=[tgt_col])
    y_clean = work[tgt_col].astype(int)

    # rebuild preprocessor on cleaned data
    preproc, num_cols, cat_cols = build_preprocessor(X_clean)
    try:
        preproc.fit(X_clean)
    except Exception as e:
        st.error(f"Preprocessor failed to fit: {e}")
        st.stop()

    # transformed names
    try:
        transformed_feature_names = list(preproc.get_feature_names_out())
    except Exception:
        # manual fallback
        transformed_feature_names = []
        if cat_cols:
            ohe = preproc.named_transformers_['cat']
            cats = ohe.categories_
            for col, levels in zip(cat_cols, cats):
                for lvl in levels:
                    transformed_feature_names.append(f"{col}__{lvl}")
        if num_cols:
            transformed_feature_names.extend(num_cols)

    # transform dataset
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
            progress_cb=ui_progress_cb,
            sleep_per_gen=sleep_per_gen
        )
    except Exception as e:
        st.error(f"BAT failed: {e}")
        st.stop()

    progress_placeholder.empty()
    status_placeholder.success("BAT finished.")

    if len(sel_idx) == 0:
        st.warning("BAT selected 0 features; falling back to mutual-info top features.")
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
        fig_fin = px.line(y=convergence_curve, labels={'value':'Best CV score','index':'Generation'}, title="BAT Convergence (final)")
        fig_fin.update_traces(mode='lines+markers')
        conv_placeholder.plotly_chart(fig_fin, use_container_width=True)
    except Exception:
        pass

    # Train GaussianNB
    try:
        X_sel = X_trans[:, selected_indices]
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y_vals, test_size=0.2, random_state=42, stratify=y_vals)
        clf = GaussianNB()
        clf.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # metrics
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc = float((y_pred == y_test).mean())
    metrics_placeholder.metric("Test accuracy", f"{acc:.3f}")
    st.subheader("Model evaluation (hold-out test set)")
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))

    # confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], colorscale="Blues"))
    cm_fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)

    # ROC
    try:
        y_prob = clf.predict_proba(X_test)[:,1]
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)
        roc_fig = go.Figure()
        roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC={roc_auc:.3f}"))
        roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='Chance'))
        roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
        st.plotly_chart(roc_fig, use_container_width=True)
    except Exception:
        pass

    # feature importance (mutual info)
    try:
        if X_sel.shape[1] > 0:
            mi = mutual_info_classif(X_sel, y_vals)
            mi_df = pd.DataFrame({'feature': selected_names, 'mi': mi}).sort_values('mi', ascending=False)
            st.subheader("Feature importance (Mutual Information)")
            st.dataframe(mi_df.style.background_gradient(cmap='Oranges'))
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

    # store in session state for immediate use
    st.session_state['clf'] = clf
    st.session_state['preproc'] = preproc
    st.session_state['selected_indices'] = selected_indices
    st.session_state['transformed_feature_names'] = transformed_feature_names
    st.session_state['convergence_curve'] = convergence_curve

# -------------------------
# Hard-coded Medical Patient Form (always available)
# -------------------------
st.markdown("---")
st.header("🧾 Patient Form — Medical-style inputs")

# Define medical form fields and controls
def default_values_from_df(Xdf):
    defaults = {}
    for col in Xdf.columns:
        if col in Xdf.select_dtypes(include=[np.number]).columns:
            defaults[col] = float(Xdf[col].mean())
        else:
            vals = Xdf[col].dropna().unique().tolist()
            defaults[col] = vals[0] if vals else ""
    return defaults

# Standard medical features we want in the form (user asked)
medical_fields = [
    ('age', 'numeric'),
    ('sex', 'categorical', ['Female','Male']),
    ('blood_type', 'categorical', ['A','B','AB','O']),
    ('tumor_size_mm', 'numeric'),
    ('family_history', 'categorical', ['Yes','No']),
    ('smoking', 'categorical', ['Never','Former','Current']),
    ('alcohol', 'categorical', ['None','Low','Moderate','High']),
    ('bmi', 'numeric'),
    ('blood_pressure', 'numeric'),
    ('glucose_level', 'numeric'),
    ('hrt_use', 'categorical', ['Yes','No']),
    ('physical_activity', 'categorical', ['Low','Moderate','High']),
    ('prior_breast_lump', 'categorical', ['Yes','No']),
    ('breast_density', 'categorical', ['A','B','C','D']),
    ('menopausal_status', 'categorical', ['Pre','Peri','Post'])
]

# Get defaults from dataset when possible
defaults = default_values_from_df(X_df)

with st.form("medical_patient_form"):
    patient = {}
    for field in medical_fields:
        fname = field[0]
        ftype = field[1]
        if ftype == 'numeric':
            vmin = float(X_df[fname].min()) if fname in X_df.columns and pd.api.types.is_numeric_dtype(X_df[fname]) else 0.0
            vmax = float(X_df[fname].max()) if fname in X_df.columns and pd.api.types.is_numeric_dtype(X_df[fname]) else 100.0
            vmean = float(X_df[fname].mean()) if fname in X_df.columns and pd.api.types.is_numeric_dtype(X_df[fname]) else (vmin+vmax)/2
            patient[fname] = st.number_input(fname.replace('_',' ').title(), min_value=vmin, max_value=vmax, value=vmean, format="%.5f")
        else:
            options = field[2] if len(field) > 2 else (X_df[fname].dropna().unique().tolist() if fname in X_df.columns else ['Unknown'])
            # ensure options is not empty
            if not options:
                options = ['Unknown']
            patient[fname] = st.selectbox(fname.replace('_',' ').title(), options=options)

    submit_patient = st.form_submit_button("Predict for this patient")

# Prediction logic: map patient inputs into preprocessor's expected original columns
def build_sample_for_prediction(patient_inputs, original_columns, Xdf):
    """
    Build a one-row DataFrame with columns = original_columns (the dataset's original columns),
    filling with patient_inputs where available and using means/modes for missing ones.
    """
    row = {}
    for c in original_columns:
        if c in patient_inputs:
            row[c] = patient_inputs[c]
        else:
            # if the dataset has the column, use its mean/mode
            if c in Xdf.columns:
                if pd.api.types.is_numeric_dtype(Xdf[c]):
                    row[c] = float(Xdf[c].mean())
                else:
                    vals = Xdf[c].dropna().unique().tolist()
                    row[c] = vals[0] if vals else ""
            else:
                # if not available in dataset, try to map from similar medical field names
                if c in patient_inputs:
                    row[c] = patient_inputs[c]
                else:
                    # default fallback
                    row[c] = 0 if isinstance(Xdf.get(c, pd.Series(dtype=float)).dtype, np.dtype) else ""
    return pd.DataFrame([row])

# Handle patient prediction
if submit_patient:
    # Determine model & preproc
    if 'clf' in st.session_state and 'preproc' in st.session_state and 'selected_indices' in st.session_state:
        model = st.session_state['clf']
        preproc_model = st.session_state['preproc']
        sel_idx = st.session_state['selected_indices']
        orig_cols = X_df.columns.tolist()
    else:
        # try to load saved artifacts
        if Path("models/model.joblib").exists() and Path("models/preprocessor.joblib").exists() and Path("models/selected_indices.npy").exists():
            try:
                model = load("models/model.joblib")
                preproc_model = load("models/preprocessor.joblib")
                sel_idx = list(np.load("models/selected_indices.npy").astype(int))
                with open("models/feature_names.json", 'r') as f:
                    transformed_feature_names = json.load(f)
                orig_cols = X_df.columns.tolist()
                st.sidebar.info("Loaded saved model for prediction.")
            except Exception as e:
                st.error(f"Could not load saved model for prediction: {e}")
                model = None
                preproc_model = None
                sel_idx = []
                orig_cols = X_df.columns.tolist()
        else:
            st.warning("No trained model available. Please run training (Run BAT & Train) in the sidebar first.")
            model = None
            preproc_model = None
            sel_idx = []
            orig_cols = X_df.columns.tolist()

    if model is not None and preproc_model is not None and len(sel_idx) > 0:
        sample_df = build_sample_for_prediction(patient, orig_cols, X_df)
        # Preprocess
        try:
            sample_trans = preproc_model.transform(sample_df)
            if not isinstance(sample_trans, np.ndarray):
                try:
                    sample_trans = sample_trans.toarray()
                except Exception:
                    sample_trans = np.array(sample_trans)
        except Exception as e:
            st.error(f"Preprocessing of patient input failed: {e}")
            sample_trans = None

        if sample_trans is not None:
            try:
                sample_sel = sample_trans[:, sel_idx]
            except Exception as e:
                st.error(f"Selecting trained features failed: {e}")
                sample_sel = None

            if sample_sel is not None:
                try:
                    pred = model.predict(sample_sel)[0]
                    prob = float(model.predict_proba(sample_sel)[0][int(pred)]) if hasattr(model, "predict_proba") else None
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred = None
                    prob = None

                if pred is not None:
                    if int(pred) == 1:
                        st.error("🔴 Prediction: Positive (Malignant)")
                    else:
                        st.success("🟢 Prediction: Negative (Benign)")
                    if prob is not None:
                        st.metric("Model confidence", f"{prob:.2%}")

                    # explanation
                    st.markdown("### Explanation & recommended next steps")
                    st.write(explain_prediction_text(prob if prob is not None else 0, int(pred)))

                    # patient summary and download
                    outdf = sample_df.copy()
                    outdf['prediction'] = int(pred)
                    outdf['confidence'] = float(prob) if prob is not None else None
                    st.subheader("Patient input summary")
                    st.dataframe(outdf.T)
                    st.download_button("Download patient report (CSV)", outdf.to_csv(index=False).encode('utf-8'), "patient_report.csv", "text/csv")
    else:
        st.error("No usable trained model found. Train model first (sidebar) or load saved artifacts.")

# -------------------------
# Additional charts & write-up
# -------------------------
st.markdown("---")
st.header("Charts, Analysis & Write-up")

# correlation heatmap
if len(numeric_cols) > 1:
    st.subheader("Correlation heatmap (numeric features)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(X_df[numeric_cols].corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# distributions for top numeric features
st.subheader("Feature distributions")
plot_cols = numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols
for c in plot_cols:
    fig = px.histogram(X_df, x=c, color=y_ser, nbins=30, title=f"{c} distribution by target", marginal='box')
    st.plotly_chart(fig, use_container_width=True)

# selected features display
if len(selected_indices) > 0:
    st.subheader("Selected transformed features (example list)")
    try:
        names = [transformed_feature_names[i] for i in selected_indices]
        st.write(names[:40])
    except Exception:
        pass

# Write-up / methodology
st.markdown("""
## Write-up — Methodology & Interpretation

**Goal:** Provide an interactive tool to evaluate breast cancer risk using medically meaningful patient inputs.
- Input fields are clinical-style (age, sex, blood type, family history, BMI, HRT use, tumor size, etc.).
- The app uses a binary **BAT (Bat Algorithm)** for feature selection in the transformed feature space (after one-hot encoding & scaling).
- A **Gaussian Naive Bayes** classifier is trained on BAT-selected features and evaluated on a hold-out test set.
- The patient form maps human inputs to the preprocessing pipeline so predictions are consistent.

**Limitations & disclaimers:**
- This is an academic prototype — **not a medical diagnostic device**.
- Model performance depends on dataset quality and representativeness.
- Always consult healthcare professionals for clinical decisions.
""")

# End of app
