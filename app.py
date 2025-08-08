# app.py
"""
Breast Cancer Risk App (medical-style patient form)
- Uses synthetic dataset by default: /mnt/data/synthetic_breast_cancer_risk_factors.csv
- Upload your own CSV (choose target column)
- Sidebar: BAT parameters, classifier, Run BAT & Train, Save artifacts
- Patient form: Age, Sex, BMI, Family History, Smoking, Alcohol, Physical Activity, Hormone Therapy, Breastfeeding, Pregnancies
- Charts: class distribution, BAT convergence, ROC, confusion matrix, feature importance, correlation heatmap
- Write-up + downloadable patient report
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

st.set_page_config(page_title="Breast Cancer Risk — Medical Form", layout="wide")
st.title("🧬 Breast Cancer Risk Predictor — Medical-style Patient Form (BAT + Naive Bayes)")
st.markdown("Use the sidebar to upload a dataset or use the built-in synthetic dataset. Configure BAT and train the model. Fill the patient form to get a prediction and explanation.")

# ---------------------------
# Helpers
# ---------------------------
def make_onehot_encoder_compat(**kwargs):
    """Create OneHotEncoder compatible across sklearn versions."""
    try:
        return OneHotEncoder(sparse_output=False, **kwargs)
    except TypeError:
        return OneHotEncoder(sparse=False, **kwargs)

def explain_prediction_text(prob, label):
    pct = f"{prob*100:.1f}%" if prob is not None else "N/A"
    if label == 1:
        return (
            f"**Prediction:** POSITIVE (likely malignant). Confidence: **{pct}**.\n\n"
            "Interpretation: The model found features that resemble malignant cases in the training data.\n\n"
            "Recommended next steps:\n"
            "1. Clinical assessment and imaging (mammogram/ultrasound/MRI).\n"
            "2. Consider biopsy if clinically indicated.\n"
            "3. Consult an oncologist — this is a screening aid not a diagnosis."
        )
    else:
        return (
            f"**Prediction:** NEGATIVE (likely benign). Confidence: **{pct}**.\n\n"
            "Interpretation: The model found features more similar to benign cases in the training data.\n"
            "Recommended next steps: routine surveillance and clinical follow-up if symptoms persist."
        )

# ---------------------------
# BAT algorithm (binary)
# ---------------------------
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

# ---------------------------
# Sidebar
# ---------------------------
st.sidebar.header("Controls & Data")
uploaded = st.sidebar.file_uploader("Upload CSV dataset (optional)", type=["csv"])
use_builtin = st.sidebar.checkbox("Use built-in synthetic dataset (recommended)", value=True if uploaded is None else False)

st.sidebar.markdown("---")
st.sidebar.subheader("BAT parameters")
num_bats = st.sidebar.slider("Number of bats", 6, 80, 24)
max_gen = st.sidebar.slider("Generations", 5, 120, 40)
loudness = st.sidebar.slider("Loudness (A)", 0.05, 1.0, 0.5, step=0.05)
pulse_rate = st.sidebar.slider("Pulse rate (r)", 0.0, 1.0, 0.5, step=0.05)
sleep_per_gen = st.sidebar.slider("UI speed (sec/gen)", 0.0, 0.05, 0.01, step=0.005)

st.sidebar.markdown("---")
st.sidebar.subheader("Classifier")
classifier_choice = st.sidebar.selectbox("Classifier (currently only Naive Bayes supported)", ["Naive Bayes (Gaussian)"])

st.sidebar.markdown("---")
run_button = st.sidebar.button("🔁 Run BAT & Train")
save_artifacts = st.sidebar.checkbox("Save artifacts to ./models", value=False)

# ---------------------------
# Load dataset (builtin or uploaded)
# ---------------------------
BUILTIN_PATH = "/mnt/data/synthetic_breast_cancer_risk_factors.csv"

def load_builtin():
    p = Path(BUILTIN_PATH)
    if p.exists():
        return pd.read_csv(p)
    else:
        st.error(f"Built-in dataset not found at {BUILTIN_PATH}. Please upload a dataset.")
        return None

if uploaded is not None:
    try:
        df_raw = pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Could not read uploaded CSV: {e}")
        st.stop()
elif use_builtin:
    df_raw = load_builtin()
else:
    st.info("Upload a CSV or enable the built-in dataset.")
    st.stop()

# preview
with st.expander("Dataset preview (first 8 rows)"):
    st.dataframe(df_raw.head(8))

# pick target column
st.sidebar.markdown("---")
tgt_col = st.sidebar.selectbox("Select target (label) column", options=list(df_raw.columns), index=list(df_raw.columns).index("Target") if "Target" in df_raw.columns else len(df_raw.columns)-1)
st.sidebar.write(f"Using target: **{tgt_col}**")

# tidy dataset
def clean_prepare(df, target_col):
    dfc = df.copy()
    dfc = dfc.dropna(how='all')
    # try to coerce target to 0/1
    if dfc[target_col].dtype == object or str(dfc[target_col].dtype).startswith("category"):
        vals = [str(x).lower() for x in dfc[target_col].dropna().unique()]
        if any(v.startswith("m") for v in vals) and any(v.startswith("b") for v in vals):
            dfc[target_col] = dfc[target_col].map(lambda x: 1 if str(x).lower().startswith("m") else 0)
        else:
            try:
                dfc[target_col] = pd.to_numeric(dfc[target_col])
            except Exception:
                pass
    dfc = dfc.loc[dfc[target_col].notna()].reset_index(drop=True)
    y = dfc[target_col].astype(int)
    X = dfc.drop(columns=[target_col])
    # drop id-like
    for c in list(X.columns):
        if c.lower() in ("id","patient_id","pid"):
            X.drop(columns=[c], inplace=True)
    return X.reset_index(drop=True), y.reset_index(drop=True)

X_df, y_ser = clean_prepare(df_raw, tgt_col)

# quick stats
st.subheader("Data summary")
col1, col2, col3 = st.columns(3)
col1.metric("Rows", f"{X_df.shape[0]}")
col2.metric("Features", f"{X_df.shape[1]}")
col3.metric("Target", f"{tgt_col}")

# missing
miss = X_df.isna().sum()
if miss.sum() > 0:
    with st.expander("Missing values per column"):
        st.dataframe(miss[miss > 0])

# class distribution plot
st.subheader("Target class distribution")
counts = y_ser.value_counts().to_dict()
fig_pie = px.pie(values=list(counts.values()), names=list(counts.keys()), title="Class distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# detect types
numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
st.markdown(f"Detected features — numeric: {len(numeric_cols)}, categorical: {len(cat_cols)}")
if len(cat_cols) > 0:
    with st.expander("Categorical columns"):
        st.write(cat_cols)

# build preprocessor
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

# placeholders
conv_placeholder = st.empty()
progress_placeholder = st.sidebar.empty()
status_placeholder = st.sidebar.empty()
metrics_placeholder = st.empty()

trained_clf = None
trained_preproc = None
selected_indices = []
transformed_feature_names = []
convergence_curve = []

# UI progress callback for BAT
def ui_progress(gen_idx, best_score, conv_so_far, max_gen_local):
    frac = int((gen_idx+1)/max_gen_local * 100)
    progress_placeholder.progress(frac)
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=conv_so_far, mode='lines+markers', name='best_cv'))
    fig.update_layout(title=f"BAT convergence (gen {gen_idx+1}/{max_gen_local})", xaxis_title="Generation", yaxis_title="Best CV score", height=380)
    conv_placeholder.plotly_chart(fig, use_container_width=True)
    status_placeholder.info(f"BAT running: gen {gen_idx+1}/{max_gen_local}")

# ---------------------------
# Train (Run BAT & Train)
# ---------------------------
if run_button:
    status_placeholder.info("Preparing data and running BAT...")
    work = pd.concat([X_df, y_ser], axis=1).dropna().reset_index(drop=True)
    if work.shape[0] < 20:
        st.warning("Few samples after dropping missing rows — results may be unstable.")
    X_clean = work.drop(columns=[tgt_col])
    y_clean = work[tgt_col].astype(int)

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
            progress_cb=ui_progress,
            sleep_per_gen=sleep_per_gen
        )
    except Exception as e:
        st.error(f"BAT failed: {e}")
        st.stop()

    progress_placeholder.empty()
    status_placeholder.success("BAT finished.")

    if len(sel_idx) == 0:
        st.warning("BAT selected zero features — falling back to mutual-info top-k.")
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

    # Train classifier
    try:
        X_sel = X_trans[:, selected_indices]
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y_vals, test_size=0.2, random_state=42, stratify=y_vals)
        clf = GaussianNB()
        clf.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # evaluate
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

    # Save artifacts
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

    # store in session state
    st.session_state['clf'] = clf
    st.session_state['preproc'] = preproc
    st.session_state['selected_indices'] = selected_indices
    st.session_state['transformed_feature_names'] = transformed_feature_names
    st.session_state['convergence_curve'] = convergence_curve

# ---------------------------
# Hard-coded medical patient form (fields match synthetic dataset)
# ---------------------------
st.markdown("---")
st.header("🧾 Patient Form — medical-style inputs")

# fields we used in synthetic dataset (and that will be used if present in uploaded dataset)
medical_fields = [
    ('Age', 'numeric'),
    ('Sex', 'categorical', ['Female', 'Male']),
    ('BMI', 'numeric'),
    ('Family_History', 'categorical', ['Yes', 'No']),
    ('Smoking_Status', 'categorical', ['Never', 'Former', 'Current']),
    ('Alcohol_Intake', 'categorical', ['Low', 'Moderate', 'High']),
    ('Physical_Activity', 'categorical', ['Low', 'Moderate', 'High']),
    ('Hormone_Therapy', 'categorical', ['Yes', 'No']),
    ('Breastfeeding_History', 'categorical', ['Yes', 'No']),
    ('Pregnancies', 'numeric')
]

# create form with sensible defaults taken from dataset where possible
def get_default(Xdf, col, ftype):
    if col in Xdf.columns:
        if ftype == 'numeric':
            try:
                return float(Xdf[col].mean())
            except Exception:
                return 0.0
        else:
            vals = Xdf[col].dropna().unique().tolist()
            return vals[0] if vals else None
    else:
        # fallback sensible defaults
        return 40.0 if ftype == 'numeric' else None

with st.form("medical_form"):
    patient = {}
    for f in medical_fields:
        name = f[0]
        if f[1] == 'numeric':
            default = get_default(X_df, name, 'numeric')
            vmin = float(X_df[name].min()) if name in X_df.columns and pd.api.types.is_numeric_dtype(X_df[name]) else 0.0
            vmax = float(X_df[name].max()) if name in X_df.columns and pd.api.types.is_numeric_dtype(X_df[name]) else 100.0
            patient[name] = st.number_input(name.replace('_',' '), value=default, min_value=vmin, max_value=vmax, format="%.2f")
        else:
            options = f[2] if len(f) > 2 else (X_df[name].dropna().unique().tolist() if name in X_df.columns else ['Unknown'])
            if not options:
                options = ['Unknown']
            default = get_default(X_df, name, 'categorical')
            idx = options.index(default) if default in options else 0
            patient[name] = st.selectbox(name.replace('_',' '), options, index=idx)
    submit_patient = st.form_submit_button("Predict for this patient")

# prediction pipeline helper: build sample from original columns
def build_sample_from_patient(patient_inputs, Xdf):
    # build a row for the model's expected original columns
    row = {}
    for c in Xdf.columns:
        # if patient form contains a matching column use it (case-insensitive)
        if c in patient_inputs:
            row[c] = patient_inputs[c]
        else:
            # try to match by name ignoring case/underscores
            key_match = None
            for k in patient_inputs:
                if k.lower().replace('_','') == c.lower().replace('_',''):
                    key_match = k
                    break
            if key_match is not None:
                row[c] = patient_inputs[key_match]
            else:
                # fallback: mean for numeric, mode for categorical
                if pd.api.types.is_numeric_dtype(Xdf[c]):
                    row[c] = float(Xdf[c].mean())
                else:
                    vals = Xdf[c].dropna().unique().tolist()
                    row[c] = vals[0] if vals else ""
    return pd.DataFrame([row], columns=Xdf.columns)

# handle patient prediction
if submit_patient:
    if 'clf' in st.session_state and 'preproc' in st.session_state and 'selected_indices' in st.session_state:
        model = st.session_state['clf']
        preproc_model = st.session_state['preproc']
        sel_idx = st.session_state['selected_indices']
        try:
            sample_df = build_sample_from_patient(patient, X_df)
            sample_trans = preproc_model.transform(sample_df)
            if not isinstance(sample_trans, np.ndarray):
                try:
                    sample_trans = sample_trans.toarray()
                except Exception:
                    sample_trans = np.array(sample_trans)
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
            st.markdown("### Explanation & Recommended next steps")
            st.write(explain_prediction_text(prob if prob is not None else 0, int(pred)))
            outdf = sample_df.copy()
            outdf['prediction'] = int(pred)
            outdf['confidence'] = float(prob) if prob is not None else None
            st.subheader("Patient input summary")
            st.dataframe(outdf.T)
            st.download_button("Download patient report (CSV)", outdf.to_csv(index=False).encode('utf-8'), "patient_report.csv", "text/csv")
    else:
        st.warning("Model not trained. Please run BAT & Train in the sidebar first (or load saved model artifacts).")

# ---------------------------
# Charts & write-up
# ---------------------------
st.markdown("---")
st.header("Charts, Analysis & Write-up")

# correlation heatmap (numeric)
if len(numeric_cols) > 1:
    st.subheader("Correlation heatmap (numeric features)")
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(X_df[numeric_cols].corr(), cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# distributions for top numeric features
st.subheader("Numeric distributions")
plot_cols = numeric_cols[:6] if len(numeric_cols) >= 6 else numeric_cols
for c in plot_cols:
    fig = px.histogram(X_df, x=c, color=y_ser, nbins=30, title=f"{c} distribution by target", marginal="box")
    st.plotly_chart(fig, use_container_width=True)

# selected features list
if len(selected_indices) > 0:
    st.subheader("Selected transformed features (example)")
    try:
        names = [transformed_feature_names[i] for i in selected_indices]
        st.write(names[:50])
    except Exception:
        pass

# write-up
st.markdown("""
## Write-up — methodology & interpretation

**Overview:** This app demonstrates an end-to-end prototype that uses clinically meaningful patient inputs (age, sex, BMI, family history, smoking, alcohol intake, physical activity, hormone therapy) to predict probable breast cancer presence using:
- **BAT (Bat Algorithm)** for feature selection (binary variant operating in the transformed feature space),
- **Gaussian Naive Bayes** as the classifier.

**How to use:** Upload your dataset or use the built-in synthetic dataset. Select the target column (0/1). Configure BAT parameters in the sidebar and press **Run BAT & Train**. Once the model finishes, fill the medical patient form on this page and press Predict.

**Interpretation & limitations:** This is a research/educational prototype — NOT a clinical diagnostic tool. Predictions are probabilistic and depend on dataset quality and representativeness. Always consult medical professionals for diagnosis and treatment.
""")

# End of app
