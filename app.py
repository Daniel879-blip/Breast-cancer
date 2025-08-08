# app.py
"""
Advanced Streamlit app — Trainable from sidebar
Breast Cancer Predictor — BAT feature selection + Gaussian Naive Bayes
Single-file solution (no external utils required)
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
import matplotlib.pyplot as plt
import seaborn as sns

# sklearn
import sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import mutual_info_classif

# persistence
from joblib import dump, load

st.set_page_config(page_title="Breast Cancer Predictor — Trainable", layout="wide", initial_sidebar_state="expanded")
st.title("🧬 Breast Cancer Predictor — Train from Sidebar (BAT + Naive Bayes)")
st.markdown("**Instructions:** Upload a CSV (or use example), pick the label/target column, adjust BAT parameters in the sidebar, then click **Run BAT & Train**. After training the patient form will appear for predictions.")

# -----------------------------------------------------------------------------
# Utility / compatibility helpers
# -----------------------------------------------------------------------------
def make_onehot_encoder_compat(**kwargs):
    """
    Create a OneHotEncoder that is compatible across scikit-learn versions.
    Newer versions use `sparse_output=False`, older use `sparse=False`.
    Pass any other kwargs you'd like (e.g., handle_unknown).
    """
    ver = sklearn.__version__
    # We'll attempt to set sparse_output first (newer sklearn), else fallback to sparse
    try:
        enc = OneHotEncoder(**{**kwargs, **({"sparse_output": False})})
    except TypeError:
        # older scikit-learn
        enc = OneHotEncoder(**{**kwargs, **({"sparse": False})})
    return enc

def explain_prediction_text(prob, label):
    """
    Human-readable explanation for a prediction.
    prob: probability (0..1) for predicted class
    label: 1 => positive (malignant); 0 => negative (benign)
    """
    pct = None if prob is None else f"{prob*100:.1f}%"
    if label == 1:
        return (
            f"**Prediction:** POSITIVE (likely malignant). Confidence: **{pct}**.\n\n"
            "Interpretation: The model found the entered features more similar to malignant cases "
            "in the training data. Recommended next steps:\n"
            "1. Urgent clinical follow-up (specialist consult).\n"
            "2. Diagnostic imaging (mammogram/ultrasound/MRI) and biopsy if indicated.\n"
            "3. Use this result as a preliminary screening aid only — not a definitive diagnosis."
        )
    else:
        return (
            f"**Prediction:** NEGATIVE (likely benign). Confidence: **{pct}**.\n\n"
            "Interpretation: The model found the entered features more similar to benign cases "
            "in the training data. Recommended next steps:\n"
            "1. Continue routine surveillance and clinical follow-up.\n"
            "2. If symptoms persist or clinical concern remains, pursue diagnostic imaging.\n"
            "3. Always consult a medical professional for final decisions."
        )

# -----------------------------------------------------------------------------
# BAT algorithm (binary) and small helpers
# -----------------------------------------------------------------------------
def objective_cv_score(features, labels, mask):
    """
    Evaluate a binary mask using cross-validated GaussianNB accuracy.
    features: numpy array (n_samples, n_features)
    labels: numpy array (n_samples,)
    mask: list/array of 0/1 selecting features
    """
    mask = np.array(mask, dtype=bool)
    if mask.sum() == 0:
        return 0.0
    X_sel = features[:, mask]
    clf = GaussianNB()
    try:
        scores = cross_val_score(clf, X_sel, labels, cv=5, error_score='raise')
        return float(scores.mean())
    except Exception:
        return 0.0

def bat_feature_selection(features, labels, num_bats=20, max_gen=40, loudness=0.5, pulse_rate=0.5, progress_updater=None, sleep_per_gen=0.0):
    """
    Binary BAT algorithm. Returns selected_feature_indices, convergence_curve
    progress_updater(gen_idx, best_score, convergence_so_far, max_gen) is optional.
    """
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
        if progress_updater is not None:
            try:
                progress_updater(gen, best_score, convergence.copy(), max_gen)
            except Exception:
                pass
        if sleep_per_gen > 0:
            time.sleep(sleep_per_gen)

    selected_indices = [int(idx) for idx, bit in enumerate(best_pos) if bit == 1]
    return selected_indices, convergence

# -----------------------------------------------------------------------------
# Sidebar: dataset upload and parameter controls
# -----------------------------------------------------------------------------
st.sidebar.header("1) Dataset and Parameters")

uploaded_file = st.sidebar.file_uploader("Upload CSV dataset (optional). Last column often target; choose target below.", type=["csv"])
use_example = st.sidebar.checkbox("Use demo example dataset if no upload", value=True if uploaded_file is None else False)

# BAT params
st.sidebar.subheader("BAT Parameters (Feature Selection)")
num_bats = st.sidebar.slider("Number of bats", 6, 80, 24)
max_gen = st.sidebar.slider("Max generations", 5, 120, 40)
loudness = st.sidebar.slider("Loudness (A)", 0.05, 1.0, 0.5, step=0.05)
pulse_rate = st.sidebar.slider("Pulse rate (r)", 0.0, 1.0, 0.5, step=0.05)
sleep_per_gen = st.sidebar.slider("UI speed (seconds per generation)", 0.0, 0.25, 0.02, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("Classifier")
clf_choice = st.sidebar.selectbox("Classifier (for now)", ["Naive Bayes (Gaussian)"])
st.sidebar.markdown("---")
run_button = st.sidebar.button("🔁 Run BAT & Train")
save_artifacts = st.sidebar.checkbox("Save artifacts to ./models (model, preprocessor, selected indices)", value=False)

# -----------------------------------------------------------------------------
# Load dataset (uploaded or demo)
# -----------------------------------------------------------------------------
def safe_read(uploaded):
    try:
        df = pd.read_csv(uploaded)
        return df
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
        return None

if uploaded_file is not None:
    df_raw = safe_read(uploaded_file)
elif use_example:
    # Build a demo dataset from sklearn plus simple demographic columns to make patient form friendly
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    df_demo = pd.concat([d.data, d.target.rename("target")], axis=1)
    rng = np.random.default_rng(42)
    n = df_demo.shape[0]
    df_demo['age'] = rng.integers(30, 85, size=n)
    df_demo['sex'] = rng.choice(['Female','Male'], size=n, p=[0.98, 0.02])
    df_demo['family_history'] = rng.choice(['Yes','No'], size=n, p=[0.12, 0.88])
    df_raw = df_demo
else:
    df_raw = None

if df_raw is None:
    st.info("Upload a CSV or enable the example dataset to begin.")
    st.stop()

# Show preview and allow selecting target column
with st.expander("Preview dataset (first 5 rows)"):
    st.dataframe(df_raw.head())

st.sidebar.markdown("---")
st.sidebar.subheader("Pick target (label) column")
tgt_col = st.sidebar.selectbox("Target column (choose label column)", options=list(df_raw.columns), index=len(df_raw.columns)-1)

# -----------------------------------------------------------------------------
# Data cleaning & splitting helpers
# -----------------------------------------------------------------------------
def clean_and_prepare(df, target_col):
    dfc = df.copy()
    dfc = dfc.dropna(how='all')  # remove empty rows
    # If target is object-like and contains 'M'/'B' or 'malignant'/'benign' map to 1/0
    if dfc[target_col].dtype == object or str(dfc[target_col].dtype).startswith('category'):
        unique_vals = list(dfc[target_col].dropna().unique())
        unique_lower = [str(x).lower() for x in unique_vals]
        if any(x.startswith('m') for x in unique_lower) and any(x.startswith('b') for x in unique_lower):
            dfc[target_col] = dfc[target_col].map(lambda x: 1 if str(x).lower().startswith('m') else 0)
        else:
            # attempt numeric conversion
            try:
                dfc[target_col] = pd.to_numeric(dfc[target_col])
            except Exception:
                pass
    # drop rows with missing target
    dfc = dfc.loc[dfc[target_col].notna()].copy()
    y = dfc[target_col]
    X = dfc.drop(columns=[target_col])
    # drop id columns if present
    for col in list(X.columns):
        if col.lower() in ('id','patient_id','pid'):
            X.drop(columns=[col], inplace=True)
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y

X_df, y_ser = clean_and_prepare(df_raw, tgt_col)

# Basic dataset info
st.subheader("Dataset summary")
c1, c2, c3 = st.columns(3)
c1.metric("Rows", f"{X_df.shape[0]}")
c2.metric("Features", f"{X_df.shape[1]}")
c3.metric("Target column", f"{tgt_col}")

# Missing values
missing_counts = X_df.isna().sum()
if missing_counts.sum() > 0:
    with st.expander("Missing values (click to view)"):
        st.dataframe(missing_counts[missing_counts > 0])

# Class balance
st.subheader("Target class distribution")
class_counts = y_ser.value_counts().to_dict()
fig_pie = px.pie(values=list(class_counts.values()), names=list(class_counts.keys()), title="Class distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# Feature types detected
numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
cat_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()
st.markdown(f"**Detected features:** {len(numeric_cols)} numeric, {len(cat_cols)} categorical.")
if len(cat_cols) > 0:
    with st.expander("Categorical columns detected"):
        st.write(cat_cols)

# -----------------------------------------------------------------------------
# Preprocessor builder (OneHotEncoder compatibility)
# -----------------------------------------------------------------------------
def build_preprocessor(Xdf):
    num_cols = Xdf.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols_local = Xdf.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers = []
    if len(cat_cols_local) > 0:
        # create compat OneHotEncoder
        ohe = make_onehot_encoder_compat(handle_unknown='ignore')
        transformers.append(('cat', ohe, cat_cols_local))
    if len(num_cols) > 0:
        transformers.append(('num', StandardScaler(), num_cols))
    col_trans = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)
    return col_trans, num_cols, cat_cols_local

preproc, num_cols, cat_cols = build_preprocessor(X_df)

# -----------------------------------------------------------------------------
# Placeholders for live UI updates while running BAT
# -----------------------------------------------------------------------------
conv_placeholder = st.empty()
progress_placeholder = st.sidebar.empty()
status_placeholder = st.sidebar.empty()
metrics_placeholder = st.empty()

trained_clf = None
trained_preproc = None
selected_indices = []
transformed_feature_names = []
convergence_curve = []

# -----------------------------------------------------------------------------
# Train (Run BAT & Train) — triggered by sidebar button
# -----------------------------------------------------------------------------
if run_button:
    status_placeholder.info("Starting preprocessing and BAT feature selection...")

    # Simple missing handling: drop any rows with missing features for this run
    work_df = pd.concat([X_df, y_ser], axis=1).dropna().reset_index(drop=True)
    if work_df.shape[0] < 10:
        st.warning("After dropping missing rows there are fewer than 10 samples — training may be unreliable.")

    X_clean = work_df.drop(columns=[tgt_col])
    y_clean = work_df[tgt_col]
    # rebuild preproc on cleaned X
    preproc, num_cols, cat_cols = build_preprocessor(X_clean)
    try:
        preproc.fit(X_clean)
    except Exception as e:
        st.error(f"Preprocessor fit failed: {e}")
        st.stop()

    try:
        # get transformed feature names (compat)
        try:
            transformed_feature_names = list(preproc.get_feature_names_out())
        except Exception:
            # manual construction fallback
            transformed_feature_names = []
            if cat_cols:
                ohe = preproc.named_transformers_['cat']
                cats = ohe.categories_
                for col, levels in zip(cat_cols, cats):
                    for lvl in levels:
                        transformed_feature_names.append(f"{col}__{lvl}")
            if num_cols:
                transformed_feature_names.extend(num_cols)
    except Exception:
        transformed_feature_names = []

    # transform entire dataset
    try:
        X_trans = preproc.transform(X_clean)
        if isinstance(X_trans, np.ndarray):
            X_trans_arr = X_trans
        else:
            # sparse matrix fallback
            X_trans_arr = X_trans.toarray()
    except Exception as e:
        st.error(f"Feature transformation failed: {e}")
        st.stop()

    # ensure labels numeric
    try:
        y_vals = pd.to_numeric(y_clean).values.astype(int)
    except Exception:
        y_vals = y_clean.map(lambda x: 1 if str(x).lower().startswith('m') else 0).values.astype(int)

    # progress updater closure to update UI
    def ui_progress_updater(gen_index, best_score, conv_so_far, max_gen_local):
        # progress
        frac = int((gen_index+1)/max_gen_local * 100)
        progress_placeholder.progress(frac)
        # live convergence plot
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=conv_so_far, mode='lines+markers', name='best_cv'))
        fig.update_layout(title=f"BAT Convergence (gen {gen_index+1}/{max_gen_local})", xaxis_title="Generation", yaxis_title="Best CV Score", height=400)
        conv_placeholder.plotly_chart(fig, use_container_width=True)
        status_placeholder.info(f"Running BAT: generation {gen_index+1}/{max_gen_local}")

    # Run BAT
    try:
        selected_indices, convergence_curve = bat_feature_selection(
            features=X_trans_arr,
            labels=y_vals,
            num_bats=num_bats,
            max_gen=max_gen,
            loudness=loudness,
            pulse_rate=pulse_rate,
            progress_updater=ui_progress_updater,
            sleep_per_gen=sleep_per_gen
        )
    except Exception as e:
        st.error(f"BAT failed: {e}")
        st.stop()

    # clear progress
    progress_placeholder.empty()
    status_placeholder.success("BAT finished.")

    if len(selected_indices) == 0:
        st.warning("BAT selected no features. Falling back to mutual information selection (top features).")
        mi = mutual_info_classif(X_trans_arr, y_vals)
        top_k = min(10, X_trans_arr.shape[1])
        selected_indices = list(np.argsort(mi)[-top_k:])
        convergence_curve = convergence_curve if len(convergence_curve) > 0 else [0.0]

    selected_names = [transformed_feature_names[i] for i in selected_indices] if transformed_feature_names else [f"f{i}" for i in selected_indices]
    st.success(f"Selected {len(selected_indices)} features.")
    st.write(selected_names)

    # show final convergence plot
    try:
        fig_final = px.line(y=convergence_curve, labels={'value':'Best CV Score','index':'Generation'}, title="BAT Convergence (final)")
        fig_final.update_traces(mode='lines+markers')
        conv_placeholder.plotly_chart(fig_final, use_container_width=True)
    except Exception:
        pass

    # Train GaussianNB on selected features
    try:
        X_sel = X_trans_arr[:, selected_indices]
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y_vals, test_size=0.20, random_state=42, stratify=y_vals)
        clf = GaussianNB()
        clf.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # metrics
    y_pred = clf.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc = float((y_pred == y_test).mean())
    st.subheader("Model evaluation (hold-out test set)")
    metrics_placeholder.metric("Test accuracy", f"{acc:.3f}")
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], colorscale="Blues"))
    cm_fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)

    # ROC plot if available
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

    # Feature importance (mutual info) for selected features
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
        # metadata for patient form
        metadata = {
            "original_columns": X_df.columns.tolist(),
            "numeric_cols": num_cols,
            "categorical_cols": cat_cols,
            "numeric_stats": {c: {'min': float(X_df[c].min()), 'max': float(X_df[c].max()), 'mean': float(X_df[c].mean())} for c in num_cols}
        }
        with open("models/metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        st.sidebar.success("Saved model and artifacts to ./models/")

    # store in-memory for immediate prediction
    trained_clf = clf
    trained_preproc = preproc
    transformed_feature_names_local = transformed_feature_names
    selected_indices_local = selected_indices
    convergence_curve_local = convergence_curve

# -----------------------------------------------------------------------------
# Patient Form and Prediction
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("🧾 Patient Prediction Form")

# Determine model / preprocessor to use for prediction:
model_to_use = None
preproc_to_use = None
selected_idx_for_prediction = []

# If we trained above in-session, use that
if 'trained_clf' in locals() and trained_clf is not None:
    model_to_use = trained_clf
    preproc_to_use = trained_preproc
    selected_idx_for_prediction = selected_indices_local if 'selected_indices_local' in locals() else []
# else try to load saved model
else:
    try:
        if Path("models/model.joblib").exists():
            model_to_use = load("models/model.joblib")
        if Path("models/preprocessor.joblib").exists():
            preproc_to_use = load("models/preprocessor.joblib")
        if Path("models/selected_indices.npy").exists():
            selected_idx_for_prediction = list(np.load("models/selected_indices.npy").astype(int))
        if Path("models/feature_names.json").exists():
            with open("models/feature_names.json", 'r') as f:
                transformed_feature_names_local = json.load(f)
    except Exception:
        model_to_use = None
        preproc_to_use = None
        selected_idx_for_prediction = []

if model_to_use is None:
    st.info("Model not yet available for prediction. Run 'Run BAT & Train' in the sidebar or save a model to ./models/ and reload the app.")
else:
    st.write("Fill in patient details (fields inferred from the dataset). Only the features used by the model were considered during training; the app will preprocess inputs the same way.")
    # Build inputs from original X_df columns (user-friendly)
    user_inputs = {}
    with st.form("patient_form"):
        for col in X_df.columns:
            if col in X_df.select_dtypes(include=[np.number]).columns:
                vmin = float(X_df[col].min())
                vmax = float(X_df[col].max())
                vmean = float(X_df[col].mean())
                # choose number_input for generality
                user_inputs[col] = st.number_input(col, min_value=vmin, max_value=vmax, value=vmean, format="%.5f")
            else:
                uniques = X_df[col].dropna().unique().tolist()
                if len(uniques) <= 12:
                    user_inputs[col] = st.selectbox(col, options=uniques)
                else:
                    user_inputs[col] = st.text_input(col, value=str(uniques[0]))
        submitted = st.form_submit_button("Predict Patient")

    if submitted:
        # Build one-row DataFrame in original order
        sample_df = pd.DataFrame([user_inputs])[X_df.columns]
        try:
            sample_trans = preproc_to_use.transform(sample_df)
            if not isinstance(sample_trans, np.ndarray):
                sample_trans = sample_trans.toarray()
        except Exception as e:
            st.error(f"Preprocessing patient input failed: {e}")
            sample_trans = None

        if sample_trans is not None:
            try:
                sample_sel = sample_trans[:, selected_idx_for_prediction]
            except Exception as e:
                st.error(f"Selecting trained features from preprocessed input failed: {e}")
                sample_sel = None

            if sample_sel is not None:
                try:
                    pred = model_to_use.predict(sample_sel)[0]
                    prob = model_to_use.predict_proba(sample_sel)[0][int(pred)] if hasattr(model_to_use, "predict_proba") else None
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

                    # show input summary and allow download
                    outdf = sample_df.copy()
                    outdf['prediction'] = int(pred)
                    outdf['confidence'] = float(prob) if prob is not None else None
                    st.subheader("Patient input summary")
                    st.dataframe(outdf.T)
                    st.download_button("Download patient result (CSV)", outdf.to_csv(index=False).encode('utf-8'), "patient_result.csv", "text/csv")

# -----------------------------------------------------------------------------
# Additional analysis charts (always available)
# -----------------------------------------------------------------------------
st.markdown("---")
st.header("Exploratory charts & write-up")

# Correlation heatmap for numeric features
if len(numeric_cols) > 1:
    st.subheader("Correlation heatmap (numeric features)")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(X_df[numeric_cols].corr(), annot=False, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# Show top features used if available
if 'selected_names' in locals() and selected_names:
    st.subheader("Selected transformed features (sample list)")
    st.write(selected_names[:30])

# Final write-up / methodology
st.markdown("""
## Write-up — methodology & interpretation

**Goal:** Build a lightweight academic assistant that predicts probable breast cancer presence using a dataset of clinical/demographic or tumor-derived features.

**Pipeline summary:**
1. **Upload/choose dataset**: The app accepts any CSV; use the sidebar to select the label/target column (0/1 or M/B).
2. **Preprocessing**: Categorical columns are One-Hot Encoded (compatibly across scikit-learn versions). Numeric features are standardized.
3. **Feature selection (BAT)**: A binary variant of the BAT algorithm explores subsets of transformed features and optimizes cross-validated accuracy using Gaussian Naive Bayes as an internal evaluator.
4. **Training**: The final classifier is trained on the BAT-selected features and evaluated on a hold-out test set. The app displays accuracy, confusion matrix, ROC and a simple mutual-information-based feature importance for the chosen subset.
5. **Prediction**: The patient form uses original feature inputs, which are preprocessed with the exact same pipeline and passed through the trained model. Output includes a probability and a textual explanation.

**Important limitations & disclaimers:**
- This is an academic demonstrator only — NOT a clinical diagnostic tool.
- Model performance and fairness depend heavily on the quality, size, and representativeness of the uploaded dataset.
- Always consult qualified medical practitioners for real clinical decisions.
""")

# -----------------------------------------------------------------------------
# End
# -----------------------------------------------------------------------------
