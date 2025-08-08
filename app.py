"""
Advanced Streamlit App: Breast Cancer Prediction (BAT feature selection + Naive Bayes)

How to use:
- Put this file (app.py) in your project root.
- Optional: place a CSV dataset in data/ or upload via sidebar. CSV should have features and a final target column
  named 'target' OR 'diagnosis' (M/B). If target is M/B it will be mapped automatically.
- Run: streamlit run app.py

Notes:
- Uses an internal BAT implementation (binary feature selector).
- Classifier choices show only Naive Bayes (others can be added).
- Plotly provides interactive charts (make sure 'plotly' is installed).
"""

# app.py
"""
Advanced Streamlit App: Breast Cancer Prediction (Train from Sidebar)
Features:
- Upload arbitrary CSV and pick the target column
- BAT feature selection (adjustable parameters in sidebar)
- Train Naive Bayes from the sidebar (BAT -> train)
- Dynamic patient form (only uses features chosen by BAT)
- Interactive charts: convergence (live), ROC, confusion matrix, distribution, corr heatmap
- Explanations and downloadable patient report
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

# sklearn
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
from sklearn.feature_selection import mutual_info_classif

# persistence (optional)
from joblib import dump, load

# ---------- Utility / BAT implementation ----------
def objective_cv_score(features, labels, mask):
    """
    features: np.array (n_samples, n_features)
    labels: np.array (n_samples,)
    mask: binary 1/0 list/array selecting features
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

def bat_feature_selection(features, labels, num_bats=20, max_gen=40, loudness=0.5, pulse_rate=0.5,
                          progress_updater=None, sleep_per_gen=0.0):
    """
    Binary BAT algorithm with optional progress_updater callback.
    progress_updater(gen_index, best_score, convergence_so_far, max_gen) -> None
    """
    n_feats = features.shape[1]
    # init
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

            # local flip
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

        # callback for UI updates
        if progress_updater is not None:
            try:
                progress_updater(gen, best_score, convergence.copy(), max_gen)
            except Exception:
                pass

        if sleep_per_gen > 0:
            time.sleep(sleep_per_gen)

    selected_indices = [int(idx) for idx, bit in enumerate(best_pos) if bit == 1]
    return selected_indices, convergence

# ---------- App UI ----------
st.set_page_config(page_title="Breast Cancer Predictor (trainable)", layout="wide", initial_sidebar_state="expanded")
st.title("🧬 Breast Cancer Predictor — Train from Sidebar (BAT + Naive Bayes)")
st.markdown("**Instructions:** Upload a CSV (or use sample), choose the target column in the sidebar, adjust BAT parameters, then click **Run BAT & Train**. After training the patient form will appear and you can predict immediately.")

# ---- Sidebar: Dataset upload & controls ----
st.sidebar.header("1) Dataset & Target")
uploaded = st.sidebar.file_uploader("Upload CSV dataset (optional). If not provided, sample demo will be used.", type=["csv"])
use_example = st.sidebar.checkbox("Use example demo dataset (if no upload)", value=True if uploaded is None else False)

# After upload, let user pick target column
# we'll load the df (small preview) first
def safe_read_csv(uploaded_file):
    try:
        return pd.read_csv(uploaded_file)
    except Exception as e:
        st.sidebar.error(f"Could not read CSV: {e}")
        return None

if uploaded is not None:
    df_raw = safe_read_csv(uploaded)
elif use_example:
    # build a small demo dataset (based on sklearn) but add age/sex/family_history style fields if helpful
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    df_temp = pd.concat([d.data, d.target.rename("target")], axis=1)
    # add synthetic demographics to demo for user-friendly patient form
    rng = np.random.default_rng(seed=42)
    n = df_temp.shape[0]
    df_temp['age'] = rng.integers(30, 85, size=n)
    # all female mostly — include 'sex' for completeness (most female)
    df_temp['sex'] = rng.choice(['Female','Male'], size=n, p=[0.98, 0.02])
    df_temp['family_history'] = rng.choice(['Yes','No'], size=n, p=[0.12, 0.88])
    df_raw = df_temp.copy()
else:
    df_raw = None

if df_raw is None:
    st.sidebar.warning("No dataset loaded yet.")
    # still continue to render other controls but training disabled
else:
    st.sidebar.write("Preview (first 5 rows):")
    st.sidebar.dataframe(df_raw.head())

# target column selection
if df_raw is not None:
    st.sidebar.markdown("---")
    st.sidebar.subheader("Pick the target (label) column")
    tgt_col = st.sidebar.selectbox("Target column (choose the column that stores 0/1 or M/B)", options=list(df_raw.columns), index=len(df_raw.columns)-1)
else:
    tgt_col = None

# allow user to override mapping if target is 'M'/'B' or string
st.sidebar.markdown("---")
st.sidebar.subheader("2) BAT Parameters (Feature Selection)")
num_bats = st.sidebar.slider("Number of bats", 6, 80, 24)
max_gen = st.sidebar.slider("Generations", 5, 120, 40)
loudness = st.sidebar.slider("Loudness (A)", 0.05, 1.0, 0.5, step=0.05)
pulse_rate = st.sidebar.slider("Pulse rate (r)", 0.0, 1.0, 0.5, step=0.05)
sleep_per_gen = st.sidebar.slider("UI speed (sleep per generation, sec)", 0.0, 0.25, 0.02, step=0.01)

st.sidebar.markdown("---")
st.sidebar.subheader("3) Classifier")
clf_choice = st.sidebar.selectbox("Classifier (for now Naive Bayes only)", ["Naive Bayes (Gaussian)"])

st.sidebar.markdown("---")
st.sidebar.subheader("4) Train / Retrain")
run_button = st.sidebar.button("🔁 Run BAT & Train")

# Save artifacts option
st.sidebar.markdown("Save & export")
save_artifacts = st.sidebar.checkbox("Save trained model and metadata to ./models", value=False)

# ---- Main area: dataset checks and cleaning ----
if df_raw is None:
    st.info("Upload a CSV in the sidebar or enable the example dataset to proceed.")
    st.stop()

# Clean dataset and prepare X, y
def clean_dataset(df, target_col):
    df = df.copy()
    df = df.dropna(how='all')
    # if target col contains 'M'/'B' map it
    if df[target_col].dtype == object or df[target_col].dtype.name == 'category':
        # try to map common labels
        uniq = df[target_col].dropna().unique()
        uniq_lower = [str(x).lower() for x in uniq]
        if set(uniq_lower) >= set(['m','b']) or set(uniq_lower) >= set(['malignant','benign']):
            # map M/B or malignant/benign -> 1/0
            df[target_col] = df[target_col].map(lambda x: 1 if str(x).lower().startswith('m') else 0)
        else:
            # try to convert to numbers
            try:
                df[target_col] = pd.to_numeric(df[target_col])
            except Exception:
                # leave as-is, will fail later
                pass
    # drop rows with missing target
    df = df.loc[df[target_col].notna()].copy()
    y = df[target_col]
    X = df.drop(columns=[target_col])
    # drop pure-id-like columns
    for c in X.columns:
        if c.lower() in ('id','patient_id','pid'):
            X = X.drop(columns=[c])
    return X.reset_index(drop=True), y.reset_index(drop=True)

# Try cleaning
try:
    X_df, y_ser = clean_dataset(df_raw, tgt_col) if tgt_col is not None else (None, None)
except Exception as e:
    st.error(f"Error preparing dataset: {e}")
    st.stop()

# show quick dataset summary
st.subheader("Dataset summary")
col1, col2, col3 = st.columns([1,1,1])
col1.write(f"Rows: **{X_df.shape[0]}**")
col2.write(f"Columns (features): **{X_df.shape[1]}**")
col3.write(f"Target column: **{tgt_col}**")

# missing values summary
missing = X_df.isnull().sum()
if missing.sum() > 0:
    st.warning("Missing values detected in features. Rows with missing target were removed. Missing feature values will be handled (dropped or imputed) before training.")
    with st.expander("Missing values per column"):
        st.dataframe(missing[missing > 0])

# show class balance
st.write("Class distribution (target)")
class_counts = y_ser.value_counts().to_dict()
fig_pie = px.pie(names=list(class_counts.keys()), values=list(class_counts.values()), title="Class distribution")
st.plotly_chart(fig_pie, use_container_width=True)

# ---- Preprocessing: infer numeric vs categorical ----
numeric_cols = X_df.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X_df.select_dtypes(exclude=[np.number]).columns.tolist()

st.markdown("**Detected feature types**")
st.write(f"Numeric: {numeric_cols[:12]}{'...' if len(numeric_cols)>12 else ''}")
st.write(f"Categorical: {categorical_cols[:12]}{'...' if len(categorical_cols)>12 else ''}")

# Build preprocessing pipeline (fit later at training time)
def build_preprocessor(Xdf):
    num_cols = Xdf.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = Xdf.select_dtypes(exclude=[np.number]).columns.tolist()
    transformers = []
    if len(cat_cols) > 0:
        transformers.append(('cat', OneHotEncoder(handle_unknown='ignore', sparse=False), cat_cols))
    if len(num_cols) > 0:
        transformers.append(('num', StandardScaler(), num_cols))
    col_trans = ColumnTransformer(transformers, remainder='drop', sparse_threshold=0)
    return col_trans, num_cols, cat_cols

preproc, num_cols, cat_cols = build_preprocessor(X_df)

# placeholders for training-time visual updates
conv_placeholder = st.empty()
progress_placeholder = st.empty()
status_placeholder = st.empty()
metrics_placeholder = st.empty()

# trained artifacts (initialized None)
trained_model = None
selected_feature_indices = []
transformed_feature_names = []
convergence_curve = []

# training logic (run when sidebar button pressed)
if run_button:
    status_placeholder.info("Starting BAT feature selection and training...")
    # prepare features and labels (drop rows with NaNs in features or imputable)
    # simple handling: drop rows with any missing numeric/categorical (you can replace with imputation later)
    working_df = pd.concat([X_df, y_ser], axis=1).dropna().reset_index(drop=True)
    X_clean = working_df.drop(columns=[tgt_col]) if tgt_col in working_df.columns else working_df.drop(columns=[y_ser.name])
    y_clean = working_df[y_ser.name] if y_ser.name in working_df.columns else working_df.iloc[:, -1]

    # rebuild preprocessor to be safe
    preproc, num_cols, cat_cols = build_preprocessor(X_clean)

    # fit preprocessor to whole X
    try:
        preproc.fit(X_clean)
        # get transformed feature names
        try:
            transformed_feature_names = list(preproc.get_feature_names_out())
        except Exception:
            # compatibility fallback: construct manually
            transformed_feature_names = []
            if cat_cols:
                ohe = preproc.named_transformers_['cat']
                cats = ohe.categories_
                for col, levels in zip(cat_cols, cats):
                    for lvl in levels:
                        transformed_feature_names.append(f"{col}__{lvl}")
            if num_cols:
                transformed_feature_names.extend(num_cols)
    except Exception as e:
        st.error(f"Failed to fit preprocessor: {e}")
        st.stop()

    # transform dataset
    try:
        X_trans = preproc.transform(X_clean)
    except Exception as e:
        st.error(f"Failed to transform features: {e}")
        st.stop()

    # ensure labels numeric
    try:
        y_vals = pd.to_numeric(y_clean).values.astype(int)
    except Exception:
        # attempt M/B mapping
        y_vals = y_clean.map(lambda x: 1 if str(x).lower().startswith('m') else 0).values.astype(int)

    # progress updater for UI: show live convergence plot and progress bar
    gen_plot = conv_placeholder.empty()
    prog = progress_placeholder.empty()
    def progress_updater(gen_index, best_score, conv_so_far, max_gen_local):
        # progress bar
        frac = int((gen_index+1)/max_gen_local * 100)
        prog.progress(frac)
        # live convergence plot with Plotly
        fig = go.Figure()
        fig.add_trace(go.Scatter(y=conv_so_far, mode='lines+markers', name='best_cv'))
        fig.update_layout(title=f"BAT Convergence (gen {gen_index+1}/{max_gen_local})", xaxis_title="Generation", yaxis_title="Best CV Score")
        gen_plot.plotly_chart(fig, use_container_width=True)

    # run BAT
    try:
        selected_feature_indices, convergence_curve = bat_feature_selection(
            features=X_trans,
            labels=y_vals,
            num_bats=num_bats,
            max_gen=max_gen,
            loudness=loudness,
            pulse_rate=pulse_rate,
            progress_updater=progress_updater,
            sleep_per_gen=sleep_per_gen
        )
    except Exception as e:
        st.error(f"BAT failed: {e}")
        st.stop()

    prog.empty()
    status_placeholder.info("BAT finished. Selected features computed.")

    if len(selected_feature_indices) == 0:
        st.warning("BAT selected 0 features. Falling back to mutual information top features.")
        mi = mutual_info_classif(X_trans, y_vals)
        top_k = min(10, X_trans.shape[1])
        selected_feature_indices = list(np.argsort(mi)[-top_k:])
        convergence_curve = convergence_curve if len(convergence_curve) > 0 else [0.0]

    # show selected names
    selected_names = [transformed_feature_names[i] for i in selected_feature_indices]
    st.success(f"Selected {len(selected_names)} features.")
    st.write(selected_names)

    # final convergence plot
    try:
        fig_final = px.line(y=convergence_curve, labels={'value':'Best CV Score', 'index':'Generation'}, title="BAT Convergence (final)")
        fig_final.update_traces(mode='lines+markers')
        conv_placeholder.plotly_chart(fig_final, use_container_width=True)
    except Exception:
        pass

    # Train final classifier on selected features
    X_sel = X_trans[:, selected_feature_indices]
    X_train, X_test, y_train, y_test = train_test_split(X_sel, y_vals, test_size=0.20, random_state=42, stratify=y_vals)
    clf = GaussianNB()
    try:
        clf.fit(X_train, y_train)
    except Exception as e:
        st.error(f"Training failed: {e}")
        st.stop()

    # compute metrics and show
    y_pred = clf.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    acc = float((y_pred == y_test).mean())
    status_placeholder.success("Training complete.")
    metrics_placeholder.metric("Test accuracy", f"{acc:.3f}")

    # show classification report
    st.subheader("Model performance")
    st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap="Blues"))

    # confusion matrix (Plotly heatmap)
    cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], colorscale="Blues"))
    cm_fig.update_layout(title="Confusion Matrix")
    st.plotly_chart(cm_fig, use_container_width=True)

    # ROC if available (GaussianNB has predict_proba)
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

    # feature importance by mutual info on selected features
    try:
        if X_sel.shape[1] > 0:
            mi = mutual_info_classif(X_sel, y_vals)
            mi_df = pd.DataFrame({"feature": selected_names, "mi": mi}).sort_values("mi", ascending=False)
            st.subheader("Feature importance (mutual information on selected features)")
            st.dataframe(mi_df.style.background_gradient(cmap='OrRd'))
    except Exception:
        pass

    # Save artifacts if requested
    if save_artifacts:
        Path("models").mkdir(parents=True, exist_ok=True)
        dump(clf, "models/model.joblib")
        dump(preproc, "models/preprocessor.joblib")
        np.save("models/selected_indices.npy", np.array(selected_feature_indices, dtype=int))
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

    # store trained artifacts in memory for prediction below
    trained_model = clf
    transformed_feature_names_list = transformed_feature_names
    selected_feature_indices_list = selected_feature_indices
    convergence_curve_list = convergence_curve

# ---- Patient prediction form (only after training OR if fallback model available) ----
st.markdown("---")
st.header("🧾 Patient Prediction Form")

# Decide which model and selected features to use:
model_for_prediction = None
sel_names_for_pred = []
preprocessor_for_pred = None
selected_idx_for_pred = []

# if we just trained in this session:
if 'trained_model' in locals() and trained_model is not None:
    model_for_prediction = trained_model
    selected_idx_for_pred = selected_feature_indices
    transformed_feature_names_list = transformed_feature_names
    sel_names_for_pred = [transformed_feature_names_list[i] for i in selected_idx_for_pred] if len(selected_idx_for_pred)>0 else []
    preprocessor_for_pred = preproc
# else, if there are saved artifacts in ./models, try to load them so you can predict without retraining
else:
    model_path = Path("models/model.joblib")
    preproc_path = Path("models/preprocessor.joblib")
    sel_idx_path = Path("models/selected_indices.npy")
    feat_names_path = Path("models/feature_names.json")
    if model_path.exists() and preproc_path.exists() and sel_idx_path.exists() and feat_names_path.exists():
        try:
            model_for_prediction = load(str(model_path))
            preprocessor_for_pred = load(str(preproc_path))
            selected_idx_for_pred = list(np.load(str(sel_idx_path)).astype(int))
            with open(str(feat_names_path), 'r') as f:
                transformed_feature_names_list = json.load(f)
            sel_names_for_pred = [transformed_feature_names_list[i] for i in selected_idx_for_pred]
            st.sidebar.info("Loaded pretrained model from ./models/")
        except Exception as e:
            st.sidebar.warning(f"Could not load saved model artifacts: {e}")

# If no model available yet, show message and allow quick train
if model_for_prediction is None:
    st.info("Model not trained yet. Use 'Run BAT & Train' in the sidebar to create a model (or save a model to ./models/).")
    # still show a light-weight quick test: let user train on all numeric features quickly
    if st.button("Quick train (all numeric features, no BAT)"):
        Xsmall = X_df.select_dtypes(include=[np.number]).copy()
        if Xsmall.shape[1] == 0:
            st.error("No numeric features to quick-train.")
        else:
            try:
                Xt = Xsmall.values
                yvals = pd.to_numeric(y_ser).values.astype(int)
                Xtr, Xte, ytr, yte = train_test_split(Xt, yvals, test_size=0.2, random_state=42, stratify=yvals)
                quick_clf = GaussianNB()
                quick_clf.fit(Xtr, ytr)
                model_for_prediction = quick_clf
                preprocessor_for_pred = None
                selected_idx_for_pred = list(range(Xsmall.shape[1]))
                transformed_feature_names_list = list(Xsmall.columns)
                sel_names_for_pred = transformed_feature_names_list
                st.success("Quick model trained on numeric features.")
            except Exception as e:
                st.error(f"Quick train failed: {e}")

# Build patient form using INPUT FEATURES (prefer original columns where possible)
if model_for_prediction is not None:
    # We want to build user-friendly inputs from original X_df columns.
    # Determine original columns required: if preprocessor available we have metadata in `num_cols` and `cat_cols`.
    st.write("Please provide patient values for the features selected by the model. Fields are generated for the *original* features used to construct the model.")
    # If we trained and have preproc and selected indices, we must map transformed feature names back to original features.
    orig_feature_inputs = {}
    if preprocessor_for_pred is not None and len(selected_idx_for_pred) > 0:
        # Find which original columns are involved in selected transformed features.
        # Simpler approach: show inputs for all original numeric + categorical columns (but highlight selected ones)
        # We collect original columns
        original_cols = list(X_df.columns)
        # Build UI inputs for all original cols (but it's fine — patient will fill essential ones)
        with st.form("patient_form"):
            user_values = {}
            for col in original_cols:
                if col in X_df.select_dtypes(include=[np.number]).columns:
                    vmin = float(X_df[col].min())
                    vmax = float(X_df[col].max())
                    vmean = float(X_df[col].mean())
                    user_values[col] = st.number_input(f"{col}", min_value=vmin, max_value=vmax, value=vmean, format="%.5f")
                else:
                    # categorical: show top categories or let user type
                    uniques = X_df[col].dropna().unique().tolist()
                    if len(uniques) <= 12:
                        user_values[col] = st.selectbox(f"{col}", options=uniques, index=0)
                    else:
                        user_values[col] = st.text_input(f"{col}", value=str(uniques[0]))
            submitted = st.form_submit_button("Predict Patient")
        if submitted:
            # Build a one-row DF using original_cols order
            sample_row = {c: user_values.get(c, X_df[c].mean() if c in X_df.select_dtypes(include=[np.number]).columns else X_df[c].dropna().unique()[0]) for c in original_cols}
            sample_df = pd.DataFrame([sample_row])[original_cols]
            # Preprocess if preprocessor exists
            try:
                sample_trans = preprocessor_for_pred.transform(sample_df)
            except Exception as e:
                st.error(f"Failed to preprocess patient input: {e}")
                sample_trans = None
            if sample_trans is not None:
                try:
                    sample_sel = sample_trans[:, selected_idx_for_pred]
                except Exception as e:
                    st.error(f"Failed to select model features from transformed input: {e}")
                    sample_sel = None
                if sample_sel is not None:
                    try:
                        pred = model_for_prediction.predict(sample_sel)[0]
                        prob = model_for_prediction.predict_proba(sample_sel)[0][int(pred)] if hasattr(model_for_prediction, "predict_proba") else None
                    except Exception as e:
                        st.error(f"Failed to predict: {e}")
                        pred = None
                        prob = None
                    if pred is not None:
                        label = "Positive (Malignant)" if int(pred) == 1 else "Negative (Benign)"
                        st.markdown(f"## Result: **{label}**")
                        if prob is not None:
                            st.metric("Model confidence", f"{prob:.2%}")
                        # explanation
                        if prob is None:
                            st.write("The model has returned a prediction. No probability available.")
                        else:
                            if pred == 1:
                                st.warning(explain_text := (
                                    f"The model predicts a POSITIVE result with confidence {prob:.2%}. "
                                    "This suggests the provided measurements are more similar to malignant cases from the training data. "
                                    "Recommended next steps: clinical evaluation, imaging, specialist consultation and biopsy if clinically indicated."
                                ))
                            else:
                                st.success(explain_text := (
                                    f"The model predicts a NEGATIVE result with confidence {prob:.2%}. "
                                    "This suggests the provided measurements are more similar to benign cases from the training data. "
                                    "Recommended next steps: routine follow-up and clinical review if symptoms persist."
                                ))
                        # show summary and allow download
                        outdf = sample_df.copy()
                        outdf['prediction'] = int(pred)
                        outdf['confidence'] = float(prob) if prob is not None else None
                        st.subheader("Patient input summary")
                        st.dataframe(outdf.T)
                        st.download_button("Download patient result (CSV)", outdf.to_csv(index=False).encode('utf-8'), "patient_result.csv", "text/csv")
    else:
        # No preprocessor or selected indices: fallback simple form based on numeric features only
        numeric_cols_local = X_df.select_dtypes(include=[np.number]).columns.tolist()
        if not numeric_cols_local:
            st.error("No numeric features available to build a quick patient form.")
        else:
            st.info("Model was trained on numeric-only quick fallback. Provide numeric inputs below.")
            with st.form("quick_form"):
                quick_vals = {}
                for c in numeric_cols_local:
                    vmin = float(X_df[c].min())
                    vmax = float(X_df[c].max())
                    vmean = float(X_df[c].mean())
                    quick_vals[c] = st.number_input(c, min_value=vmin, max_value=vmax, value=vmean, format="%.5f")
                submitted = st.form_submit_button("Predict (quick)")
            if submitted:
                try:
                    sample_arr = np.array([list(quick_vals.values())])
                    pred = model_for_prediction.predict(sample_arr)[0]
                    prob = model_for_prediction.predict_proba(sample_arr)[0][int(pred)] if hasattr(model_for_prediction, "predict_proba") else None
                except Exception as e:
                    st.error(f"Prediction failed: {e}")
                    pred = None
                    prob = None
                if pred is not None:
                    label = "Positive (Malignant)" if int(pred) == 1 else "Negative (Benign)"
                    st.markdown(f"## Result: **{label}**")
                    if prob is not None:
                        st.metric("Model confidence", f"{prob:.2%}")
                    # explanation
                    if pred == 1:
                        st.warning("Model indicates possible malignancy. Seek medical advice.")
                    else:
                        st.success("Model indicates likely benign. Continue routine care.")

st.markdown("---")
st.header("About & Write-up")
st.markdown("""
**Project purpose:**  
This application demonstrates an end-to-end workflow for building a breast cancer predictive assistant that uses a bio-inspired **BAT** algorithm for feature selection and **Gaussian Naive Bayes** for classification.

**What happens when you train:**
1. The uploaded dataset is cleaned and preprocessed (categorical features are one-hot encoded; numeric features are scaled).  
2. The BAT algorithm searches the transformed feature space for a compact subset of features that maximizes cross-validated accuracy.  
3. A Gaussian Naive Bayes classifier is trained on the selected features and evaluated on a hold-out test set.  
4. You get interactive charts (convergence, ROC, confusion matrix) and feature importance.

**Prediction & interpretation:**  
- The patient form uses the same original features used to build the model; inputs are mapped and preprocessed the same way as during training.  
- The model produces a prediction (Positive = likely malignant, Negative = likely benign) plus a confidence score.  
- The app provides a written explanation and recommended next steps.  
- **Important:** This is an academic demonstration. Predictions are probabilistic estimates based on the provided dataset and should not be used as a diagnostic device. Clinical decisions require healthcare professionals.

**Notes & troubleshooting:**  
- If your CSV does not contain a clear target column, use the dropdown in the sidebar to choose the correct label column.  
- If BAT returns zero features, the app falls back to mutual information selection.  
- If you want to persist models for quick reuse, check 'Save trained model...' in the sidebar to save artifacts to `./models/`.
""")

# helper: small explanation function (used earlier)
def explain_prediction(prob, label):
    pct = (prob*100) if prob is not None else None
    if label == 1:
        return f"Predicted POSITIVE (malignancy). Model confidence: {pct:.1f}% — recommend clinical follow-up and diagnostic confirmation."
    else:
        return f"Predicted NEGATIVE (benign). Model confidence: {pct:.1f}% — continue routine surveillance and consult clinician if symptoms persist."
