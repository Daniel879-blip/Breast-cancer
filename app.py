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

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.feature_selection import mutual_info_classif
import time

# -------------------------
# Helper functions
# -------------------------

def load_default_dataset():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    df = pd.concat([d.data, d.target.rename("target")], axis=1)
    return df

def clean_dataset(df):
    """Cleans uploaded or default dataset and returns X (features df) and y (Series)."""
    df = df.copy()
    df = df.dropna(how='all')  # drop fully empty rows
    # If diagnosis is present as M/B map it
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].map({'M': 1, 'B': 0})
    # If target not present, assume last column is target
    if 'target' not in df.columns and 'diagnosis' not in df.columns:
        df = df.rename(columns={df.columns[-1]: 'target'})
    # At this point prefer 'diagnosis' over 'target'
    if 'diagnosis' in df.columns:
        y = df['diagnosis'].astype(float)
        X = df.drop(columns=['diagnosis'])
    else:
        y = df['target'].astype(float)
        X = df.drop(columns=['target'])
    # remove non-numeric columns (id, text)
    X = X.select_dtypes(include=[np.number])
    # align X and y indexes
    X = X.loc[y.index].reset_index(drop=True)
    y = y.reset_index(drop=True)
    return X, y

# Simple BAT binary feature selector (internal)
def objective_cv_score(features, labels, mask):
    """Return cross-validated score for selected mask (list/array of 0/1)."""
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

def bat_feature_selection(features, labels, num_bats=20, max_gen=40, loudness=0.5, pulse_rate=0.5, progress_callback=None):
    """
    Binary BAT implementation returning selected feature indices and convergence list.
    - features: numpy array (n_samples, n_features)
    - labels: numpy array (n_samples,)
    - returns: list(selected_indices), convergence_scores
    """
    n_feats = features.shape[1]
    # positions: 0/1
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
            # update velocity (binary idea)
            velocities[i] = velocities[i] + freq * (positions[i] ^ best_pos)
            prob = 1.0 / (1.0 + np.exp(-velocities[i]))
            new_pos = (np.random.rand(n_feats) < prob).astype(int)

            # local random walk (pulse)
            if np.random.rand() > pulse_rate:
                flip_mask = (np.random.rand(n_feats) < 0.05).astype(int)
                tmp = best_pos.copy()
                tmp[flip_mask == 1] = 1 - tmp[flip_mask == 1]
                new_pos = tmp

            new_score = objective_cv_score(features, labels, new_pos)
            # acceptance
            if (new_score > fitness[i]) and (np.random.rand() < loudness):
                positions[i] = new_pos
                fitness[i] = new_score
            if new_score > best_score:
                best_pos = new_pos.copy()
                best_score = new_score
        convergence.append(best_score)
        # progress callback for UI animation
        if progress_callback is not None:
            try:
                progress_callback(gen, best_score, convergence.copy(), max_gen)
            except Exception:
                pass

    selected_indices = [int(idx) for idx, bit in enumerate(best_pos) if bit == 1]
    return selected_indices, convergence

def explain_prediction(prob, label):
    """Return human-readable explanation string for the prediction."""
    pct = prob * 100
    if label == 1:
        text = (
            f"The model predicts a **POSITIVE** result (likely presence of malignant tumor) with confidence **{pct:.1f}%**.\n\n"
            "This means the model found features in the provided measurements that resemble malignant cases in the training data.\n\n"
            "Recommended next steps:\n"
            "1. Confirm the result with clinical exams (ultrasound, mammography) and consult an oncologist.\n"
            "2. Consider biopsy for definitive diagnosis.\n"
            "3. Share these results and model probabilities with a medical professional; this model is an aid, not a final diagnosis."
        )
    else:
        text = (
            f"The model predicts a **NEGATIVE** result (likely benign / no malignant tumor) with confidence **{pct:.1f}%**.\n\n"
            "This suggests the measurements are more similar to benign cases in the training data.\n\n"
            "Recommended next steps:\n"
            "1. Continue routine screening as advised by your physician.\n"
            "2. If symptoms persist or clinical concern remains, follow up with further testing.\n"
            "3. Use this result as a helpful indicator — consult a medical professional for clinical decisions."
        )
    return text

# -------------------------
# Streamlit App UI
# -------------------------
st.set_page_config(page_title="Breast Cancer Predictor — BAT + Naive Bayes", layout="wide")
st.title("🧬 Breast Cancer Predictor — BAT (feature selection) + Naive Bayes")
st.markdown("**Note:** This tool is an academic demonstrator. Results are for informational purposes only and not a medical diagnosis.")

# Sidebar controls
st.sidebar.header("Configuration & Data")
uploaded = st.sidebar.file_uploader("Upload CSV dataset (optional). Target column should be 'target' or 'diagnosis' (M/B).", type=["csv"])
use_default = st.sidebar.checkbox("Use default sklearn breast cancer dataset", value=True if uploaded is None else False)
st.sidebar.markdown("---")

st.sidebar.subheader("Feature Selection")
feat_method = st.sidebar.selectbox("Method", ["BAT (default)"], index=0)
num_bats = st.sidebar.slider("BAT: number of bats (population)", 6, 80, 24)
max_gen = st.sidebar.slider("BAT: generations", 5, 100, 40)
loudness = st.sidebar.slider("BAT: loudness (A)", 0.1, 1.0, 0.5, 0.05)
pulse_rate = st.sidebar.slider("BAT: pulse rate (r)", 0.0, 1.0, 0.5, 0.05)
st.sidebar.markdown("---")

st.sidebar.subheader("Classifier")
clf_choice = st.sidebar.selectbox("Classifier", ["Naive Bayes (Gaussian)"], index=0)  # placeholder for extension
st.sidebar.markdown("---")

run_button = st.sidebar.button("Run feature selection & train model")

# Option: quick manual patient form (independent of dataset)
st.sidebar.subheader("Manual Patient Quick Test")
quick_mode = st.sidebar.checkbox("Enable quick manual patient form (use default 5 features)", value=False)

# -------------------------
# Load and prepare dataset
# -------------------------
if uploaded is not None:
    try:
        raw_df = pd.read_csv(uploaded)
    except Exception as e:
        st.sidebar.error(f"Could not read uploaded file: {e}")
        st.stop()
elif use_default:
    raw_df = load_default_dataset()
else:
    st.sidebar.warning("No dataset uploaded and default not selected. Using default dataset.")
    raw_df = load_default_dataset()

# show preview
with st.expander("Dataset preview (first 5 rows)"):
    st.dataframe(raw_df.head())

# Clean
X_df, y = clean_dataset(raw_df)
st.sidebar.write(f"Features (numeric): {X_df.shape[1]}  |  Samples: {X_df.shape[0]}")

# If too few samples, warn
if X_df.shape[0] < 10:
    st.warning("Dataset has very few rows — this may produce unreliable results.")

# Show class distribution
class_counts = y.value_counts().to_dict()
fig_dist = px.pie(names=list(class_counts.keys()), values=list(class_counts.values()), title="Class distribution (target)")
st.plotly_chart(fig_dist, use_container_width=True)

# -------------------------
# Run BAT and train (on button)
# -------------------------
# placeholders for progress & charts
conv_placeholder = st.empty()
status_placeholder = st.sidebar.empty()
metric_placeholder = st.sidebar.empty()
progress_placeholder = st.sidebar.empty()

selected_indices = None
convergence = None
model = None
X_selected_df = None
train_results = {}

def progress_cb(gen_index, best_score, curve_so_far, max_gen_local):
    # update progress bar and live chart
    frac = int((gen_index+1)/max_gen_local * 100)
    progress_placeholder.progress(frac)
    metric_placeholder.metric(label=f"Best CV acc (gen {gen_index+1}/{max_gen_local})", value=f"{best_score:.3f}")
    # draw small plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=curve_so_far, mode='lines+markers', name='best'))
    fig.update_layout(title="BAT Convergence (live)", xaxis_title="Generation", yaxis_title="CV Accuracy", height=350)
    conv_placeholder.plotly_chart(fig, use_container_width=True)
    status_placeholder.info(f"Running BAT — generation {gen_index+1}/{max_gen_local}")

if run_button:
    status_placeholder.info("Starting BAT feature selection...")
    features_arr = X_df.values
    labels_arr = y.values.astype(int)
    # Run BAT
    selected_indices, convergence = bat_feature_selection(
        features=features_arr,
        labels=labels_arr,
        num_bats=num_bats,
        max_gen=max_gen,
        loudness=loudness,
        pulse_rate=pulse_rate,
        progress_callback=progress_cb
    )
    status_placeholder.success("BAT finished.")
    progress_placeholder.empty()
    metric_placeholder.empty()
    # If no features selected, warn
    if not selected_indices:
        st.error("BAT selected zero features. Try increasing generations / bats or check dataset.")
    else:
        selected_feature_names = list(X_df.columns[selected_indices])
        st.success(f"Selected {len(selected_feature_names)} features: {selected_feature_names}")
        # final convergence plot
        fig_final = px.line(y=convergence, labels={'value':'Best CV Accuracy', 'index':'Generation'}, title="BAT Convergence (final)")
        fig_final.update_traces(mode='lines+markers')
        conv_placeholder.plotly_chart(fig_final, use_container_width=True)

        # Train final model with selected features
        X_selected_df = X_df.iloc[:, selected_indices]
        X_train, X_test, y_train, y_test = train_test_split(X_selected_df, y, test_size=0.20, random_state=42, stratify=y)
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1] if hasattr(model, "predict_proba") else None

        # metrics
        acc = float((y_pred == y_test).mean())
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        conf_mat = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob) if y_prob is not None else (None, None, None)
        roc_auc = float(auc(fpr, tpr)) if y_prob is not None else None

        train_results = {
            "accuracy": acc,
            "report": report,
            "confusion_matrix": conf_mat,
            "roc": (fpr, tpr, roc_auc),
            "selected_features": selected_feature_names,
            "X_selected_df": X_selected_df
        }

        # show performance
        st.subheader("Model performance on held-out test set")
        st.metric("Test accuracy", f"{acc:.3f}")
        st.write("**Classification report (precision / recall / f1)**")
        st.dataframe(pd.DataFrame(report).transpose().style.background_gradient(cmap='Blues'))

        # confusion matrix plot
        cm = conf_mat
        cm_fig = go.Figure(data=go.Heatmap(z=cm, x=["Pred 0","Pred 1"], y=["True 0","True 1"], colorscale="Blues"))
        cm_fig.update_layout(title="Confusion Matrix")
        st.plotly_chart(cm_fig, use_container_width=True)

        # ROC
        if roc_auc is not None:
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC={roc_auc:.3f}"))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='chance'))
            roc_fig.update_layout(title="ROC Curve", xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            st.plotly_chart(roc_fig, use_container_width=True)

        # feature ranking (mutual info) on selected features
        if X_selected_df.shape[1] > 0:
            mi = mutual_info_classif(X_selected_df.values, y.values)
            mi_df = pd.DataFrame({"feature": X_selected_df.columns, "mi": mi}).sort_values("mi", ascending=False)
            st.subheader("Feature importance (mutual information)")
            st.dataframe(mi_df.style.background_gradient(cmap='Oranges'))

        # allow download of selected dataset used for training (with target)
        df_sel = X_selected_df.copy()
        df_sel['target'] = y
        csv_bytes = df_sel.to_csv(index=False).encode('utf-8')
        st.download_button("Download selected features dataset (CSV)", csv_bytes, "selected_features_data.csv", "text/csv")

# -------------------------
# Patient prediction form
# -------------------------
st.markdown("---")
st.header("🧾 Patient Prediction Form")

# Two modes: quick manual (default), or full patient form based on selected features
if quick_mode:
    st.subheader("Quick manual test (5 commonly relevant features)")
    # choose up to 5 features from dataset if available; otherwise use first 5 columns
    quick_feats = list(X_df.columns[:5])
    quick_inputs = []
    for feat in quick_feats:
        vmin = float(X_df[feat].min())
        vmax = float(X_df[feat].max())
        vmean = float(X_df[feat].mean())
        quick_inputs.append(st.number_input(f"{feat}", min_value=vmin, max_value=vmax, value=vmean, format="%.5f"))
    if st.button("Predict (quick)"):
        # if model not trained, train using all features as fallback
        if model is None:
            # train on all numeric features (fallback)
            X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
            model = GaussianNB()
            model.fit(X_train, y_train)
        sample = np.array(quick_inputs).reshape(1, -1)
        # If model was trained on different features, try to match - fallback will use provided order if same columns
        try:
            pred = model.predict(sample)[0]
            prob = model.predict_proba(sample)[0][int(pred)] if hasattr(model, "predict_proba") else None
        except Exception:
            st.error("Model input shape doesn't match the quick form shape. Train model with BAT first or use dataset with matching columns.")
            pred = None
            prob = None
        if pred is not None:
            label = "Positive (Malignant)" if pred == 1 else "Negative (Benign)"
            st.markdown(f"### Result: **{label}**")
            if prob is not None:
                st.markdown(f"Confidence: **{prob:.2%}**")
            st.info(explain_prediction(prob if prob is not None else 0, int(pred)))
else:
    # Full patient form based on selected features if available, else use top features
    if (train_results and train_results.get("selected_features")):
        feat_list = train_results["selected_features"]
    else:
        # fallback: use first N features
        feat_list = list(X_df.columns[:6])

    st.subheader("Enter patient measurements for the features below")
    patient_vals = []
    patient_inputs = {}
    for f in feat_list:
        vmin = float(X_df[f].min())
        vmax = float(X_df[f].max())
        vmean = float(X_df[f].mean())
        val = st.number_input(f"{f}", min_value=vmin, max_value=vmax, value=vmean, format="%.5f")
        patient_vals.append(val)
        patient_inputs[f] = val

    if st.button("Predict Patient"):
        if model is None:
            st.warning("Model not trained yet. Training a fallback model on all numeric features...")
            X_train, X_test, y_train, y_test = train_test_split(X_df, y, test_size=0.2, random_state=42, stratify=y)
            model = GaussianNB()
            model.fit(X_train, y_train)
        # prepare sample respecting training feature order
        try:
            # If model was trained on selected_features, use that order
            if train_results and train_results.get("X_selected_df") is not None:
                cols_order = list(train_results["X_selected_df"].columns)
                sample_arr = np.array([patient_inputs[c] for c in cols_order]).reshape(1, -1)
            else:
                # fallback: try using first len(patient_vals) columns from original dataset
                sample_arr = np.array(patient_vals).reshape(1, -1)
            pred = model.predict(sample_arr)[0]
            prob = float(model.predict_proba(sample_arr)[0][int(pred)]) if hasattr(model, "predict_proba") else None
        except Exception as e:
            st.error(f"Error making prediction: {e}")
            pred = None
            prob = None

        if pred is not None:
            label = "Positive (Malignant)" if pred == 1 else "Negative (Benign)"
            st.markdown(f"## Result: **{label}**")
            if prob is not None:
                st.metric("Confidence", f"{prob:.2%}")
            # Explanation + recommended actions
            st.markdown("### Explanation")
            st.write(explain_prediction(prob if prob is not None else 0, int(pred)))
            # Create small report to download
            report_df = pd.DataFrame([patient_inputs])
            report_df['prediction'] = int(pred)
            report_df['confidence'] = prob if prob is not None else np.nan
            csv_report = report_df.to_csv(index=False).encode('utf-8')
            st.download_button("Download patient result (CSV)", csv_report, "patient_result.csv", "text/csv")

# -------------------------
# Footer: guidance & help
# -------------------------
st.markdown("---")
st.header("About this tool")
st.markdown("""
This application demonstrates an academic pipeline for **feature selection using a BAT-inspired algorithm** and classification using **Gaussian Naive Bayes**.
- **BAT** (bio-inspired) searches for compact feature subsets that maximize cross-validated accuracy.
- **Naive Bayes** is a simple probabilistic classifier; it's fast and interpretable but has independence assumptions.

**Important:** This model should NOT be used as a substitute for clinical judgment. Always consult qualified medical professionals for diagnostic decisions.
""")
