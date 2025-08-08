import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from utils.bat_algorithm import bat_algorithm  # ensure utils/bat_algorithm.py exists

st.set_page_config(page_title="Breast Cancer — BAT + Naive Bayes (Animated)", layout="wide", initial_sidebar_state="expanded")

# ---------------- Sidebar: Data + Settings ----------------
st.sidebar.title("⚙️ App Controls")
uploaded_file = st.sidebar.file_uploader("Upload CSV (last column is target or named 'diagnosis')", type=["csv"])
st.sidebar.markdown("### BAT Settings")
num_bats = st.sidebar.slider("Number of bats", 8, 60, 24)
max_gen = st.sidebar.slider("Generations", 5, 120, 40)
loudness = st.sidebar.slider("Loudness (A)", 0.1, 1.0, 0.6, 0.05)
pulse_rate = st.sidebar.slider("Pulse rate (r)", 0.0, 1.0, 0.5, 0.05)
run_bat_button = st.sidebar.button("🔁 Run BAT & Train")

# quick help
st.sidebar.markdown("---")
st.sidebar.markdown("**Tips:**\n- Upload CSV with numeric features and final column `target` or `diagnosis` (M/B or 1/0)\n- If no file uploaded, default dataset will be used")

# ---------------- Load & Clean Data ----------------
@st.cache_data
def load_default():
    from sklearn.datasets import load_breast_cancer
    d = load_breast_cancer(as_frame=True)
    df = pd.concat([d.data, d.target.rename("target")], axis=1)
    return df

def clean_and_prepare(df):
    # drop totally-empty rows
    df = df.dropna(how='all').copy()

    # If diagnosis column present as M/B convert to 1/0
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].map({'M':1, 'B':0}).astype(float)

    # If 'target' isn't present but last column probably contains label, assume it's target
    if 'target' not in df.columns and 'diagnosis' not in df.columns:
        df = df.rename(columns={df.columns[-1]: 'target'})

    # Now ensure target is numeric and no missing
    if 'diagnosis' in df.columns:
        y = df['diagnosis'].astype(float).dropna()
        X = df.drop(columns=['diagnosis'])
    else:
        y = df['target'].astype(float).dropna()
        X = df.drop(columns=['target'])

    # remove non-numeric columns (id, strings) from features
    X = X.select_dtypes(include=[np.number])
    # align X and y just in case dropped rows
    X = X.loc[y.index]

    return X.reset_index(drop=True), y.reset_index(drop=True)

if uploaded_file is not None:
    raw_df = pd.read_csv(uploaded_file)
else:
    raw_df = load_default()

st.title("🧬 Breast Cancer Prediction — BAT (animated) + Naive Bayes")

# show preview
with st.expander("Preview dataset (first 5 rows)"):
    st.dataframe(raw_df.head())

# quick dataset summary
st.sidebar.subheader("Dataset preview")
st.sidebar.write(f"Rows: {raw_df.shape[0]}")
st.sidebar.write(f"Columns: {raw_df.shape[1]}")

# Clean
X, y = clean_and_prepare(raw_df)
st.write(f"Using **{X.shape[1]}** numeric features for analysis.")

# ---------------- UI placeholders for animation ----------------
col1, col2 = st.columns([1, 1])

# convergence chart placeholder (Plotly)
conv_placeholder = col1.container()
# progress + metric placeholders sidebar
progress_placeholder = st.sidebar.empty()
metric_placeholder = st.sidebar.empty()
status_placeholder = st.sidebar.empty()

# training & results placeholders
train_status = st.empty()
results_container = st.container()

# show feature names (collapsible)
with st.expander("Selected feature candidates (all numeric features)"):
    st.write(list(X.columns))

# ---------------- Function to update animations ----------------
def progress_callback(gen_index, best_score, curve_so_far, max_gen_local):
    """
    Called by bat_algorithm each generation to update UI.
    """
    # update progress bar
    frac = int((gen_index+1)/max_gen_local * 100)
    progress_placeholder.progress(frac)

    # update live metric (best score with delta)
    metric_placeholder.metric(label=f"Best CV accuracy (gen {gen_index+1}/{max_gen_local})",
                              value=f"{best_score:.3f}",
                              delta=f"+{(best_score - (curve_so_far[-2] if len(curve_so_far)>1 else 0)):.3f}" if len(curve_so_far)>1 else "")

    # update convergence plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=curve_so_far, mode='lines+markers', name='best_accuracy'))
    fig.update_layout(title="BAT Convergence (live)", xaxis_title="Generation", yaxis_title="CV Accuracy", height=400)
    conv_placeholder.plotly_chart(fig, use_container_width=True)

    # update status text
    status_placeholder.info(f"Running BAT — generation {gen_index+1} / {max_gen_local}")

# ---------------- Run BAT (when user clicks) ----------------
if run_bat_button:
    st.sidebar.success("Starting BAT optimization...")

    # Convert to numpy arrays for algorithm
    features_arr = X.values
    labels_arr = y.values.astype(int)

    # run bat_algorithm with progress callback (animated)
    with st.spinner("Running BAT algorithm — this may take a moment..."):
        selected_indices, conv_curve = bat_algorithm(
            features=features_arr,
            labels=labels_arr,
            num_bats=num_bats,
            max_gen=max_gen,
            loudness=loudness,
            pulse_rate=pulse_rate,
            progress_callback=progress_callback
        )

    # Clear status/progress
    status_placeholder.empty()
    progress_placeholder.empty()

    if not selected_indices:
        st.error("BAT returned no features — try increasing generations or bats.")
    else:
        selected_feature_names = list(X.columns[selected_indices])
        st.success(f"BAT finished — selected {len(selected_feature_names)} features.")
        st.write("**Selected features:**", selected_feature_names)

        # show final convergence (interactive)
        fig_final = px.line(y=conv_curve, labels={'index':'Generation','value':'Best CV Accuracy'},
                            title="BAT Convergence Curve (final)")
        fig_final.update_traces(mode='lines+markers')
        conv_placeholder.plotly_chart(fig_final, use_container_width=True)

        # ---------------- Train final Naive Bayes ----------------
        X_sel = X.iloc[:, selected_indices]
        X_train, X_test, y_train, y_test = train_test_split(X_sel, y, test_size=0.20, random_state=42, stratify=y)
        model = GaussianNB()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:,1]

        # metrics
        acc = (y_pred == y_test).mean()
        report_text = classification_report(y_test, y_pred, digits=4)
        cm = classification_matrix = confusion_matrix(y_test, y_pred)
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        roc_auc = auc(fpr, tpr)

        # show results
        with results_container:
            st.subheader("✅ Model Results")
            st.metric("Test accuracy", f"{acc:.4f}", delta=None)
            st.markdown("**Classification report**")
            st.text(report_text)

            # confusion matrix (Plotly heatmap)
            cm_fig = go.Figure(data=go.Heatmap(
                z=cm,
                x=["Pred 0","Pred 1"],
                y=["True 0","True 1"],
                colorscale="Blues",
                showscale=True
            ))
            cm_fig.update_layout(title="Confusion Matrix")
            st.plotly_chart(cm_fig, use_container_width=True)

            # ROC curve
            roc_fig = go.Figure()
            roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name=f"AUC={roc_auc:.3f}"))
            roc_fig.add_trace(go.Scatter(x=[0,1], y=[0,1], mode='lines', line=dict(dash='dash'), name='chance'))
            roc_fig.update_layout(title="ROC Curve", xaxis_title="FPR", yaxis_title="TPR")
            st.plotly_chart(roc_fig, use_container_width=True)

            # Provide download of selected dataset used for training
            df_selected = X_sel.copy()
            df_selected['target'] = y
            csv = df_selected.to_csv(index=False).encode('utf-8')
            st.download_button("Download selected features dataset (CSV)", csv, "selected_features_data.csv", "text/csv")

        # ---------------- Patient prediction form (live) ----------------
        st.sidebar.markdown("---")
        st.sidebar.header("🧾 Predict for a patient (using selected features)")
        user_input = {}
        for fname in selected_feature_names:
            # use min/max/mean for slider ranges
            col_min = float(X_sel[fname].min())
            col_max = float(X_sel[fname].max())
            col_mean = float(X_sel[fname].mean())
            user_input[fname] = st.sidebar.number_input(fname, value=col_mean, min_value=col_min, max_value=col_max, format="%.5f")

        if st.sidebar.button("Predict patient"):
            sample = np.array([list(user_input.values())])
            pred = model.predict(sample)[0]
            prob = model.predict_proba(sample)[0][pred]
            label = "Benign (0)" if pred==0 else "Malignant (1)"
            # Animated result: quick progress then show
            pb = st.sidebar.progress(0)
            for i in range(5):
                time.sleep(0.08)
                pb.progress((i+1)*20)
            pb.empty()
            if pred == 1:
                st.sidebar.error(f"Prediction: {label} — Confidence {prob:.2%}")
            else:
                st.sidebar.success(f"Prediction: {label} — Confidence {prob:.2%}")

else:
    st.info("Press **Run BAT & Train** in the sidebar to start optimization (or upload CSV first).")
