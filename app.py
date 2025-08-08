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
from sklearn.metrics import confusion_matrix, classification_report, roc_curve, auc
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
    try:
        df = pd.read_csv(uploaded_obj)
        return df
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

if not isinstance(df_raw, pd.DataFrame):
    st.error("Loaded data is not a table. Make sure you uploaded a CSV file.")
    st.stop()

columns = list(df_raw.columns)
tgt_col = st.sidebar.selectbox("Select target (label) column", options=columns, index=columns.index("Target") if "Target" in columns else len(columns)-1)
st.sidebar.write(f"Using target column: **{tgt_col}**")

# Patient form fields
medical_fields = [
    'Age','Sex','BMI','Family_History','Smoking_Status','Alcohol_Intake',
    'Physical_Activity','Hormone_Therapy','Breastfeeding_History','Pregnancies'
]

st.sidebar.markdown("### Map patient-form fields to dataset columns")
mapping = {}
for f in medical_fields:
    default = f if f in columns else "Not present"
    options = ["Not present"] + columns
    mapping[f] = st.sidebar.selectbox(f"{f} ->", options=options, index=options.index(default) if default in options else 0)

# -----------------------
# Prepare X, y for training
# -----------------------
def clean_and_prepare(df, target_col):
    dfc = df.copy()
    dfc = dfc.dropna(how='all')
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
    return X.reset_index(drop=True), y.reset_index(drop=True)

X_df, y_ser = clean_and_prepare(df_raw, tgt_col)

# -----------------------
# Correlation heatmap
# -----------------------
st.subheader("Correlation Heatmap (Numeric Features)")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(X_df.corr(), cmap="coolwarm", annot=True, ax=ax)
st.pyplot(fig)
st.write(
    """
    **Explanation:**
    The correlation heatmap shows the relationship between numerical features. The values on the heatmap represent the correlation coefficients, ranging from -1 to 1.
    - A **positive correlation** indicates that as one feature increases, the other increases (e.g., Age vs BMI).
    - A **negative correlation** means that as one feature increases, the other decreases.
    - A value of **0** indicates **no correlation**.
    This visualization helps in identifying redundant features or those that are highly correlated.
    """
)

# Distributions of numeric features (first 6)
st.subheader("Numeric Feature Distributions (By Target)")
for col in X_df.select_dtypes(include=[np.number]).columns[:6]:
    fig = px.histogram(X_df, x=col, color=y_ser, nbins=30, title=f"{col} Distribution by Target", marginal="box")
    st.plotly_chart(fig, use_container_width=True)
    st.write(
        f"""
        **Explanation:**
        This chart shows the distribution of the `{col}` feature. The bars represent the frequency of values within bins, while the boxplot shows the spread of data.
        The colors represent the target label: **0** (benign) and **1** (malignant).
        This helps us understand how this feature varies between the two classes.
        """
    )

# -----------------------
# Feature importance (Mutual Information)
# -----------------------
st.subheader("Feature Importance (Mutual Information)")
mi = mutual_info_classif(X_df, y_ser)
mi_df = pd.DataFrame({'Feature': X_df.columns, 'Mutual Information': mi}).sort_values('Mutual Information', ascending=False)
st.dataframe(mi_df.style.background_gradient(cmap='Oranges'))
st.write(
    """
    **Explanation:**
    This table shows the **mutual information** between each feature and the target class. Higher mutual information indicates that the feature provides more valuable information to predict the target.
    Features with **higher mutual information** are typically more important in classification.
    """
)

# -----------------------
# Step-by-step Explanation
# -----------------------
st.markdown("""
## Step-by-step: What the app does & how to interpret graphs

**1) Data Upload**: Upload your dataset (CSV) or use the built-in synthetic dataset.
**2) BAT Training**: The BAT algorithm selects important features for classification based on cross-validation scores.
**3) Prediction**: The model predicts whether a given patient is likely to have benign or malignant breast cancer based on input features.
**4) Evaluation**: Evaluate the model’s performance with metrics like accuracy, confusion matrix, ROC, and feature importance.
**5) Interpretation**: The app explains the predictions in plain language, including next steps and confidence levels.

### Important Notes:
- This is a prototype for educational use and not for actual clinical diagnosis.
- Always consult medical professionals for any health concerns or diagnostic decisions.
""")
