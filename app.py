import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from utils.bat_algorithm import bat_algorithm

st.set_page_config(page_title="Breast Cancer Prediction", layout="wide")

# Sidebar: Dataset Upload
st.sidebar.title("📂 Data Settings")
uploaded_file = st.sidebar.file_uploader("Upload CSV Dataset", type=["csv"])
num_bats = st.sidebar.slider("Number of Bats", 10, 50, 30)
max_gen = st.sidebar.slider("Max Generations", 10, 100, 50)
page = st.sidebar.radio("📋 Navigation", ["🏠 Home", "📊 Analysis", "🧪 Patient Form", "📈 Model Metrics"])

# Load dataset
@st.cache_data(show_spinner=True)
def load_default_data():
    from sklearn.datasets import load_breast_cancer
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target)
    return X, y, data.feature_names

@st.cache_data(show_spinner=True)
def load_uploaded_data(file):
    df = pd.read_csv(file)
    y = df.iloc[:, -1]  # last column is target
    X = df.iloc[:, :-1]
    return X, y, X.columns

# Choose data
if uploaded_file:
    X, y, feature_names = load_uploaded_data(uploaded_file)
else:
    X, y, feature_names = load_default_data()

# Run BAT
@st.cache_data(show_spinner=True)
def run_bat_algorithm(X, y):
    selected_features, convergence = bat_algorithm(X.values, y.values, num_bats=num_bats, max_gen=max_gen)
    return selected_features, convergence

selected_features, convergence_curve = run_bat_algorithm(X, y)
X_selected = X.loc[:, selected_features == 1]
selected_names = X_selected.columns

# Train model
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)
model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
accuracy = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# ---------------- UI Pages ----------------

# 🏠 HOME
if page == "🏠 Home":
    st.title("🧬 Breast Cancer Predictor")
    st.success(f"Model Accuracy: **{accuracy:.2%}**")
    st.write(f"Selected Features ({len(selected_names)}):")
    st.code(", ".join(selected_names))

    # Convergence plot
    st.subheader("📈 BAT Algorithm Convergence")
    fig1, ax1 = plt.subplots()
    ax1.plot(convergence_curve, marker='o')
    ax1.set_title("BAT Convergence Curve")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Cross-Validated Accuracy")
    st.pyplot(fig1)

# 📊 ANALYSIS
elif page == "📊 Analysis":
    st.title("📊 Feature Analysis")
    feat = st.selectbox("Select Feature", selected_names)
    st.subheader(f"Distribution of {feat}")
    fig2, ax2 = plt.subplots()
    sns.histplot(x=X[feat], hue=y, kde=True, ax=ax2)
    st.pyplot(fig2)

# 🧪 PATIENT FORM
elif page == "🧪 Patient Form":
    st.title("🧪 Patient Prediction Form")
    st.markdown("### Enter patient measurements for selected features")

    with st.form("patient_form"):
        inputs = []
        for feat in selected_names:
            val = st.number_input(f"{feat}", value=float(X[feat].mean()), step=0.01)
            inputs.append(val)
        submitted = st.form_submit_button("Predict")

    if submitted:
        sample = np.array(inputs).reshape(1, -1)
        prediction = model.predict(sample)[0]
        confidence = model.predict_proba(sample)[0][prediction]
        label = "🟢 Benign" if prediction == 1 else "🔴 Malignant"
        st.success(f"Prediction: {label} (Confidence: {confidence:.2%})")

# 📈 METRICS
elif page == "📈 Model Metrics":
    st.title("📈 Evaluation Metrics")
    st.metric("Accuracy", f"{accuracy:.2%}")

    st.subheader("Confusion Matrix")
    st.write(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

    st.subheader("Classification Report")
    st.dataframe(df_report.style.background_gradient(cmap="Blues"))

    st.subheader("ROC Curve")
    fig3, ax3 = plt.subplots()
    ax3.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
    ax3.plot([0, 1], [0, 1], linestyle='--', color='gray')
    ax3.set_xlabel("False Positive Rate")
    ax3.set_ylabel("True Positive Rate")
    ax3.set_title("ROC Curve")
    ax3.legend()
    st.pyplot(fig3)
