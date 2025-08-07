import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import train_test_split
from utils.bat_algorithm import bat_algorithm

st.set_page_config(page_title="🧬 Breast Cancer BAT App", layout="wide", initial_sidebar_state="expanded")

# Sidebar
st.sidebar.title("🔧 App Settings")
num_bats = st.sidebar.slider("Number of Bats", 10, 50, 30)
max_gen = st.sidebar.slider("Max Generations", 10, 100, 50)
st.sidebar.markdown("---")
page = st.sidebar.radio("📂 Navigate", ["🏠 Home", "📊 Analysis", "🧪 Patient Form", "📈 Model Metrics", "📥 Export", "ℹ️ About"])

# Load data
data = load_breast_cancer()
X = data.data
y = data.target
feature_names = data.feature_names

@st.cache_data(show_spinner=True)
def run_bat(num_bats, max_gen):
    selected_features, convergence = bat_algorithm(X, y, num_bats=num_bats, max_gen=max_gen)
    return selected_features, convergence

selected_features, convergence = run_bat(num_bats, max_gen)
X_selected = X[:, selected_features == 1]
selected_names = feature_names[selected_features == 1]
X_train, X_test, y_train, y_test = train_test_split(X_selected, y, test_size=0.2, random_state=42)

model = GaussianNB()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
probs = model.predict_proba(X_test)[:, 1]
acc = accuracy_score(y_test, y_pred)
fpr, tpr, _ = roc_curve(y_test, probs)
roc_auc = auc(fpr, tpr)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred, output_dict=True)
df_report = pd.DataFrame(report).transpose()

# ----------------- PAGES ------------------

# Home
if page == "🏠 Home":
    st.title("🧬 Breast Cancer Predictor using BAT + Naive Bayes")
    st.markdown("""
    This app applies the **BAT Algorithm** for feature selection and **Naive Bayes** for prediction.
    """)
    st.success(f"🎯 Accuracy: {acc:.4f}")
    st.info(f"🔎 Selected Features: {selected_names.tolist()}")

    # BAT convergence curve
    st.subheader("📉 BAT Convergence Curve")
    fig1, ax1 = plt.subplots()
    ax1.plot(convergence, marker='o')
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Cross-Validated Accuracy")
    ax1.set_title("BAT Algorithm Convergence")
    st.pyplot(fig1)

# Analysis
elif page == "📊 Analysis":
    st.title("📊 Feature Analysis")
    feature_to_plot = st.selectbox("Choose Feature", selected_names)
    idx = list(feature_names).index(feature_to_plot)

    st.subheader("📌 Distribution by Class")
    df_feat = pd.DataFrame({feature_to_plot: X[:, idx], "Target": y})
    fig2, ax2 = plt.subplots()
    sns.histplot(data=df_feat, x=feature_to_plot, hue="Target", kde=True, ax=ax2, palette="Set1")
    st.pyplot(fig2)

    st.subheader("📌 Correlation Heatmap (Selected Features)")
    corr_df = pd.DataFrame(X_selected, columns=selected_names)
    fig3, ax3 = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr_df.corr(), annot=True, cmap="coolwarm", ax=ax3)
    st.pyplot(fig3)

# Patient Prediction
elif page == "🧪 Patient Form":
    st.title("🧪 Patient Form for Prediction")
    with st.form("patient_form"):
        inputs = []
        for name in selected_names:
            val = st.number_input(f"{name}", min_value=0.0, value=1.0)
            inputs.append(val)
        submitted = st.form_submit_button("Predict")

    if submitted:
        patient_data = np.array(inputs).reshape(1, -1)
        pred = model.predict(patient_data)[0]
        conf = model.predict_proba(patient_data)[0][pred]
        diagnosis = "🟢 Benign" if pred == 1 else "🔴 Malignant"
        st.markdown(f"### 🔍 Prediction: {diagnosis}")
        st.metric(label="Confidence Score", value=f"{conf:.2%}")

# Model Metrics
elif page == "📈 Model Metrics":
    st.title("📈 Model Evaluation")
    st.metric("🎯 Accuracy", f"{acc:.4f}")
    st.subheader("📘 Confusion Matrix")
    st.write(pd.DataFrame(cm, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"]))

    st.subheader("📋 Classification Report")
    st.dataframe(df_report.style.background_gradient(cmap='YlOrBr'))

    st.subheader("🎯 ROC Curve")
    fig4, ax4 = plt.subplots()
    ax4.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}", linewidth=2)
    ax4.plot([0, 1], [0, 1], '--', color='gray')
    ax4.set_xlabel("False Positive Rate")
    ax4.set_ylabel("True Positive Rate")
    ax4.set_title("ROC Curve")
    ax4.legend()
    st.pyplot(fig4)

# Export
elif page == "📥 Export":
    st.title("📥 Export Report")
    df_all = pd.DataFrame(X_selected, columns=selected_names)
    df_all["target"] = y
    csv = df_all.to_csv(index=False).encode('utf-8')
    st.download_button("Download Selected Dataset CSV", csv, "selected_features_data.csv", "text/csv")

    report_csv = df_report.to_csv().encode('utf-8')
    st.download_button("Download Metrics Report CSV", report_csv, "classification_report.csv", "text/csv")

# About
elif page == "ℹ️ About":
    st.title("ℹ️ About This Project")
    st.markdown("""
    - 🔬 **Title:** Breast Cancer Prediction using BAT Algorithm and Naive Bayes  
    - 🧠 **Feature Selection:** BAT Algorithm (bio-inspired)  
    - 📈 **Classifier:** Gaussian Naive Bayes  
    - 🛠 **Built With:** Streamlit, scikit-learn, matplotlib, seaborn  
    - 👤 **Developer:** Dani (with AI Assistant)
    """)
