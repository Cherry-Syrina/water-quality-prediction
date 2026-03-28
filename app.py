import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from data.generate_data import generate_dataset, get_feature_names, WHO_STANDARDS
from model.train_model import train_pipeline
from utils.predictor import predict, get_risk_level

st.set_page_config(page_title="AI Water Quality Prediction", page_icon="💧", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0a1628, #0d2137, #0a3d52); color: #f0f8ff !important; }
.hero  { background: linear-gradient(135deg, #0077b6, #00b4d8); padding: 2.5rem 2rem; border-radius: 16px; text-align: center; margin-bottom: 1.5rem; }
.hero h1 { font-size: 2.4rem; color: white; margin: 0; font-weight: 700; }
.hero p  { font-size: 1.1rem; color: #caf0f8; margin: 0.5rem 0 0; }
.card    { background: rgba(255,255,255,0.07); border-radius: 12px; padding: 1.2rem; text-align: center; border: 1px solid rgba(255,255,255,0.1); margin-bottom: 0.8rem; }
.card h3 { font-size: 2rem; margin: 0; }
.card p  { font-size: 0.85rem; color: #90e0ef; margin: 0.3rem 0 0; }
.safe    { border-left: 5px solid #00b894; }
.unsafe  { border-left: 5px solid #e17055; }
.stTabs [data-baseweb="tab"]   { background: #0d2137; color: #caf0f8 !important; border-radius: 8px 8px 0 0; }
.stTabs [aria-selected="true"] { background: #00b4d8 !important; color: white !important; font-weight: 600 !important; }
label, .stSelectbox label, .stSlider label { color: #caf0f8 !important; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return generate_dataset(n_samples=3000)

@st.cache_resource
def load_model():
    return train_pipeline(n_samples=3000)

df = load_data()
model, scaler, metrics, X_test_s, y_test = load_model()
feature_names = get_feature_names()

st.markdown("""
<div class="hero">
  <h1>💧 AI Water Quality Prediction System</h1>
  <p>Machine Learning-powered real-time potability analysis for aquatic ecosystem monitoring</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔬 Predict", "📊 Model Insights", "🌍 About"])

with tab1:
    c1, c2, c3, c4 = st.columns(4)
    stats = [
        (f"{len(df):,}", "Water Samples"),
        (f"{metrics['accuracy']*100:.1f}%", "Model Accuracy"),
        (f"{metrics['roc_auc']:.3f}", "ROC-AUC Score"),
        ("8", "Parameters Analyzed"),
    ]
    for col, (val, label) in zip([c1, c2, c3, c4], stats):
        with col:
            st.markdown(f'<div class="card"><h3>{val}</h3><p>{label}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌊 How It Works")
    cols = st.columns(3)
    steps = [
        ("📥", "Input Parameters", "Enter 8 physicochemical water parameters using the sliders."),
        ("🤖", "AI Analysis", "Gradient Boosting model predicts potability with confidence score."),
        ("📋", "Verdict + Advice", "Instant Safe/Unsafe verdict with WHO-based remediation tips."),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="card"><h3>{icon}</h3><p><b>{title}</b><br>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Dataset Overview")
    col1, col2 = st.columns(2)
    with col1:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0a1628")
        ax.set_facecolor("#0a1628")
        counts = df["Potability"].value_counts()
        bars = ax.bar(["Safe (Potable)", "Unsafe (Non-Potable)"], counts.values, color=["#00b894", "#e17055"], alpha=0.85, width=0.5)
        for b in bars:
            ax.text(b.get_x() + b.get_width()/2, b.get_height()+20, str(int(b.get_height())), ha="center", color="white", fontsize=12, fontweight="bold")
        ax.set_title("Sample Distribution", color="white", fontsize=13)
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    with col2:
        fig, ax = plt.subplots(figsize=(6, 4), facecolor="#0a1628")
        ax.set_facecolor("#0a1628")
        safe_df = df[df["Potability"]==1]
        unsafe_df = df[df["Potability"]==0]
        ax.scatter(safe_df["pH"], safe_df["Dissolved_Oxygen"], alpha=0.35, color="#00b894", label="Safe", s=14)
        ax.scatter(unsafe_df["pH"], unsafe_df["Dissolved_Oxygen"], alpha=0.35, color="#e17055", label="Unsafe", s=14)
        ax.set_xlabel("pH", color="white")
        ax.set_ylabel("Dissolved Oxygen (mg/L)", color="white")
        ax.set_title("pH vs Dissolved Oxygen", color="white", fontsize=13)
        ax.tick_params(colors="white")
        ax.legend(facecolor="#0d2137", labelcolor="white")
        for spine in ax.spines.values(): spine.set_color("#ffffff33")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()

with tab2:
    st.markdown("## 🔬 Real-Time Water Quality Prediction")
    st.markdown("Adjust the sliders to enter water sample parameters:")
    col1, col2 = st.columns(2)
    with col1:
        ph        = st.slider("pH Level", 4.0, 11.0, 7.2, 0.1, help="WHO standard: 6.5–8.5")
        do        = st.slider("Dissolved Oxygen (mg/L)", 0.0, 14.0, 8.0, 0.1, help="Safe: >6.5 mg/L")
        turbidity = st.slider("Turbidity (NTU)", 0.0, 20.0, 2.5, 0.1, help="WHO limit: <4 NTU")
        temp      = st.slider("Temperature (°C)", 5.0, 40.0, 22.0, 0.5)
    with col2:
        nitrate  = st.slider("Nitrate (mg/L)", 0.0, 50.0, 5.0, 0.5, help="WHO limit: <10 mg/L")
        bod      = st.slider("BOD (mg/L)", 0.0, 15.0, 2.0, 0.1, help="Good water: <3 mg/L")
        cond     = st.slider("Conductivity (µS/cm)", 100, 1500, 400, 10)
        coliform = st.slider("Coliform (CFU/100mL)", 0, 1000, 30, 5, help="Safe: <50 CFU/100mL")

    if st.button("🚀 Predict Water Quality", type="primary", use_container_width=True):
        input_values = {"pH": ph, "Dissolved_Oxygen": do, "Turbidity": turbidity, "Temperature": temp, "Nitrate": nitrate, "BOD": bod, "Conductivity": cond, "Coliform": coliform}
        result = predict(model, scaler, input_values)
        level, color = get_risk_level(result)
        st.markdown("---")
        verdict_class = "safe" if result.is_potable else "unsafe"
        verdict_icon  = "✅" if result.is_potable else "⚠️"
        verdict_text  = "SAFE — Water is Potable" if result.is_potable else "UNSAFE — Water is Non-Potable"
        verdict_color = "#00b894" if result.is_potable else "#e17055"
        st.markdown(f"""
        <div class="card {verdict_class}" style="padding:1.5rem; margin:1rem 0">
          <h2 style="color:{verdict_color}; margin:0">{verdict_icon} {verdict_text}</h2>
          <p style="font-size:1.05rem; color:#caf0f8; margin-top:0.5rem">
            Confidence: <b>{result.confidence:.1f}%</b> &nbsp;|&nbsp; Risk Level: <b style="color:{color}">{level}</b>
          </p>
        </div>""", unsafe_allow_html=True)
        st.markdown("### 📋 Parameter Analysis")
        cols = st.columns(4)
        for i, status in enumerate(result.parameters):
            icon  = "✅" if status.is_safe else "❌"
            cls   = "safe" if status.is_safe else "unsafe"
            label = "OK" if status.is_safe else "OUT OF RANGE"
            lcolor = "#00b894" if status.is_safe else "#e17055"
            with cols[i % 4]:
                st.markdown(f"""
                <div class="card {cls}">
                  <b>{icon} {status.name.replace('_',' ')}</b><br>
                  Value: <b>{status.value:.1f}</b><br>
                  <span style="font-size:0.78rem; color:#90e0ef">{status.description}</span><br>
                  <span style="color:{lcolor}; font-weight:bold">{label}</span>
                </div>""", unsafe_allow_html=True)
        if result.recommendations:
            st.markdown("### 🛠️ Remediation Recommendations")
            for rec in result.recommendations:
                st.warning(rec)
        else:
            st.success("✅ All parameters within WHO safe limits. No remediation required.")

with tab3:
    st.markdown("## 📊 Model Performance & Insights")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"**Accuracy: {metrics['accuracy']*100:.2f}%** &nbsp;|&nbsp; **ROC-AUC: {metrics['roc_auc']:.4f}**")
        report_df = pd.DataFrame(metrics["classification_report"]).transpose().round(3)
        st.dataframe(report_df[["precision", "recall", "f1-score", "support"]], use_container_width=True)
    with col2:
        fig, ax = plt.subplots(figsize=(5, 4), facecolor="#0a1628")
        ax.set_facecolor("#0a1628")
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d", cmap="Blues", ax=ax, xticklabels=["Non-Potable", "Potable"], yticklabels=["Non-Potable", "Potable"])
        ax.set_title("Confusion Matrix", color="white")
        ax.tick_params(colors="white")
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual", color="white")
        plt.tight_layout()
        st.pyplot(fig)
        plt.close()
    st.markdown("### 🔍 Feature Importance")
    fig, ax = plt.subplots(figsize=(9, 4), facecolor="#0a1628")
    ax.set_facecolor("#0a1628")
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    palette = plt.cm.Blues(np.linspace(0.4, 0.9, len(feature_names)))
    ax.bar(range(len(feature_names)), importances[idx], color=palette[::-1], alpha=0.9)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=30, ha="right", color="white", fontsize=10)
    ax.set_title("Feature Importance — Gradient Boosting Classifier", color="white", fontsize=13)
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_visible(False)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

with tab4:
    st.markdown("## 🌍 About This Project")
    st.markdown("""
### AI-Based Water Quality Prediction System for Aquatic Ecosystems

Access to safe drinking water is a critical global challenge. This project applies
**Machine Learning** to predict water potability from physicochemical parameters.

**Technical Approach**
- **Algorithm:** Gradient Boosting Classifier
- **Preprocessing:** StandardScaler normalization; 80/20 stratified train-test split
- **Features:** 8 physicochemical water quality parameters (WHO-aligned)

| Parameter | Safe Range |
|---|---|
| pH | 6.5 – 8.5 |
| Dissolved Oxygen | > 6.5 mg/L |
| Turbidity | < 4 NTU |
| Temperature | 15 – 27°C |
| Nitrate | < 10 mg/L |
| BOD | < 3 mg/L |
| Conductivity | 200–600 µS/cm |
| Coliform | < 50 CFU/100mL |

**Developed by:** Sushma Shukla | B.Tech Software Engineering | VIT Vellore
    """)

st.markdown("---")
st.markdown('<p style="text-align:center; color:#90e0ef; font-size:0.85rem">💧 AI Water Quality Prediction | Streamlit + Scikit-learn | Sushma Shukla, VIT Vellore</p>', unsafe_allow_html=True)
