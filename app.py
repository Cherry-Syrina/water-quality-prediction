import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from data.generate_data import generate_dataset, get_feature_names, WHO_STANDARDS
from model.train_model import train_pipeline
from utils.predictor import predict, get_risk_level

st.set_page_config(page_title="💧 Water Quality AI", page_icon="💧", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;600;700&display=swap');

* { font-family: 'Poppins', sans-serif; }

.stApp {
    background: linear-gradient(135deg, #1a1a2e 0%, #16213e 40%, #0f3460 100%);
    color: #ffffff !important;
}

.hero {
    background: linear-gradient(135deg, #f72585, #7209b7, #3a0ca3, #4361ee, #4cc9f0);
    background-size: 300% 300%;
    animation: gradientShift 6s ease infinite;
    padding: 3rem 2rem;
    border-radius: 20px;
    text-align: center;
    margin-bottom: 2rem;
    box-shadow: 0 20px 60px rgba(247,37,133,0.3);
}
@keyframes gradientShift {
    0%{background-position:0% 50%}
    50%{background-position:100% 50%}
    100%{background-position:0% 50%}
}
.hero h1 { font-size: 2.6rem; color: white; margin: 0; font-weight: 700; text-shadow: 0 2px 10px rgba(0,0,0,0.3); }
.hero p  { font-size: 1.1rem; color: rgba(255,255,255,0.9); margin: 0.8rem 0 0; }

.stat-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.1), rgba(255,255,255,0.05));
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.15);
    backdrop-filter: blur(10px);
    margin-bottom: 1rem;
    transition: transform 0.2s;
}
.stat-card:hover { transform: translateY(-4px); }
.stat-card h3 { font-size: 2.2rem; margin: 0; font-weight: 700; }
.stat-card p  { font-size: 0.85rem; margin: 0.3rem 0 0; opacity: 0.8; }

.s1 h3 { color: #f72585; }
.s2 h3 { color: #4cc9f0; }
.s3 h3 { color: #7209b7; }
.s4 h3 { color: #4361ee; }

.step-card {
    background: linear-gradient(135deg, rgba(255,255,255,0.08), rgba(255,255,255,0.03));
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    border: 1px solid rgba(255,255,255,0.1);
    margin-bottom: 1rem;
}
.step-card .icon { font-size: 2.5rem; }
.step-card h4 { color: #4cc9f0; margin: 0.5rem 0 0.3rem; font-size: 1rem; }
.step-card p  { font-size: 0.83rem; opacity: 0.8; margin: 0; }

.result-safe {
    background: linear-gradient(135deg, #0f3460, #00b4d8);
    border-radius: 16px; padding: 2rem; text-align: center;
    border: 2px solid #00b4d8;
    box-shadow: 0 0 30px rgba(0,180,216,0.3);
    margin: 1rem 0;
}
.result-unsafe {
    background: linear-gradient(135deg, #3d0000, #f72585);
    border-radius: 16px; padding: 2rem; text-align: center;
    border: 2px solid #f72585;
    box-shadow: 0 0 30px rgba(247,37,133,0.3);
    margin: 1rem 0;
}
.result-safe h2, .result-unsafe h2 { color: white; margin: 0; font-size: 1.8rem; }
.result-safe p,  .result-unsafe p  { color: rgba(255,255,255,0.85); margin: 0.5rem 0 0; }

.param-safe {
    background: linear-gradient(135deg, rgba(0,180,216,0.15), rgba(0,180,216,0.05));
    border-radius: 12px; padding: 1rem; text-align: center;
    border: 1px solid rgba(0,180,216,0.4);
    margin-bottom: 0.8rem;
}
.param-unsafe {
    background: linear-gradient(135deg, rgba(247,37,133,0.15), rgba(247,37,133,0.05));
    border-radius: 12px; padding: 1rem; text-align: center;
    border: 1px solid rgba(247,37,133,0.4);
    margin-bottom: 0.8rem;
}
.param-safe b, .param-unsafe b { font-size: 0.9rem; }
.param-safe .status  { color: #4cc9f0; font-weight: 700; font-size: 0.8rem; }
.param-unsafe .status { color: #f72585; font-weight: 700; font-size: 0.8rem; }

.stTabs [data-baseweb="tab"] {
    background: rgba(255,255,255,0.07) !important;
    color: rgba(255,255,255,0.7) !important;
    border-radius: 10px 10px 0 0 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #f72585, #7209b7) !important;
    color: white !important;
    font-weight: 700 !important;
}
label, .stSlider label { color: #4cc9f0 !important; font-weight: 600 !important; }

section[data-testid="stSidebar"] { display: none; }
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

# Hero
st.markdown("""
<div class="hero">
  <h1>💧 AI Water Quality Prediction</h1>
  <p>Next-generation Machine Learning for real-time potability analysis & aquatic ecosystem monitoring</p>
</div>
""", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["🏠 Home", "🔬 Predict", "📊 Model Insights", "🌍 About"])

# ── TAB 1 ──────────────────────────────────────────────────────────────────────
with tab1:
    c1, c2, c3, c4 = st.columns(4)
    cards = [
        ("s1", f"{len(df):,}", "Water Samples"),
        ("s2", f"{metrics['accuracy']*100:.1f}%", "Model Accuracy"),
        ("s3", f"{metrics['roc_auc']:.3f}", "ROC-AUC Score"),
        ("s4", "8", "Parameters Analyzed"),
    ]
    for col, (cls, val, label) in zip([c1,c2,c3,c4], cards):
        with col:
            st.markdown(f'<div class="stat-card {cls}"><h3>{val}</h3><p>{label}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🌊 How It Works")
    cols = st.columns(3)
    steps = [
        ("📥", "Input Parameters", "Enter 8 physicochemical water parameters using the sliders."),
        ("🤖", "AI Analysis",      "Gradient Boosting model predicts potability with confidence score."),
        ("📋", "Verdict + Advice", "Instant Safe/Unsafe verdict with WHO-based remediation tips."),
    ]
    for col, (icon, title, desc) in zip(cols, steps):
        with col:
            st.markdown(f'<div class="step-card"><div class="icon">{icon}</div><h4>{title}</h4><p>{desc}</p></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📈 Dataset Overview")
    col1, col2 = st.columns(2)

    with col1:
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        counts = df["Potability"].value_counts()
        colors = ["#4cc9f0", "#f72585"]
        bars = ax.bar(["💧 Safe", "⚠️ Unsafe"], counts.values, color=colors, alpha=0.9, width=0.5)
        for b in bars:
            ax.text(b.get_x()+b.get_width()/2, b.get_height()+20,
                    str(int(b.get_height())), ha="center", color="white", fontsize=13, fontweight="bold")
        ax.set_title("Sample Distribution", color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")
        for spine in ax.spines.values(): spine.set_visible(False)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    with col2:
        fig, ax = plt.subplots(figsize=(6,4), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        safe_df   = df[df["Potability"]==1]
        unsafe_df = df[df["Potability"]==0]
        ax.scatter(safe_df["pH"],   safe_df["Dissolved_Oxygen"],   alpha=0.35, color="#4cc9f0", label="Safe",   s=14)
        ax.scatter(unsafe_df["pH"], unsafe_df["Dissolved_Oxygen"], alpha=0.35, color="#f72585", label="Unsafe", s=14)
        ax.set_xlabel("pH", color="white")
        ax.set_ylabel("Dissolved Oxygen (mg/L)", color="white")
        ax.set_title("pH vs Dissolved Oxygen", color="white", fontsize=13, fontweight="bold")
        ax.tick_params(colors="white")
        ax.legend(facecolor="#0f3460", labelcolor="white")
        for spine in ax.spines.values(): spine.set_color("#ffffff22")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ── TAB 2 ──────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("## 🔬 Real-Time Water Quality Prediction")
    st.markdown("Adjust the sliders to enter water sample parameters:")

    col1, col2 = st.columns(2)
    with col1:
        ph        = st.slider("💧 pH Level",                4.0,  11.0,  7.2, 0.1,  help="WHO: 6.5–8.5")
        do        = st.slider("🌬️ Dissolved Oxygen (mg/L)", 0.0,  14.0,  8.0, 0.1,  help="Safe: >6.5")
        turbidity = st.slider("🌊 Turbidity (NTU)",         0.0,  20.0,  2.5, 0.1,  help="WHO: <4 NTU")
        temp      = st.slider("🌡️ Temperature (°C)",        5.0,  40.0, 22.0, 0.5)
    with col2:
        nitrate  = st.slider("🌿 Nitrate (mg/L)",           0.0,  50.0,  5.0, 0.5,  help="WHO: <10 mg/L")
        bod      = st.slider("🧪 BOD (mg/L)",               0.0,  15.0,  2.0, 0.1,  help="Good: <3 mg/L")
        cond     = st.slider("⚡ Conductivity (µS/cm)",      100, 1500,   400,  10)
        coliform = st.slider("🦠 Coliform (CFU/100mL)",       0,  1000,    30,   5,  help="Safe: <50")

    if st.button("🚀 Predict Water Quality", type="primary", use_container_width=True):
        input_values = {
            "pH": ph, "Dissolved_Oxygen": do, "Turbidity": turbidity,
            "Temperature": temp, "Nitrate": nitrate, "BOD": bod,
            "Conductivity": cond, "Coliform": coliform,
        }
        result = predict(model, scaler, input_values)
        level, color = get_risk_level(result)

        st.markdown("---")
        if result.is_potable:
            st.markdown(f"""
            <div class="result-safe">
              <h2>✅ SAFE — Water is Potable</h2>
              <p>Confidence: <b>{result.confidence:.1f}%</b> &nbsp;|&nbsp; Risk Level: <b>{level}</b></p>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-unsafe">
              <h2>⚠️ UNSAFE — Water is Non-Potable</h2>
              <p>Confidence: <b>{result.confidence:.1f}%</b> &nbsp;|&nbsp; Risk Level: <b>{level}</b></p>
            </div>""", unsafe_allow_html=True)

        st.markdown("### 📋 Parameter Analysis")
        cols = st.columns(4)
        for i, s in enumerate(result.parameters):
            icon  = "✅" if s.is_safe else "❌"
            cls   = "param-safe" if s.is_safe else "param-unsafe"
            label = "✔ IN RANGE" if s.is_safe else "✘ OUT OF RANGE"
            with cols[i % 4]:
                st.markdown(f"""
                <div class="{cls}">
                  <b>{icon} {s.name.replace('_',' ')}</b><br>
                  <span style="font-size:1.1rem; font-weight:700">{s.value:.1f}</span><br>
                  <span style="font-size:0.72rem; opacity:0.75">{s.description}</span><br>
                  <span class="status">{label}</span>
                </div>""", unsafe_allow_html=True)

        if result.recommendations:
            st.markdown("### 🛠️ Remediation Recommendations")
            for rec in result.recommendations:
                st.warning(rec)
        else:
            st.success("✅ All parameters within WHO safe limits!")

# ── TAB 3 ──────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("## 📊 Model Performance & Insights")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown(f"**Accuracy: {metrics['accuracy']*100:.2f}%** &nbsp;|&nbsp; **ROC-AUC: {metrics['roc_auc']:.4f}**")
        report_df = pd.DataFrame(metrics["classification_report"]).transpose().round(3)
        st.dataframe(report_df[["precision","recall","f1-score","support"]], use_container_width=True)

    with col2:
        fig, ax = plt.subplots(figsize=(5,4), facecolor="#1a1a2e")
        ax.set_facecolor("#1a1a2e")
        sns.heatmap(metrics["confusion_matrix"], annot=True, fmt="d",
                    cmap=sns.color_palette("rocket", as_cmap=True), ax=ax,
                    xticklabels=["Non-Potable","Potable"],
                    yticklabels=["Non-Potable","Potable"])
        ax.set_title("Confusion Matrix", color="white", fontweight="bold")
        ax.tick_params(colors="white")
        ax.set_xlabel("Predicted", color="white")
        ax.set_ylabel("Actual", color="white")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    st.markdown("### 🔍 Feature Importance")
    fig, ax = plt.subplots(figsize=(9,4), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")
    importances = model.feature_importances_
    idx = np.argsort(importances)[::-1]
    bar_colors = ["#f72585","#7209b7","#3a0ca3","#4361ee","#4895ef","#4cc9f0","#72efdd","#b5e48c"]
    ax.bar(range(len(feature_names)), importances[idx], color=bar_colors, alpha=0.9)
    ax.set_xticks(range(len(feature_names)))
    ax.set_xticklabels([feature_names[i] for i in idx], rotation=30, ha="right", color="white", fontsize=10)
    ax.set_title("Feature Importance — Gradient Boosting", color="white", fontsize=13, fontweight="bold")
    ax.tick_params(colors="white")
    for spine in ax.spines.values(): spine.set_visible(False)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# ── TAB 4 ──────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("## 🌍 About This Project")
    st.markdown("""
### AI-Based Water Quality Prediction System

This project applies **Machine Learning** to predict water potability from physicochemical parameters — enabling rapid, cost-effective assessment without expensive laboratory testing.

**Technical Approach**
- **Algorithm:** Gradient Boosting Classifier (ensemble of decision trees)
- **Preprocessing:** StandardScaler normalization; 80/20 stratified train-test split
- **Features:** 8 WHO-aligned physicochemical parameters

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
st.markdown('<p style="text-align:center; color:#4cc9f0; font-size:0.85rem">💧 AI Water Quality Prediction | Streamlit + Scikit-learn | Sushma Shukla, VIT Vellore</p>', unsafe_allow_html=True)
