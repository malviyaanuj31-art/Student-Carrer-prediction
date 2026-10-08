"""
app.py  –  Student AI Analyzer
Streamlit application with a polished, modern UI.
Run:  streamlit run app.py
"""

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import matplotlib
matplotlib.use("Agg")

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Student AI Analyzer",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
#  GLOBAL CUSTOM CSS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Google Font ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

/* ── Root variables ── */
:root {
    --primary:   #6C63FF;
    --secondary: #FF6584;
    --accent:    #43E97B;
    --dark:      #1A1A2E;
    --card-bg:   #16213E;
    --text:      #E0E0E0;
    --muted:     #8892A4;
    --border:    rgba(108,99,255,0.25);
}

/* ── Global ── */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

/* ── Hide default Streamlit header/footer ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1A1A2E 0%, #16213E 60%, #0F3460 100%);
    border-right: 1px solid var(--border);
}
[data-testid="stSidebar"] * { color: #E0E0E0 !important; }
[data-testid="stSidebar"] .stRadio label {
    background: rgba(108,99,255,0.08);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 8px 14px;
    margin: 3px 0;
    display: block;
    transition: all 0.2s;
}
[data-testid="stSidebar"] .stRadio label:hover {
    background: rgba(108,99,255,0.25);
    border-color: var(--primary);
}

/* ── Main background ── */
.stApp {
    background: linear-gradient(135deg, #0D0D1A 0%, #111128 50%, #0A1628 100%);
}

/* ── Page title ── */
h1 {
    background: linear-gradient(90deg, #6C63FF, #FF6584, #43E97B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700 !important;
    font-size: 2.4rem !important;
}
h2 { color: #C9D1E0 !important; font-weight: 600 !important; }
h3 { color: #A0AABB !important; font-weight: 500 !important; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: rgba(108,99,255,0.1);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 14px 18px;
    backdrop-filter: blur(4px);
}
[data-testid="metric-container"] label { color: #8892A4 !important; font-size:0.78rem !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    color: #6C63FF !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #6C63FF, #9B59B6) !important;
    color: white !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 10px 28px !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.3px;
    transition: all 0.25s ease;
    box-shadow: 0 4px 15px rgba(108,99,255,0.35);
}
.stButton > button:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 25px rgba(108,99,255,0.55) !important;
}

/* ── Sliders ── */
[data-testid="stSlider"] > div > div > div > div {
    background: var(--primary) !important;
}

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,0.04);
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
}
.stTabs [data-baseweb="tab"] {
    border-radius: 9px !important;
    color: #8892A4 !important;
    font-weight: 500 !important;
}
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6C63FF, #9B59B6) !important;
    color: white !important;
}

/* ── DataFrames ── */
[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }

/* ── Alerts ── */
.stSuccess { border-radius: 12px !important; border-left: 4px solid #43E97B !important; background: rgba(67,233,123,0.08) !important; }
.stInfo    { border-radius: 12px !important; border-left: 4px solid #6C63FF  !important; background: rgba(108,99,255,0.08) !important; }
.stWarning { border-radius: 12px !important; border-left: 4px solid #F9CA24  !important; background: rgba(249,202,36,0.08) !important; }

/* ── Expander ── */
[data-testid="stExpander"] {
    border: 1px solid var(--border) !important;
    border-radius: 12px !important;
    background: rgba(255,255,255,0.03) !important;
}

/* ── Custom card helper ── */
.ai-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(108,99,255,0.2);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 16px;
    backdrop-filter: blur(6px);
}
.ai-card h4 { color: #6C63FF; margin: 0 0 6px 0; font-size: 1.05rem; }
.ai-card p  { color: #A0AABB; margin: 0; font-size: 0.88rem; line-height: 1.5; }

/* ── Hero banner ── */
.hero-banner {
    background: linear-gradient(135deg, rgba(108,99,255,0.18) 0%, rgba(255,101,132,0.12) 50%, rgba(67,233,123,0.10) 100%);
    border: 1px solid rgba(108,99,255,0.3);
    border-radius: 20px;
    padding: 38px 44px;
    margin-bottom: 28px;
    text-align: center;
}
.hero-banner h1 {
    font-size: 2.8rem !important;
    margin-bottom: 10px;
}
.hero-banner p { color: #A0AABB; font-size: 1.05rem; line-height: 1.6; }

/* ── Workflow step ── */
.wf-step {
    background: rgba(108,99,255,0.10);
    border: 1px solid rgba(108,99,255,0.25);
    border-radius: 14px;
    padding: 18px 14px;
    text-align: center;
}
.wf-step .icon { font-size: 1.8rem; margin-bottom: 6px; }
.wf-step .label { color: #6C63FF; font-weight: 600; font-size: 0.88rem; }
.wf-step .desc  { color: #8892A4; font-size: 0.78rem; margin-top: 4px; }

/* ── Badge ── */
.badge {
    display: inline-block;
    background: linear-gradient(135deg,#6C63FF,#9B59B6);
    color: white;
    border-radius: 20px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}

/* ── Divider ── */
hr { border-color: rgba(108,99,255,0.18) !important; }

/* ── Prediction result box ── */
.pred-box {
    background: linear-gradient(135deg, rgba(108,99,255,0.22), rgba(67,233,123,0.12));
    border: 1px solid rgba(108,99,255,0.45);
    border-radius: 18px;
    padding: 26px 32px;
    text-align: center;
    margin: 16px 0;
}
.pred-box .career {
    font-size: 1.9rem;
    font-weight: 700;
    background: linear-gradient(90deg,#6C63FF,#43E97B);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.pred-box .sub { color: #8892A4; font-size: 0.9rem; margin-top: 6px; }
</style>
""", unsafe_allow_html=True)

# ── Plotly dark theme shared settings ─────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(255,255,255,0.03)",
    font=dict(family="Inter", color="#C9D1E0"),
    xaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
    yaxis=dict(gridcolor="rgba(255,255,255,0.06)", zerolinecolor="rgba(255,255,255,0.06)"),
    margin=dict(t=50, b=40, l=40, r=20),
)
COLORS = ["#6C63FF", "#FF6584", "#43E97B", "#F9CA24", "#A29BFE", "#FD79A8"]

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style='text-align:center; padding: 18px 0 10px 0;'>
  <div style='font-size:2.4rem;'>🎓</div>
  <div style='font-size:1.1rem; font-weight:700; color:#6C63FF; letter-spacing:0.5px;'>Student AI Analyzer</div>
  <div style='font-size:0.75rem; color:#8892A4; margin-top:4px;'>ML & Applied AI Internship</div>
</div>
<hr style='border-color:rgba(108,99,255,0.25); margin:10px 0 18px 0;'>
""", unsafe_allow_html=True)

PAGES = [
    "🏠  Home",
    "🎯  Career Prediction",
    "📈  Salary Regression",
    "🔵  Student Clustering",
    "🧠  Deep Learning",
    "🤖  Q-Learning Demo",
    "ℹ️  About",
]
page = st.sidebar.radio("", PAGES, label_visibility="collapsed")

st.sidebar.markdown("""
<hr style='border-color:rgba(108,99,255,0.2); margin:18px 0 10px 0;'>
<div style='font-size:0.72rem; color:#8892A4; text-align:center; line-height:1.6;'>
  AICTE | IBM SkillsBuild<br>
  <span style='color:#6C63FF;'>Machine Learning & Applied AI</span><br>
  Internship 2026
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 1 – HOME
# ══════════════════════════════════════════════════════════════════════════════
if page == "🏠  Home":

    st.markdown("""
    <div class="hero-banner">
      <h1>🎓 Student AI Analyzer</h1>
      <p>
        An end-to-end Machine Learning application that demonstrates<br>
        <strong style="color:#6C63FF;">Classification · Regression · Clustering · Deep Learning · Reinforcement Learning</strong>
      </p>
      <div style="margin-top:14px;">
        <span class="badge">Decision Tree</span>
        <span class="badge">Linear Regression</span>
        <span class="badge">K-Means</span>
        <span class="badge">Neural Network</span>
        <span class="badge">Q-Learning</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="background:rgba(249,202,36,0.10); border:1px solid rgba(249,202,36,0.3);
                border-radius:12px; padding:12px 18px; font-size:0.85rem; color:#F9CA24; margin-bottom:24px;">
      ⚠️ <strong>Dataset Notice:</strong> This project uses a <strong>synthetic / sample educational dataset</strong>
      created to demonstrate the ML workflow. It does <strong>not</strong> contain data from real students.
    </div>
    """, unsafe_allow_html=True)

    # ── Workflow ──────────────────────────────────────────────────────────────
    st.markdown("### 🔄 ML Workflow")
    steps = [
        ("📂", "Data",          "Load CSV datasets"),
        ("🧹", "Preprocessing", "Clean & transform"),
        ("🏋️", "Training",      "Fit ML models"),
        ("📊", "Evaluation",    "Metrics & plots"),
        ("🔮", "Prediction",    "Predict new inputs"),
    ]
    cols = st.columns(len(steps))
    for col, (icon, label, desc) in zip(cols, steps):
        col.markdown(f"""
        <div class="wf-step">
          <div class="icon">{icon}</div>
          <div class="label">{label}</div>
          <div class="desc">{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Module cards ──────────────────────────────────────────────────────────
    st.markdown("### 📌 Application Modules")
    modules = [
        ("🎯", "Career Prediction",   "Decision Tree Classifier predicts the best-fit tech career from your academic & skill profile."),
        ("📈", "Salary Regression",   "Linear Regression estimates expected salary based on years of experience."),
        ("🔵", "Student Clustering",  "K-Means groups students into performance clusters: Strong, Average, Needs Improvement."),
        ("🧠", "Deep Learning",       "Small neural network demo (TensorFlow/Keras) showing training curves."),
        ("🤖", "Q-Learning Demo",     "Simple Q-Learning simulation — agent learns the optimal study path to the Goal."),
    ]
    c1, c2 = st.columns(2)
    for i, (icon, title, desc) in enumerate(modules):
        col = c1 if i % 2 == 0 else c2
        col.markdown(f"""
        <div class="ai-card">
          <h4>{icon} {title}</h4>
          <p>{desc}</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; color:#8892A4; font-size:0.78rem; margin-top:30px;'>
      Built for AICTE | IBM SkillsBuild ML & Applied AI Internship 2026 · Synthetic dataset · Educational purpose only
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 2 – CAREER PREDICTION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🎯  Career Prediction":

    st.title("🎯 Student Career Prediction")
    st.markdown("<p style='color:#8892A4;'>Enter your profile below. The Decision Tree model will suggest a career path.</p>", unsafe_allow_html=True)

    @st.cache_resource
    def get_classifier():
        from models.classifier import train_classifier
        return train_classifier()

    with st.spinner("⚙️ Training Decision Tree…"):
        model, X_test, y_test, y_pred, accuracy, cm, report, feature_imp, classes = get_classifier()

    # Accuracy banner
    st.markdown(f"""
    <div style='background:linear-gradient(135deg,rgba(108,99,255,0.18),rgba(67,233,123,0.10));
                border:1px solid rgba(108,99,255,0.35); border-radius:14px;
                padding:14px 22px; display:flex; align-items:center; gap:14px; margin-bottom:20px;'>
      <span style='font-size:2rem;'>✅</span>
      <div>
        <div style='color:#6C63FF; font-weight:700; font-size:1.05rem;'>Model Ready</div>
        <div style='color:#A0AABB; font-size:0.85rem;'>Decision Tree · Test Accuracy:
          <strong style='color:#43E97B;'>{accuracy*100:.1f}%</strong>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📋 Your Profile")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("**📚 Academic**")
        study_hours  = st.slider("Study Hours / Day",         0.0, 12.0, 5.0, 0.5)
        attendance   = st.slider("Attendance (%)",            0,   100,  80)
        prev_marks   = st.slider("Previous Marks (%)",        0,   100,  70)
    with col2:
        st.markdown("**💻 Technical & Soft Skills**")
        python_skill = st.slider("Python Skill  (1–10)",      1,   10,   6)
        sql_skill    = st.slider("SQL Skill      (1–10)",     1,   10,   5)
        ml_skill     = st.slider("ML Skill       (1–10)",     1,   10,   5)
        comm_skill   = st.slider("Communication  (1–10)",     1,   10,   6)
        ps_skill     = st.slider("Problem Solving (1–10)",    1,   10,   6)

    st.markdown("")
    predict_btn = st.button("🔮 Predict My Career", use_container_width=True)

    if predict_btn:
        from models.classifier import predict_career
        input_values = [study_hours, attendance, prev_marks,
                        python_skill, sql_skill, ml_skill, comm_skill, ps_skill]
        career, proba = predict_career(model, input_values)

        st.markdown(f"""
        <div class="pred-box">
          <div style='color:#8892A4; font-size:0.85rem; margin-bottom:8px;'>🎯 Best Career Match</div>
          <div class="career">{career}</div>
          <div class="sub">Based on your academic profile and skill ratings</div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence chart
        proba_df = pd.DataFrame(list(proba.items()), columns=["Career","Confidence (%)"]).sort_values("Confidence (%)", ascending=False)
        fig = px.bar(proba_df, x="Career", y="Confidence (%)",
                     color="Career", text="Confidence (%)",
                     color_discrete_sequence=COLORS)
        fig.update_traces(texttemplate="%{text}%", textposition="outside",
                          marker_line_width=0)
        fig.update_layout(title="Prediction Confidence per Career",
                          showlegend=False, yaxis_range=[0,115],
                          **PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

        # Why explanation
        top_features = sorted(feature_imp.items(), key=lambda x: x[1], reverse=True)[:3]
        feat_text = " · ".join(f"**{f}** ({v:.0%})" for f, v in top_features)
        st.markdown(f"""
        <div class="ai-card">
          <h4>💡 Why this prediction?</h4>
          <p>The Decision Tree relied most on {feat_text} to reach this decision.
             Model test accuracy: <strong style='color:#43E97B;'>{accuracy*100:.1f}%</strong></p>
        </div>
        """, unsafe_allow_html=True)

    # ── Evaluation section ─────────────────────────────────────────────────────
    st.markdown("---")
    st.markdown("### 📊 Model Evaluation")

    tab1, tab2, tab3 = st.tabs(["  Confusion Matrix  ", "  Classification Report  ", "  Feature Importance  "])

    with tab1:
        fig_cm = px.imshow(cm, text_auto=True, x=classes, y=classes,
                           labels={"x":"Predicted","y":"Actual"},
                           color_continuous_scale=[[0,"#1A1A2E"],[1,"#6C63FF"]])
        fig_cm.update_layout(title="Confusion Matrix", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_cm, use_container_width=True)

    with tab2:
        st.markdown(f"<pre style='background:rgba(255,255,255,0.04); border:1px solid rgba(108,99,255,0.2); border-radius:12px; padding:18px; color:#C9D1E0; font-size:0.85rem;'>{report}</pre>", unsafe_allow_html=True)

    with tab3:
        imp_df = pd.DataFrame(feature_imp.items(), columns=["Feature","Importance"]).sort_values("Importance")
        fig_fi = px.bar(imp_df, x="Importance", y="Feature", orientation="h",
                        color="Importance", color_continuous_scale=["#1A1A2E","#6C63FF","#43E97B"])
        fig_fi.update_layout(title="Feature Importance", showlegend=False, **PLOTLY_LAYOUT)
        st.plotly_chart(fig_fi, use_container_width=True)

    with st.expander("📂 Preview dataset"):
        from utils.preprocessing import load_student_data
        st.dataframe(load_student_data().head(20), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 3 – REGRESSION
# ══════════════════════════════════════════════════════════════════════════════
elif page == "📈  Salary Regression":

    st.title("📈 Salary Prediction — Linear Regression")
    st.markdown("<p style='color:#8892A4;'>Demonstrates supervised regression: predict salary from years of experience.</p>", unsafe_allow_html=True)

    @st.cache_resource
    def get_regression():
        from models.regression import train_regression
        return train_regression()

    with st.spinner("⚙️ Training regression model…"):
        reg_model, df, metrics, X_test, y_test, y_pred = get_regression()

    # Metrics row
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("MAE",      f"${metrics['MAE']:,.0f}")
    c2.metric("MSE",      f"${metrics['MSE']:,.0f}")
    c3.metric("RMSE",     f"${metrics['RMSE']:,.0f}")
    c4.metric("R² Score", f"{metrics['R2']:.4f}")

    st.markdown("---")

    # Scatter + regression line
    x_range = np.linspace(df["YearsExperience"].min(), df["YearsExperience"].max(), 200)
    y_line  = reg_model.predict(x_range.reshape(-1,1))

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["YearsExperience"], y=df["Salary"],
                             mode="markers", name="Actual",
                             marker=dict(color="#6C63FF", size=9, opacity=0.75,
                                         line=dict(color="#A29BFE", width=1))))
    fig.add_trace(go.Scatter(x=x_range, y=y_line,
                             mode="lines", name="Regression Line",
                             line=dict(color="#FF6584", width=2.5)))
    fig.update_layout(title="Years of Experience vs Salary",
                      xaxis_title="Years of Experience",
                      yaxis_title="Salary ($)", **PLOTLY_LAYOUT)
    st.plotly_chart(fig, use_container_width=True)

    # Residuals
    residuals = y_test - y_pred
    fig_res = go.Figure(go.Bar(x=list(range(len(residuals))), y=residuals,
                               marker_color=["#43E97B" if r >= 0 else "#FF6584" for r in residuals]))
    fig_res.update_layout(title="Residuals (Actual − Predicted) on Test Set",
                          xaxis_title="Sample", yaxis_title="Residual ($)", **PLOTLY_LAYOUT)
    st.plotly_chart(fig_res, use_container_width=True)

    st.markdown("---")
    st.markdown("### 🔮 Predict Your Salary")
    years_input = st.slider("Years of Experience", 0.5, 20.0, 3.0, 0.5)

    if st.button("💰 Predict Salary", use_container_width=True):
        from models.regression import predict_salary
        sal = predict_salary(reg_model, years_input)
        st.markdown(f"""
        <div class="pred-box">
          <div style='color:#8892A4; font-size:0.85rem; margin-bottom:6px;'>💼 Estimated Salary</div>
          <div class="career">${sal:,.0f}</div>
          <div class="sub">For {years_input} years of experience</div>
        </div>
        """, unsafe_allow_html=True)
        st.info(f"📐 Model equation: **Salary ≈ ${reg_model.intercept_:,.0f} + ${reg_model.coef_[0]:,.0f} × YearsExperience**")

    with st.expander("📂 View salary dataset"):
        st.dataframe(df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 4 – K-MEANS CLUSTERING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🔵  Student Clustering":

    st.title("🔵 K-Means Student Clustering")
    st.markdown("<p style='color:#8892A4;'>Unsupervised learning — students grouped into 3 performance clusters.</p>", unsafe_allow_html=True)

    @st.cache_resource
    def get_clustering():
        from models.clustering import run_clustering
        return run_clustering()

    with st.spinner("⚙️ Running K-Means…"):
        df_clust, kmeans, scaler, summary, label_map = get_clustering()

    # Cluster legend cards
    cluster_colors = {"Strong": "#43E97B", "Average": "#F9CA24", "Needs Improvement": "#FF6584"}
    cluster_icons  = {"Strong": "🌟", "Average": "📘", "Needs Improvement": "📈"}
    cols = st.columns(3)
    for i, (label, color) in enumerate(cluster_colors.items()):
        count = int((df_clust["ClusterLabel"] == label).sum())
        cols[i].markdown(f"""
        <div style='background:rgba(255,255,255,0.04); border:1px solid {color}44;
                    border-radius:14px; padding:18px; text-align:center;'>
          <div style='font-size:2rem;'>{cluster_icons[label]}</div>
          <div style='color:{color}; font-weight:700; font-size:1rem;'>{label}</div>
          <div style='color:#8892A4; font-size:0.82rem; margin-top:4px;'>{count} students</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col1, col2 = st.columns(2, gap="large")
    with col1:
        count_df = df_clust["ClusterLabel"].value_counts().reset_index()
        count_df.columns = ["Cluster","Count"]
        fig_pie = px.pie(count_df, names="Cluster", values="Count",
                         color="Cluster",
                         color_discrete_map=cluster_colors,
                         hole=0.45)
        fig_pie.update_layout(title="Student Distribution", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col2:
        fig_scatter = px.scatter(df_clust, x="PreviousMarks", y="MLSkill",
                                 color="ClusterLabel",
                                 color_discrete_map=cluster_colors,
                                 hover_data=["StudyHours","Attendance","PythonSkill"],
                                 size_max=14)
        fig_scatter.update_layout(title="PreviousMarks vs ML Skill by Cluster", **PLOTLY_LAYOUT)
        st.plotly_chart(fig_scatter, use_container_width=True)

    # Radar chart per cluster
    st.markdown("### 📡 Cluster Profile Radar")
    from models.clustering import CLUSTER_FEATURES
    radar_df = df_clust.groupby("ClusterLabel")[CLUSTER_FEATURES].mean()

    fig_radar = go.Figure()
    for label, color in cluster_colors.items():
        if label in radar_df.index:
            vals = radar_df.loc[label].tolist()
            vals += [vals[0]]
            cats = CLUSTER_FEATURES + [CLUSTER_FEATURES[0]]
            fig_radar.add_trace(go.Scatterpolar(r=vals, theta=cats,
                                                fill="toself", name=label,
                                                line_color=color,
                                                fillcolor=color.replace("#","") and f"rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.15)"))
    fig_radar.update_layout(polar=dict(
        bgcolor="rgba(0,0,0,0)",
        radialaxis=dict(gridcolor="rgba(255,255,255,0.1)", linecolor="rgba(255,255,255,0.1)"),
        angularaxis=dict(gridcolor="rgba(255,255,255,0.08)")),
        paper_bgcolor="rgba(0,0,0,0)", font=dict(color="#C9D1E0"),
        legend=dict(bgcolor="rgba(0,0,0,0)"),
        title="Average Feature Profile per Cluster")
    st.plotly_chart(fig_radar, use_container_width=True)

    st.markdown("### 📋 Cluster Averages")
    st.dataframe(summary.set_index("ClusterLabel"), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 5 – DEEP LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🧠  Deep Learning":

    st.title("🧠 Deep Learning Basics")
    st.markdown("<p style='color:#8892A4;'>A small educational Neural Network demonstration using TensorFlow / Keras.</p>", unsafe_allow_html=True)

    # Architecture diagram
    st.markdown("### 🏗️ Network Architecture")
    c1, c2, c3 = st.columns(3, gap="small")
    for col, icon, title, detail, color in [
        (c1, "⬛", "Input Layer",  "8 neurons\n(student features)", "#6C63FF"),
        (c2, "🔷", "Hidden Layer", "16 neurons\nActivation: ReLU",  "#FF6584"),
        (c3, "🟢", "Output Layer", "4 neurons\nActivation: Softmax","#43E97B"),
    ]:
        col.markdown(f"""
        <div style='background:rgba(255,255,255,0.04); border:1px solid {color}55;
                    border-radius:14px; padding:18px; text-align:center;'>
          <div style='font-size:1.8rem;'>{icon}</div>
          <div style='color:{color}; font-weight:700; font-size:0.95rem; margin:6px 0 4px;'>{title}</div>
          <div style='color:#8892A4; font-size:0.8rem; white-space:pre-line;'>{detail}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class="ai-card" style="margin-top:18px;">
      <h4>💡 How it works</h4>
      <p>The input layer receives 8 student features. The hidden layer (ReLU) learns
      non-linear patterns. The output layer (Softmax) produces a probability for each
      of the 4 career classes. Trained with <strong>Adam</strong> optimiser and
      <strong>sparse categorical cross-entropy</strong> loss.</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    if st.button("▶️  Train Neural Network  (30 epochs)", use_container_width=True):
        try:
            from models.deep_learning import build_and_train
            with st.spinner("🔄 Training neural network…"):
                history, test_acc, class_names = build_and_train(epochs=30)

            st.markdown(f"""
            <div class="pred-box">
              <div style='color:#8892A4; font-size:0.85rem;'>🧠 Neural Network Result</div>
              <div class="career">{test_acc:.1f}%</div>
              <div class="sub">Test Accuracy after 30 epochs</div>
            </div>
            """, unsafe_allow_html=True)

            epochs_range = list(range(1, len(history.history["accuracy"]) + 1))
            hist_df = pd.DataFrame({
                "Epoch":          epochs_range,
                "Training":       [v*100 for v in history.history["accuracy"]],
                "Validation":     [v*100 for v in history.history["val_accuracy"]],
                "Train Loss":     history.history["loss"],
                "Val Loss":       history.history["val_loss"],
            })

            col1, col2 = st.columns(2)
            with col1:
                fig_a = px.line(hist_df, x="Epoch", y=["Training","Validation"],
                                title="Accuracy over Epochs (%)",
                                color_discrete_sequence=["#6C63FF","#43E97B"])
                fig_a.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_a, use_container_width=True)
            with col2:
                fig_l = px.line(hist_df, x="Epoch", y=["Train Loss","Val Loss"],
                                title="Loss over Epochs",
                                color_discrete_sequence=["#FF6584","#F9CA24"])
                fig_l.update_layout(**PLOTLY_LAYOUT)
                st.plotly_chart(fig_l, use_container_width=True)

            st.info("ℹ️ On a small synthetic dataset the neural network accuracy may be similar to the Decision Tree. This module is purely for educational demonstration.")

        except ImportError as e:
            st.warning(f"⚠️ TensorFlow not installed.\n\n`{e}`\n\nInstall: `pip install tensorflow`")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 6 – Q-LEARNING
# ══════════════════════════════════════════════════════════════════════════════
elif page == "🤖  Q-Learning Demo":

    st.title("🤖 Q-Learning Demo")
    st.markdown("<p style='color:#8892A4;'>Basic Reinforcement Learning — an agent learns the optimal study path.</p>", unsafe_allow_html=True)

    # Concept cards
    concepts = [
        ("🗺️","State",    "Current milestone the student is at"),
        ("⚡","Action",   "Stay (−0.5) or Advance (+1 / +10 at Goal)"),
        ("🎁","Reward",   "Feedback signal guiding the agent"),
        ("📊","Q-Value",  "Estimated future reward for (state, action)"),
    ]
    cols = st.columns(4)
    for col, (icon, title, desc) in zip(cols, concepts):
        col.markdown(f"""
        <div style='background:rgba(108,99,255,0.09); border:1px solid rgba(108,99,255,0.25);
                    border-radius:13px; padding:16px; text-align:center;'>
          <div style='font-size:1.7rem;'>{icon}</div>
          <div style='color:#6C63FF; font-weight:700; font-size:0.9rem; margin:5px 0 3px;'>{title}</div>
          <div style='color:#8892A4; font-size:0.78rem;'>{desc}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <br>
    <div style='text-align:center; color:#A0AABB; font-size:0.95rem; letter-spacing:1px; margin-bottom:6px;'>
      🎯 &nbsp; Learning Path
    </div>
    <div style='text-align:center; font-size:1.05rem; color:#6C63FF; font-weight:600;'>
      Start &nbsp;→&nbsp; Learn Python &nbsp;→&nbsp; Practice Coding &nbsp;→&nbsp; Learn ML &nbsp;→&nbsp; Complete Project &nbsp;→&nbsp; 🏆 Goal
    </div>
    <br>
    """, unsafe_allow_html=True)

    if st.button("▶️  Run Q-Learning  (200 episodes)", use_container_width=True):
        from models.q_learning import train_q_learning, STATES, ACTIONS

        with st.spinner("🔄 Training Q-Learning agent…"):
            Q_table, episode_rewards, learned_path = train_q_learning(episodes=200)

        # Learned path display
        path_html = " &nbsp;→&nbsp; ".join(
            f"<span style='background:rgba(108,99,255,0.2); border:1px solid #6C63FF55; border-radius:8px; padding:4px 10px; color:#A29BFE; font-weight:600;'>{s}</span>"
            for s in learned_path
        )
        st.markdown(f"""
        <div style='background:rgba(67,233,123,0.07); border:1px solid rgba(67,233,123,0.3);
                    border-radius:14px; padding:18px 22px; margin:14px 0;'>
          <div style='color:#43E97B; font-weight:700; font-size:0.9rem; margin-bottom:10px;'>✅ Learned Path (Greedy Policy)</div>
          <div style='line-height:2.2;'>{path_html}</div>
        </div>
        """, unsafe_allow_html=True)

        col1, col2 = st.columns(2, gap="large")

        with col1:
            # Q-table heatmap
            q_df = pd.DataFrame(Q_table.round(2), index=STATES, columns=ACTIONS)
            fig_q = px.imshow(q_df, text_auto=True,
                              color_continuous_scale=[[0,"#1A1A2E"],[0.5,"#6C63FF"],[1,"#43E97B"]],
                              aspect="auto")
            fig_q.update_layout(title="Q-Table Heatmap", **PLOTLY_LAYOUT)
            st.plotly_chart(fig_q, use_container_width=True)

        with col2:
            # Rewards over episodes
            reward_df = pd.DataFrame({"Episode": range(1, len(episode_rewards)+1),
                                      "Total Reward": episode_rewards})
            fig_r = px.line(reward_df, x="Episode", y="Total Reward",
                            title="Total Reward per Episode",
                            color_discrete_sequence=["#6C63FF"])
            fig_r.update_layout(**PLOTLY_LAYOUT)
            st.plotly_chart(fig_r, use_container_width=True)

        st.info("💡 As episodes progress the agent stops 'exploring' and always chooses 'Advance', so rewards grow and stabilise near the maximum.")


# ══════════════════════════════════════════════════════════════════════════════
#  PAGE 7 – ABOUT
# ══════════════════════════════════════════════════════════════════════════════
elif page == "ℹ️  About":

    st.title("ℹ️ About This Project")

    st.markdown("""
    <div class="hero-banner" style="padding:28px 36px; text-align:left;">
      <h1 style="font-size:1.7rem !important; margin-bottom:6px;">Student AI Analyzer</h1>
      <p style="margin:0;">
        Built as part of the <strong style="color:#6C63FF;">AICTE | IBM SkillsBuild
        Machine Learning & Applied AI Internship 2026</strong>.
        Demonstrates end-to-end ML concepts in a clean, interactive Streamlit app.
      </p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("### 🛠️ Technologies")
        tech = [("Python 3.12","Core language"),("Pandas","Data manipulation"),
                ("NumPy","Numerical ops"),("Scikit-learn","ML algorithms"),
                ("Plotly","Interactive charts"),("Streamlit","Web UI"),
                ("TensorFlow","Deep Learning (optional)")]
        for name, desc in tech:
            st.markdown(f"<span class='badge'>{name}</span> <span style='color:#8892A4; font-size:0.83rem;'>— {desc}</span><br>", unsafe_allow_html=True)

    with col2:
        st.markdown("### 🤖 Algorithms Used")
        algos = [("Decision Tree","Classification","Career Prediction"),
                 ("Linear Regression","Regression","Salary Prediction"),
                 ("K-Means","Unsupervised","Student Clustering"),
                 ("Neural Network","Deep Learning","DL Basics page"),
                 ("Q-Learning","Reinforcement RL","Q-Learning Demo")]
        rows = "".join(f"<tr><td style='color:#A29BFE;padding:5px 10px;'>{a}</td><td style='color:#8892A4;padding:5px 10px;'>{t}</td><td style='color:#43E97B;padding:5px 10px;'>{p}</td></tr>" for a,t,p in algos)
        st.markdown(f"""
        <table style='width:100%; border-collapse:collapse; font-size:0.85rem;'>
          <tr style='border-bottom:1px solid rgba(108,99,255,0.2);'>
            <th style='color:#6C63FF; text-align:left; padding:6px 10px;'>Algorithm</th>
            <th style='color:#6C63FF; text-align:left; padding:6px 10px;'>Type</th>
            <th style='color:#6C63FF; text-align:left; padding:6px 10px;'>Page</th>
          </tr>
          {rows}
        </table>
        """, unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""
    <div style='background:rgba(249,202,36,0.08); border:1px solid rgba(249,202,36,0.3);
                border-radius:14px; padding:18px 22px;'>
      <div style='color:#F9CA24; font-weight:700; margin-bottom:6px;'>⚠️ Dataset Notice</div>
      <div style='color:#A0AABB; font-size:0.88rem; line-height:1.6;'>
        Both datasets (<strong>student_data.csv</strong> and <strong>salary_data.csv</strong>)
        are <strong>synthetically generated</strong> for educational purposes only.
        They do <strong>NOT</strong> contain data from real students.
        Patterns are designed to make the ML workflow clearly understandable.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚠️ Limitations")
    for lim in ["Synthetic dataset — predictions are for demonstration only.",
                "TensorFlow is optional; all other modules work without it.",
                "Only 4 simplified career categories.",
                "No real student data collected or validated."]:
        st.markdown(f"<span style='color:#FF6584;'>▸</span> <span style='color:#A0AABB; font-size:0.88rem;'>{lim}</span>", unsafe_allow_html=True)

    st.markdown("### 🚀 Future Improvements")
    for imp in ["Collect real anonymised student data.",
                "Add more career categories and richer features.",
                "Try Random Forest / XGBoost for higher accuracy.",
                "Add CSV upload so users can train on their own data.",
                "Deploy to Streamlit Cloud or Hugging Face Spaces."]:
        st.markdown(f"<span style='color:#43E97B;'>▸</span> <span style='color:#A0AABB; font-size:0.88rem;'>{imp}</span>", unsafe_allow_html=True)

    st.markdown("""
    <div style='text-align:center; color:#8892A4; font-size:0.75rem; margin-top:36px; padding-top:16px; border-top:1px solid rgba(108,99,255,0.18);'>
      Built for AICTE | IBM SkillsBuild ML & Applied AI Internship 2026 · Synthetic dataset · Educational purpose only
    </div>
    """, unsafe_allow_html=True)
