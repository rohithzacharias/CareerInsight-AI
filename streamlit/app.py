import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE_DIR))
from auth_store import authenticate_user, create_user
from feature_engineering import MODEL_FEATURES, add_placement_features
from career_intelligence import (company_type_suitability, india_entry_salary_ranges, learning_plan,
    load_salary_data, readiness_score, recommended_skills, role_matches, skill_gaps)

MODEL_PATH = BASE_DIR / "models" / "placement_prediction_model.pkl"
USER_DB_PATH = BASE_DIR / "auth" / "users.db"
DEFAULT_PROFILE = {"gender": "Male", "age": 21, "degree": "BTech", "branch": "CS", "cgpa": 7.5,
    "backlogs": 0, "internships": 1, "certifications": 2, "coding_skills": 6,
    "communication_skills": 6, "aptitude_score": 65, "projects": 2, "technical_skills": []}


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def load_salary_dataset():
    return load_salary_data(BASE_DIR)


def probability(profile: dict) -> float:
    student_features = add_placement_features(pd.DataFrame([profile]))
    return float(model.predict_proba(student_features[MODEL_FEATURES])[0][1])


def profile_chart(profile: dict, label: str = "Your Profile", color: str = "#6366f1") -> go.Figure:
    labels = ["Coding", "Communication", "Aptitude", "CGPA", "Projects", "Internships", "Certifications"]
    values = [profile["coding_skills"] * 10, profile["communication_skills"] * 10, profile["aptitude_score"],
              profile["cgpa"] * 10, min(profile["projects"] * 25, 100), min(profile["internships"] * 50, 100), min(profile["certifications"] * 100 / 3, 100)]
    values.append(values[0])
    r, g, b = (99, 102, 241) if color == "#6366f1" else (20, 184, 166)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=values, theta=labels + [labels[0]], fill="toself",
        line=dict(color=color, width=2.5),
        fillcolor=f"rgba({r},{g},{b},0.18)",
        name=label
    ))
    fig.update_layout(
        polar=dict(
            radialaxis=dict(visible=True, range=[0, 100], gridcolor="rgba(51,65,85,0.8)", tickfont=dict(color="#94a3b8", size=10)),
            angularaxis=dict(gridcolor="rgba(51,65,85,0.8)", linecolor="rgba(71,85,105,0.7)", tickfont=dict(color="#e2e8f0", size=11)),
            bgcolor="rgba(0,0,0,0)"
        ),
        showlegend=False,
        margin={"l": 50, "r": 50, "t": 30, "b": 30},
        height=360,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#e2e8f0")
    )
    return fig


def readiness_color(score: float) -> str:
    if score >= 75: return "#10b981"
    if score >= 55: return "#f59e0b"
    return "#ef4444"


def skill_bar_html(label: str, pct: float, color: str = "#6366f1") -> str:
    pct = min(pct, 100)
    return f"""<div class="progress-wrap">
        <div class="progress-label"><span>{label}</span><span>{pct:.0f}%</span></div>
        <div class="progress-track"><div class="progress-fill" style="width:{pct}%;background:{color}"></div></div>
    </div>"""


def insight_card_html(icon: str, title: str, body: str) -> str:
    return f"""<div class="insight-card">
        <div class="insight-icon">{icon}</div>
        <div class="insight-text">
            <div class="title">{title}</div>
            <div class="body">{body}</div>
        </div>
    </div>"""


def inject_css():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Space+Grotesk:wght@400;500;600;700&display=swap');
    * { font-family: 'Inter', sans-serif; }
    .stApp { background: #090d16; color: #e2e8f0; }
    .stApp::before {
        content: '';
        position: fixed; top: -20%; left: -10%;
        width: 500px; height: 500px;
        background: radial-gradient(circle, rgba(99,102,241,0.08) 0%, transparent 70%);
        border-radius: 50%;
        animation: floatOrb1 12s ease-in-out infinite;
        pointer-events: none; z-index: 0;
    }
    .stApp::after {
        content: '';
        position: fixed; bottom: -20%; right: -10%;
        width: 600px; height: 600px;
        background: radial-gradient(circle, rgba(20,184,166,0.06) 0%, transparent 70%);
        border-radius: 50%;
        animation: floatOrb2 15s ease-in-out infinite;
        pointer-events: none; z-index: 0;
    }
    @keyframes floatOrb1 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(40px,60px) scale(1.15); } }
    @keyframes floatOrb2 { 0%,100% { transform: translate(0,0) scale(1); } 50% { transform: translate(-50px,-40px) scale(1.1); } }
    [data-testid="stSidebar"] {
        background: rgba(9,13,22,0.92) !important;
        border-right: 1px solid rgba(255,255,255,0.08) !important;
        backdrop-filter: blur(20px);
    }
    [data-testid="stSidebarContent"] { padding-top: 1.5rem; }
    .hero-banner {
        position: relative; padding: 2.2rem 2.5rem; border-radius: 20px;
        background: rgba(15,23,42,0.64); border: 1px solid rgba(255,255,255,0.10); margin-bottom: 1.8rem; overflow: hidden;
        animation: heroSlideIn 0.6s ease-out; backdrop-filter: blur(10px);
    }
    .hero-banner::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: linear-gradient(90deg, transparent, rgba(99,102,241,.8), rgba(20,184,166,.6), transparent);
        animation: shimmerTop 6s ease-in-out infinite;
    }
    @keyframes shimmerTop { 0%,100% { opacity: 0.5; } 50% { opacity: 1; } }
    @keyframes heroSlideIn { from { opacity: 0; transform: translateY(-18px); } to { opacity: 1; transform: translateY(0); } }
    .hero-banner h1 {
        margin: 0 0 0.4rem 0; font-family: 'Space Grotesk', sans-serif; font-size: 2rem; font-weight: 700;
        color: #f8fafc;
    }
    .hero-banner p { color: #94a3b8; font-size: 1.0rem; margin: 0; }
    .hero-badge {
        display: inline-flex; align-items: center; gap: 6px;
        background: rgba(99,102,241,0.12); border: 1px solid rgba(99,102,241,0.25);
        border-radius: 999px; padding: 4px 12px; font-size: 0.75rem; color: #a5b4fc;
        font-weight: 600; margin-bottom: 0.8rem;
    }
    @keyframes badgePulse { 0%,100% { box-shadow: 0 0 0 0 rgba(139,92,246,0); } 50% { box-shadow: 0 0 12px 3px rgba(139,92,246,0.25); } }
    div[data-testid="stMetric"] {
        background: rgba(15,23,42,0.62) !important; padding: 1.35rem 1.2rem !important;
        border-radius: 12px !important; border: 1px solid rgba(255,255,255,0.09) !important;
        backdrop-filter: blur(12px); transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        position: relative; overflow: hidden; animation: cardFadeUp 0.5s ease-out;
    }
    div[data-testid="stMetric"]::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
        background: #6366f1; border-radius: 12px 12px 0 0;
    }
    div[data-testid="stMetric"]:hover { border-color: rgba(139,92,246,0.55) !important; transform: translateY(-4px); box-shadow: 0 12px 40px rgba(139,92,246,0.2); }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"] { color: #94a3b8 !important; font-size: 0.72rem !important; font-weight: 600 !important; letter-spacing: 0.08em !important; text-transform: uppercase; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] { color: #f8fafc !important; font-size: 1.9rem !important; font-weight: 700 !important; font-family: 'Space Grotesk', sans-serif !important; }
    @keyframes cardFadeUp { from { opacity: 0; transform: translateY(14px); } to { opacity: 1; transform: translateY(0); } }
    .stat-row { display: flex; gap: 14px; flex-wrap: wrap; margin-bottom: 1.6rem; animation: cardFadeUp 0.7s ease-out; }
    .stat-pill {
        flex: 1; min-width: 130px; background: rgba(15,23,42,0.62);
        border: 1px solid rgba(255,255,255,0.09); border-radius: 12px;
        padding: 1rem 1.2rem; text-align: center; position: relative; overflow: hidden;
        transition: all 0.3s; backdrop-filter: blur(12px);
    }
    .stat-pill::after { content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px; border-radius: 14px 14px 0 0; }
    .stat-pill.purple::after { background: #6366f1; }
    .stat-pill.teal::after { background: #14b8a6; }
    .stat-pill.gold::after { background: #0ea5e9; }
    .stat-pill:hover { transform: translateY(-3px); box-shadow: 0 12px 28px rgba(0,0,0,0.24); border-color: rgba(255,255,255,0.16); }
    .stat-pill .label { font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase; color: #94a3b8; margin-bottom: 4px; }
    .stat-pill.teal .label, .stat-pill.gold .label { color: #94a3b8; }
    .stat-pill .value { font-family: 'Space Grotesk', sans-serif; font-size: 1.7rem; font-weight: 700; color: #f8fafc; line-height: 1; }
    .stat-pill .sub { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
    .progress-wrap { margin-bottom: 1rem; }
    .progress-label { display: flex; justify-content: space-between; font-size: 0.82rem; color: #cbd5e1; margin-bottom: 5px; font-weight: 500; }
    .progress-track { background: #1e293b; border-radius: 100px; height: 7px; overflow: hidden; }
    .progress-fill { height: 100%; border-radius: 100px; animation: growBar 1.2s cubic-bezier(0.4,0,0.2,1) forwards; }
    @keyframes growBar { from { width: 0; } }
    .insight-card {
        background: rgba(15,23,42,0.62); border: 1px solid rgba(255,255,255,0.09); border-radius: 12px;
        padding: 1rem 1.2rem; margin-bottom: 0.9rem; display: flex; align-items: center;
        gap: 12px; transition: all 0.3s; backdrop-filter: blur(10px);
    }
    .insight-card:hover { border-color: rgba(99,102,241,0.45); transform: translateX(3px); box-shadow: -4px 0 20px rgba(0,0,0,0.16); }
    .insight-icon { font-size: 1.4rem; width: 40px; height: 40px; border-radius: 10px; background: rgba(99,102,241,0.12); display: flex; align-items: center; justify-content: center; flex-shrink: 0; }
    .insight-text .title { font-size: 0.75rem; font-weight: 600; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.06em; }
    .insight-text .body { font-size: 0.9rem; color: #e2e8f0; font-weight: 500; }
    .section-title { display: flex; align-items: center; gap: 10px; font-family: 'Space Grotesk',sans-serif; font-size: 1.0rem; font-weight: 600; color: #e2e8f0; margin: 1.5rem 0 1rem; padding-bottom: 0.65rem; border-bottom: 1px solid rgba(255,255,255,0.08); }
    .stButton > button { border-radius: 9px !important; border: 1px solid rgba(99,102,241,0.55) !important; background: #4f46e5 !important; color: #ffffff !important; font-weight: 600 !important; letter-spacing: 0; padding: 0.58rem 1.2rem !important; transition: all 0.2s !important; font-size: 0.9rem !important; }
    .stButton > button:hover { background: #6366f1 !important; box-shadow: 0 8px 22px rgba(79,70,229,0.25) !important; transform: translateY(-1px) !important; }
    h2, h3 { font-family: 'Space Grotesk', sans-serif !important; color: #f1f5f9 !important; }
    hr { border-color: rgba(255,255,255,0.08) !important; }
    .stAlert { border-radius: 12px !important; backdrop-filter: blur(10px); }
    .stCaption { color: rgba(196,181,253,0.55) !important; }
    .stDataFrame { border: 1px solid rgba(255,255,255,0.08); border-radius: 12px; overflow: hidden; }
    .stDataFrame thead tr th { background: #111827 !important; color: #94a3b8 !important; }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-baseweb="select"] > div {
        background: #090d16 !important; border-color: #334155 !important; color: #f1f5f9 !important; border-radius: 8px !important;
    }
    [data-testid="stTextInput"] input:focus, [data-testid="stNumberInput"] input:focus { border-color: #6366f1 !important; box-shadow: 0 0 0 1px #6366f1 !important; }
    [data-testid="stSlider"] div[data-baseweb="slider"] [role="slider"] { background: #6366f1 !important; border-color: #c7d2fe !important; }
    [data-testid="stSlider"] div[data-baseweb="slider"] div[role="progressbar"] { background: #6366f1 !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { padding: 0.45rem 0.6rem; border-radius: 8px; color: #94a3b8; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background: rgba(99,102,241,.16); color: #a5b4fc; border-right: 2px solid #6366f1; }
    .stTabs [data-baseweb="tab-list"] { background: #0f172a !important; border-radius: 10px !important; padding: 4px !important; border: 1px solid rgba(255,255,255,.08) !important; gap: 4px; }
    .stTabs [data-baseweb="tab"] { border-radius: 7px !important; color: #94a3b8 !important; font-weight: 500 !important; transition: all .2s !important; }
    .stTabs [aria-selected="true"] { background: rgba(99,102,241,.18) !important; color: #c7d2fe !important; }
    /* Light-mode enterprise override */
    .stApp { background: #f8fafc !important; color: #0f172a !important; }
    .stApp::before, .stApp::after { display: none !important; }
    [data-testid="stSidebar"] { background: rgba(255,255,255,.94) !important; border-right: 1px solid #e2e8f0 !important; }
    .hero-banner { background: #ffffff !important; border: 1px solid #e2e8f0 !important; box-shadow: 0 1px 2px rgba(15,23,42,.04) !important; backdrop-filter: blur(12px); }
    .hero-banner::before { background: #4f46e5 !important; animation: none !important; }
    .hero-banner h1, h2, h3 { color: #0f172a !important; -webkit-text-fill-color: #0f172a !important; }
    .hero-banner p, .stCaption { color: #64748b !important; }
    .hero-badge { background: #f1f5f9 !important; border-color: #e2e8f0 !important; color: #475569 !important; animation: none !important; }
    div[data-testid="stMetric"], .stat-pill, .insight-card { background: #ffffff !important; border-color: #e2e8f0 !important; box-shadow: 0 1px 2px rgba(15,23,42,.04) !important; }
    div[data-testid="stMetric"]::before, .stat-pill::after { background: #4f46e5 !important; }
    div[data-testid="stMetric"] [data-testid="stMetricLabel"], .stat-pill .label, .insight-text .title { color: #64748b !important; }
    div[data-testid="stMetric"] [data-testid="stMetricValue"], .stat-pill .value, .insight-text .body, .section-title { color: #0f172a !important; }
    .stat-pill .sub { color: #64748b !important; }
    .progress-track { background: #e2e8f0 !important; }
    .progress-label { color: #475569 !important; }
    .insight-icon { background: #eef2ff !important; }
    .section-title { border-bottom-color: #e2e8f0 !important; }
    .stButton > button { background: #0f172a !important; border-color: #0f172a !important; box-shadow: none !important; }
    .stButton > button:hover { background: #1e293b !important; border-color: #1e293b !important; }
    .stDataFrame { border-color: #e2e8f0 !important; }
    .stDataFrame thead tr th { background: #f8fafc !important; color: #475569 !important; }
    [data-testid="stTextInput"] input, [data-testid="stNumberInput"] input, [data-baseweb="select"] > div { background: #ffffff !important; border-color: #cbd5e1 !important; color: #0f172a !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { color: #64748b !important; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) { background: #eef2ff !important; color: #4f46e5 !important; border-right-color: #4f46e5 !important; }
    .stTabs [data-baseweb="tab-list"] { background: #f1f5f9 !important; border-color: #e2e8f0 !important; }
    .stTabs [data-baseweb="tab"] { color: #64748b !important; }
    .stTabs [aria-selected="true"] { background: #ffffff !important; color: #4f46e5 !important; box-shadow: 0 1px 2px rgba(15,23,42,.08); }
    </style>""", unsafe_allow_html=True)


def login_screen():
    st.markdown("""<div class='hero-banner'>
        <div class='hero-badge'>Data-backed career analytics</div>
        <h1>CareerInsight AI</h1>
        <p>See a clear, data-backed view of your placement readiness and next career moves.</p>
    </div>""", unsafe_allow_html=True)
    sign_in, create = st.tabs(["Sign in", "Create account"])
    with sign_in:
        with st.form("sign_in_form"):
            email = st.text_input("Email address")
            password = st.text_input("Password", type="password")
            submitted = st.form_submit_button("Sign in", use_container_width=True)
        if submitted:
            authenticated, name = authenticate_user(USER_DB_PATH, email, password)
            if authenticated:
                st.session_state.user_name = name
                st.session_state.user_email = email.strip().lower()
                st.session_state.profile = DEFAULT_PROFILE.copy()
                st.session_state.page = "Dashboard"
                st.rerun()
            st.error("Invalid email address or password.")
    with create:
        with st.form("create_account_form", clear_on_submit=True):
            name = st.text_input("Full name")
            email = st.text_input("Email address", key="new_email")
            password = st.text_input("Password (at least 8 characters)", type="password", key="new_password")
            submitted = st.form_submit_button("Create account", use_container_width=True)
        if submitted:
            success, message = create_user(USER_DB_PATH, name, email, password)
            (st.success if success else st.error)(message)


def profile_form():
    profile = st.session_state.profile
    st.markdown("""<div class='hero-banner'>
        <h1>My Profile</h1>
        <p>These inputs drive your career analysis. Save changes before running it.</p>
    </div>""", unsafe_allow_html=True)
    with st.form("profile_form"):
        left, right = st.columns(2)
        with left:
            gender = st.selectbox("Gender", ["Male", "Female"], index=["Male", "Female"].index(profile["gender"]))
            age = st.number_input("Age", 18, 30, int(profile["age"]))
            degree = st.selectbox("Degree", ["BTech", "BE", "BSc", "BCA"], index=["BTech", "BE", "BSc", "BCA"].index(profile["degree"]))
            branch = st.selectbox("Branch", ["CS", "IT", "AI", "DS", "Electrical", "Mechanical"], index=["CS", "IT", "AI", "DS", "Electrical", "Mechanical"].index(profile["branch"]))
            cgpa = st.slider("CGPA", 5.0, 10.0, float(profile["cgpa"]), .01)
            backlogs = st.number_input("Backlogs", 0, 10, int(profile["backlogs"]))
        with right:
            internships = st.number_input("Internships", 0, 10, int(profile["internships"]))
            certifications = st.number_input("Certifications", 0, 15, int(profile["certifications"]))
            coding = st.slider("Coding skills", 1, 10, int(profile["coding_skills"]))
            communication = st.slider("Communication skills", 1, 10, int(profile["communication_skills"]))
            aptitude = st.slider("Aptitude score", 0, 100, int(profile["aptitude_score"]))
            projects = st.number_input("Projects", 0, 15, int(profile["projects"]))
            technical_skills = st.multiselect(
                "Technical skills you currently have",
                ["Python", "SQL", "Excel", "Power BI", "Data Visualization", "Statistics", "Pandas", "Machine Learning", "Deep Learning", "TensorFlow", "PyTorch", "MLOps", "Docker", "NLP", "Transformers", "Computer Vision", "OpenCV", "CUDA", "Research"],
                default=profile.get("technical_skills", []),
                help="These selections directly influence career-role matching, not the placement ML probability."
            )
        saved = st.form_submit_button("Save profile", use_container_width=True)
    if saved:
        st.session_state.profile = {"gender": gender, "age": age, "degree": degree, "branch": branch, "cgpa": cgpa, "backlogs": backlogs, "internships": internships, "certifications": certifications, "coding_skills": coding, "communication_skills": communication, "aptitude_score": aptitude, "projects": projects, "technical_skills": technical_skills}
        st.session_state.pop("analysis_probability", None)
        st.success("Profile saved successfully.")


def report(profile: dict, placement: float):
    salary_data = load_salary_dataset()
    matches = role_matches(profile)
    best_role = matches.iloc[0]["Role"]
    gaps = skill_gaps(profile, best_role)
    st.markdown("""<div class='hero-banner'>
        <h1>Placement AI Report</h1>
        <p>A focused view of your placement estimate, role fit, and practical next steps.</p>
    </div>""", unsafe_allow_html=True)
    a, b, c = st.columns(3)
    a.metric("Placement Probability", f"{placement * 100:.2f}%")
    b.metric("Career Readiness", f"{readiness_score(profile):.1f}/100")
    c.metric("Best Career Match", best_role, f"{matches.iloc[0]['Profile match']:.1f}% match")
    left, right = st.columns([1, 1.25])
    with left:
        st.markdown("<div class='section-title'>Skill Radar</div>", unsafe_allow_html=True)
        st.plotly_chart(profile_chart(profile), use_container_width=True)
    with right:
        st.markdown("<div class='section-title'>Role Match Comparison</div>", unsafe_allow_html=True)
        top5 = matches.head(5)
        colors = ["#6366f1", "#14b8a6", "#0ea5e9", "#64748b", "#475569"]
        fig = go.Figure(go.Bar(
            x=top5["Profile match"], y=top5["Role"], orientation="h",
            marker=dict(color=colors, opacity=0.9),
            text=[f"{v:.0f}%" for v in top5["Profile match"]],
            textposition="outside", textfont=dict(color="#cbd5e1", size=11)
        ))
        fig.update_layout(
            xaxis_range=[0, 110], xaxis=dict(gridcolor="rgba(51,65,85,.7)", color="#94a3b8"),
            yaxis=dict(autorange="reversed", color="#cbd5e1"),
            height=360, margin={"l": 10, "r": 60, "t": 20, "b": 20},
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1")
        )
        st.plotly_chart(fig, use_container_width=True)
    st.markdown("<div class='section-title'>Estimated Salary Context (India Entry-Level)</div>", unsafe_allow_html=True)
    ranges = india_entry_salary_ranges(salary_data, matches.head(5)["Role"].tolist())
    st.dataframe(ranges, use_container_width=True, hide_index=True)
    st.caption("Observed 25th–75th percentile India entry-level salaries in USD from the provided dataset—not live market data or an offer guarantee.")
    st.markdown(f"<div class='section-title'>Skill Gap for {best_role}</div>", unsafe_allow_html=True)
    st.dataframe(gaps, use_container_width=True, hide_index=True)
    st.write("Suggested topics:", ", ".join(recommended_skills(salary_data, best_role)))
    st.markdown("<div class='section-title'>30-Day Learning Plan</div>", unsafe_allow_html=True)
    for week, actions in learning_plan(gaps):
        with st.expander(week, expanded=week == "Week 1"):
            for action in actions: st.write(f"• {action}")


def simulator(profile: dict, baseline_probability: float):
    st.markdown("""<div class='hero-banner'>
        <h1>Score Simulator</h1>
        <p>Explore hypothetical profile improvements. Not a guarantee of placement or salary.</p>
    </div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        coding = st.slider("Simulated coding", 1, 10, int(profile["coding_skills"]))
        projects = st.slider("Simulated projects", 0, 15, int(profile["projects"]))
    with c2:
        aptitude = st.slider("Simulated aptitude", 0, 100, int(profile["aptitude_score"]))
        internships = st.slider("Simulated internships", 0, 10, int(profile["internships"]))
    with c3:
        communication = st.slider("Simulated communication", 1, 10, int(profile["communication_skills"]))
        certifications = st.slider("Simulated certifications", 0, 15, int(profile["certifications"]))
    simulated = profile | {"coding_skills": coding, "projects": projects, "aptitude_score": aptitude, "internships": internships, "communication_skills": communication, "certifications": certifications}
    new_probability = probability(simulated)
    a, b = st.columns(2)
    a.metric("Simulated Placement Probability", f"{new_probability * 100:.2f}%", f"{(new_probability - baseline_probability) * 100:+.2f} pts")
    b.metric("Simulated Readiness", f"{readiness_score(simulated):.1f}/100", f"{readiness_score(simulated) - readiness_score(profile):+.1f} pts")
    left, right = st.columns(2)
    with left:
        st.markdown("<div class='section-title'>Current Profile</div>", unsafe_allow_html=True)
        st.plotly_chart(profile_chart(profile), use_container_width=True)
    with right:
        st.markdown("<div class='section-title'>Simulated Profile</div>", unsafe_allow_html=True)
        st.plotly_chart(profile_chart(simulated, label="Simulated", color="#06b6d4"), use_container_width=True)


st.set_page_config(page_title="CareerInsight AI", page_icon="C", layout="wide")
inject_css()
model = load_model()
if "user_name" not in st.session_state:
    login_screen()
    st.stop()

if "profile" not in st.session_state:
    st.session_state.profile = DEFAULT_PROFILE.copy()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<div style='text-align:center;padding:1rem 0 0.5rem'>
        <div style='font-family:"Space Grotesk",sans-serif;font-size:1.15rem;font-weight:700;
            color:#0f172a;'>CareerInsight AI</div>
    </div>""", unsafe_allow_html=True)

    profile_now = st.session_state.profile
    rs = readiness_score(profile_now)
    rc = readiness_color(rs)
    st.markdown(f"""<div style='margin:0.8rem 0;padding:0.9rem 1rem;background:rgba(139,92,246,0.1);
        border-radius:12px;border:1px solid rgba(139,92,246,0.2);'>
        <div style='font-size:0.7rem;font-weight:600;letter-spacing:0.08em;text-transform:uppercase;
            color:#9d86e9;margin-bottom:4px;'>Signed in as</div>
        <div style='color:#e2d9ff;font-weight:600;font-size:0.95rem;'>{st.session_state.user_name}</div>
        <div style='margin-top:8px;font-size:0.7rem;color:#9d86e9;margin-bottom:3px;'>Career Readiness</div>
        <div style='background:rgba(139,92,246,0.12);border-radius:100px;height:6px;overflow:hidden;'>
            <div style='width:{rs}%;height:100%;background:{rc};border-radius:100px;transition:width 1s ease;'></div>
        </div>
        <div style='text-align:right;font-size:0.75rem;color:{rc};font-weight:700;margin-top:2px;'>{rs}/100</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='font-size:0.7rem;font-weight:600;letter-spacing:0.1em;text-transform:uppercase;color:#6d5fa8;padding:0 0.2rem 0.4rem;'>Navigate</div>", unsafe_allow_html=True)
    page = st.radio("", ["Dashboard", "My Profile", "Placement AI", "Career Match", "Salary Insights", "Skill Gap", "Progress", "Settings"], key="page", label_visibility="collapsed")
    st.divider()
    if st.button("Logout", use_container_width=True):
        for key in ["user_name", "user_email", "profile", "analysis_probability", "page"]:
            st.session_state.pop(key, None)
        st.rerun()

# ── Pages ─────────────────────────────────────────────────────────────────────
profile = st.session_state.profile

if page == "Dashboard":
    rs = readiness_score(profile)
    matches = role_matches(profile)
    best_role = matches.iloc[0]["Role"]
    match_pct = matches.iloc[0]["Profile match"]

    # Hero banner
    st.markdown(f"""<div class='hero-banner'>
        <div class='hero-badge'>Current profile summary</div>
        <h1>Welcome back, {st.session_state.user_name}</h1>
        <p>Here’s your real-time career analytics breakdown based on your trained dataset.</p>
    </div>""", unsafe_allow_html=True)

    # Animated stat pills
    rc = readiness_color(rs)
    st.markdown(f"""<div class="stat-row">
        <div class="stat-pill purple">
            <div class="label">Career Readiness</div>
            <div class="value">{rs:.0f}<span style="font-size:1rem;color:#c4b5fd">/100</span></div>
            <div class="sub">Overall Score</div>
        </div>
        <div class="stat-pill teal">
            <div class="label">Top Role Match</div>
            <div class="value">{match_pct:.0f}<span style="font-size:1rem;color:#67e8f9">%</span></div>
            <div class="sub">{best_role}</div>
        </div>
        <div class="stat-pill gold">
            <div class="label">Tech Skills</div>
            <div class="value">{len(profile.get('technical_skills', []))}</div>
            <div class="sub">Selected</div>
        </div>
        <div class="stat-pill gold">
            <div class="label">Projects</div>
            <div class="value">{profile['projects']}</div>
            <div class="sub">Portfolio items</div>
        </div>
        <div class="stat-pill purple">
            <div class="label">CGPA</div>
            <div class="value">{profile['cgpa']:.1f}</div>
            <div class="sub">Academic Score</div>
        </div>
        <div class="stat-pill teal">
            <div class="label">Internships</div>
            <div class="value">{profile['internships']}</div>
            <div class="sub">Completed</div>
        </div>
    </div>""", unsafe_allow_html=True)

    # Radar + Skill breakdown
    col_left, col_right = st.columns([1.1, 0.9])

    with col_left:
        st.markdown("<div class='section-title'>Skill Radar</div>", unsafe_allow_html=True)
        st.plotly_chart(profile_chart(profile), use_container_width=True)

    with col_right:
        st.markdown("<div class='section-title'>Skill Breakdown</div>", unsafe_allow_html=True)
        skill_data = [
            ("Coding", profile["coding_skills"] * 10, "#4f46e5"),
            ("Communication", profile["communication_skills"] * 10, "#0284c7"),
            ("Aptitude", float(profile["aptitude_score"]), "#0d9488"),
            ("CGPA", profile["cgpa"] * 10, "#4f46e5"),
            ("Projects", min(profile["projects"] * 25, 100), "#0284c7"),
            ("Certifications", min(profile["certifications"] * 100 / 3, 100), "#0d9488"),
        ]
        bars_html = "".join(skill_bar_html(lbl, pct, color) for lbl, pct, color in skill_data)
        st.markdown(bars_html, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("<div class='section-title'>Key Insights</div>", unsafe_allow_html=True)
        weak_skill = min(skill_data, key=lambda x: x[1])
        improvement_tip = f"Your {weak_skill[0].split(' ',1)[-1]} score ({weak_skill[1]:.0f}%) is lowest — focus here for max impact."
        st.markdown(insight_card_html("", "Recommended Focus", improvement_tip), unsafe_allow_html=True)
        st.markdown(insight_card_html("", "Best Role Match", f"{best_role} — {match_pct:.1f}% compatibility"), unsafe_allow_html=True)
        intern_msg = "Great job! You have solid internship experience." if profile["internships"] >= 2 else "Adding one more internship could boost your readiness significantly."
        st.markdown(insight_card_html("", "Internship Tip", intern_msg), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("Analyze My Career", use_container_width=True):
        with st.status("Analyzing your profile...", expanded=True) as status:
            st.write("Running placement estimation...")
            st.session_state.analysis_probability = probability(profile)
            st.write("Preparing career recommendations...")
            status.update(label="Analysis complete.", state="complete")
        st.success("Analysis ready! Open **Placement AI** in the sidebar to view your full report.")

elif page == "My Profile":
    profile_form()

elif page == "Placement AI":
    if st.button("Refresh Analysis", use_container_width=True):
        st.session_state.analysis_probability = probability(profile)
    if "analysis_probability" not in st.session_state:
        st.session_state.analysis_probability = probability(profile)
    report(profile, st.session_state.analysis_probability)

elif page == "Career Match":
    st.markdown("""<div class='hero-banner'>
        <h1>Career Match</h1>
        <p>Explainable profile-match scores for your top AI career paths.</p>
    </div>""", unsafe_allow_html=True)
    matches = role_matches(profile)
    st.markdown("<div class='section-title'>Top Career Matches</div>", unsafe_allow_html=True)
    for _, match in matches.head(5).iterrows():
        st.markdown(skill_bar_html(f"{match['Role']} — {match['Profile match']:.1f}%", match["Profile match"]), unsafe_allow_html=True)
        st.caption(f"Matched skills: {match['Matched skills']}")
    st.markdown("<div class='section-title'>Company Type Suitability</div>", unsafe_allow_html=True)
    company_fit = company_type_suitability(profile)
    for _, fit in company_fit.iterrows():
        st.markdown(skill_bar_html(f"{fit['Company type']} — {fit['Suitability']:.1f}%", fit["Suitability"], "linear-gradient(90deg,#06b6d4,#10b981)"), unsafe_allow_html=True)
        st.caption(fit["Why this fits"])
    st.caption("These are explainable profile-match scores, not job-market predictions.")

elif page == "Salary Insights":
    salary_data = load_salary_dataset()
    matches = role_matches(profile)
    ranges = india_entry_salary_ranges(salary_data, matches.head(5)["Role"].tolist())
    st.markdown("""<div class='hero-banner'><h1>Salary Insights</h1><p>Dataset-based India entry-level salary context for roles that fit your profile.</p></div>""", unsafe_allow_html=True)
    if ranges.empty:
        st.info("No India entry-level salary records are available for your top matched roles.")
    else:
        best = ranges.merge(matches, on="Role", how="left").sort_values("Profile match", ascending=False)
        primary = best.iloc[0]
        a, b, c = st.columns(3)
        a.metric("Best-matched role", primary["Role"])
        b.metric("Observed range", f"${primary['Lower USD']:,} – ${primary['Upper USD']:,}")
        c.metric("Dataset records", int(primary["Sample size"]))
        fig = go.Figure()
        for _, row in best.iterrows():
            fig.add_trace(go.Scatter(x=[row["Lower USD"], row["Upper USD"]], y=[row["Role"], row["Role"]], mode="lines+markers", line=dict(width=10, color="#8b5cf6"), marker=dict(size=11, color="#67e8f9"), showlegend=False))
        fig.update_layout(xaxis_title="Salary range (USD)", height=360, paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", font=dict(color="#c4b5fd"), xaxis=dict(gridcolor="rgba(139,92,246,.15)"), yaxis=dict(color="#e2d9ff"), margin=dict(l=10,r=10,t=15,b=45))
        st.plotly_chart(fig, use_container_width=True)
        st.dataframe(best[["Role", "Lower USD", "Upper USD", "Sample size", "Profile match"]], use_container_width=True, hide_index=True)
        st.caption("Ranges are 25th–75th percentiles from the supplied dataset, in USD. They are not live salary quotes or guaranteed offers.")

elif page == "Skill Gap":
    target_role = role_matches(profile).iloc[0]["Role"]
    st.markdown(f"""<div class='hero-banner'>
        <h1>Skill Gap &amp; Learning Plan</h1>
        <p>Personalised gap analysis and 30-day roadmap for <strong>{target_role}</strong>.</p>
    </div>""", unsafe_allow_html=True)
    gaps = skill_gaps(profile, target_role)
    st.dataframe(gaps, use_container_width=True, hide_index=True)
    for week, actions in learning_plan(gaps):
        st.subheader(week)
        for action in actions:
            st.write(f"• {action}")

else:
    if page == "Settings":
        st.markdown("""<div class='hero-banner'><h1>Settings</h1><p>View your local account and manage your current analysis session.</p></div>""", unsafe_allow_html=True)
        st.subheader("Account")
        st.write(f"**Name:** {st.session_state.user_name}")
        st.write(f"**Email:** {st.session_state.user_email}")
        st.subheader("Analysis session")
        st.caption("Profile changes already refresh the analysis. You can clear only the current report without deleting your account.")
        if st.button("Clear current analysis"):
            st.session_state.pop("analysis_probability", None)
            st.success("Current analysis cleared.")
    else:
        baseline_probability = st.session_state.get("analysis_probability", probability(profile))
        simulator(profile, baseline_probability)

st.divider()
st.caption("CareerInsight AI provides educational analytical guidance. The salary source contains role and company-type categories, not named companies.")
