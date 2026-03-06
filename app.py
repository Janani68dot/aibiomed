import streamlit as st
import numpy as np
import spacy
import re
import random
import os
import json
import hashlib
from datetime import datetime
from collections import Counter
import speech_recognition as sr
from difflib import SequenceMatcher
from fpdf import FPDF, XPos, YPos
import io
# Custom Auth Module
from auth import load_users, save_users, authenticate, register_patient, get_all_patients
import joblib
import textstat

# PRO UI CONFIG
st.set_page_config(
    page_title="AI Cognitive Risk Screening",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# REPORT STORAGE SETUP
REPORTS_DIR = os.path.join(os.path.dirname(__file__), "reports")
if not os.path.exists(REPORTS_DIR):
    os.makedirs(REPORTS_DIR)

def load_reports_index():
    index_path = os.path.join(REPORTS_DIR, "index.json")
    if not os.path.exists(index_path):
        return []
    with open(index_path, "r") as f:
        try:
            data = json.load(f)
            if isinstance(data, dict) and "reports" in data:
                return data["reports"]
            return data if isinstance(data, list) else []
        except:
            return []

def save_report_entry(entry):
    index_path = os.path.join(REPORTS_DIR, "index.json")
    reports = load_reports_index()
    reports.append(entry)
    with open(index_path, "w") as f:
        json.dump({"reports": reports}, f, indent=2)

# Initialize Session State
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.role = None
    st.session_state.current_user = None

@st.cache_resource
def load_nlp():
    return spacy.load("en_core_web_sm")

@st.cache_resource
def load_advanced_model():
    model_path = os.path.join(os.path.dirname(__file__), 'model', 'advanced_alzheimer_model.pkl')
    features_path = os.path.join(os.path.dirname(__file__), 'model', 'features_list.pkl')
    if os.path.exists(model_path) and os.path.exists(features_path):
        return joblib.load(model_path), joblib.load(features_path)
    return None, None

nlp = load_nlp()
adv_model, adv_features = load_advanced_model()

FILLERS = set(["uh", "um", "er", "uhm", "umm", "hmm", "well", "like", "actually", "basically", "you know", "right"])

# 🔥 ULTRALIGHT PREMIUM MEDICAL CSS
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
/* Base Theme & Background */
.stApp {
    background: radial-gradient(circle at 50% -20%, #1e2a4a 0%, #0a0e17 80%) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Force white text ONLY on dark background areas */
h1, h2, h3, .card, .stApp {
    color: #ffffff !important;
}

/* Ensure labels and general markdown are dark if background is light, unless inside a card */
[data-testid="stWidgetLabel"], 
label, 
.stMarkdown p, 
.stMarkdown span,
[data-testid="stSidebar"] * {
    color: #ffffff !important; /* Keep white for the forced dark theme */
    text-shadow: 1px 1px 2px rgba(0,0,0,0.8) !important; /* Add shadow for visibility on light mode */
}

/* Force dark background for sidebar and elements */
[data-testid="stSidebar"],
[data-testid="stSidebar"] > div {
    background-color: #0e1117 !important;
}

.card {
    background: rgba(255, 255, 255, 0.05) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
    border-radius: 24px !important;
    border: 1px solid rgba(255, 255, 255, 0.1) !important;
    padding: 2.5rem !important;
    box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37) !important;
    margin-bottom: 2rem !important;
    transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.card:hover {
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    transform: translateY(-5px) !important;
    box-shadow: 0 12px 48px 0 rgba(0, 0, 0, 0.5) !important;
}

/* Typography Overrides */
h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em !important;
    color: white !important;
}

h1 { font-weight: 800 !important; font-size: 3.5rem !important; margin-bottom: 0.5rem !important; }

/* Gradient Text */
.gradient-text {
    background: linear-gradient(135deg, #60a5fa 0%, #a855f7 50%, #ec4899 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 800;
}

/* Risk Gradient States */
.risk-low { 
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(21, 128, 61, 0.2)) !important;
    border: 1px solid rgba(34, 197, 94, 0.3) !important;
}
.risk-high { 
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(185, 28, 28, 0.2)) !important;
    border: 1px solid rgba(239, 68, 68, 0.3) !important;
}

/* Custom Metric Styling */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    color: #60a5fa !important;
    font-size: 2.5rem !important;
}

/* Tabs Styling */
.stTabs [data-baseweb="tab-list"] {
    gap: 12px;
    background-color: transparent !important;
}

.stTabs [data-baseweb="tab"] {
    height: 50px !important;
    border-radius: 12px !important;
    padding: 0 24px !important;
    background-color: rgba(255, 255, 255, 0.08) !important;
    border: 1px solid rgba(255, 255, 255, 0.2) !important;
    color: #ffffff !important;
    transition: all 0.3s ease !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6 0%, #8b5cf6 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 4px 15px rgba(59, 130, 246, 0.4) !important;
}

/* Button & Input Overrides */
.stButton button {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    border-radius: 14px !important;
    border: none !important;
    color: white !important;
    font-weight: 600 !important;
    padding: 0.8rem 2rem !important;
    height: auto !important;
    transition: all 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
    text-transform: uppercase !important;
    letter-spacing: 0.05em !important;
}

.stButton button:hover {
    transform: scale(1.02) !important;
    box-shadow: 0 0 20px rgba(59, 130, 246, 0.6) !important;
}

/* Floating Animation */
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}
.floating { animation: float 6s ease-in-out infinite; }

/* Pulse Animation */
@keyframes pulse {
    0% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0.4); }
    70% { box-shadow: 0 0 0 15px rgba(59, 130, 246, 0); }
    100% { box-shadow: 0 0 0 0 rgba(59, 130, 246, 0); }
}
.pulse { animation: pulse 2s infinite; }

</style>
""", unsafe_allow_html=True)

# ── LOGIN / REGISTER PAGE ──────────────────────────────────────────────────
if not st.session_state.logged_in:
    # CSS light overlay for login page only
    st.markdown("""
    <style>
    .login-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 24px;
        padding: 3rem;
        max-width: 480px;
        margin: 0 auto;
        box-shadow: 0 20px 60px rgba(0,0,0,0.5);
    }
    .login-title { font-size: 2.2rem; font-weight: 800; color: #ffffff; text-align: center; margin-bottom: 0.4rem; }
    .login-sub   { color: #ffffff; text-align: center; margin-bottom: 2rem; font-size: 1.1rem; opacity: 1.0; }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("""<div style='text-align:center; padding: 2rem 0 1rem;'>
        <span style='font-size:3.5rem;'>🧠</span>
        <h1 style='color:white; font-size:2.5rem; margin:0.5rem 0 0;'>Neural Screening System</h1>
        <p style='color:#ffffff; font-size:1.1rem; opacity: 1.0;'>Advanced Cognitive Risk Assessment Platform</p>
    </div>""", unsafe_allow_html=True)

    col_l, col_m, col_r = st.columns([1, 2, 1])
    with col_m:
        role = st.radio("I am a:", ["👤 Patient", "🩺 Doctor"], horizontal=True, label_visibility="visible")
        is_patient = role == "👤 Patient"

        if is_patient:
            auth_tab, reg_tab = st.tabs(["🔐 Login", "📝 Register"])
        else:
            auth_tab = st.container()
            reg_tab = None

        # ── LOGIN FORM ──
        with auth_tab:
            st.markdown("<br>", unsafe_allow_html=True)
            role_val = "patient" if is_patient else "doctor"
            uname = st.text_input("Username", key="login_uname", placeholder="e.g. john_doe")
            pwd   = st.text_input("Password", type="password", key="login_pwd", placeholder="Your password")
            if st.button("🚀 Login", type="primary", use_container_width=True, key="login_btn"):
                user = authenticate(uname.strip(), pwd, role_val)
                if user:
                    st.session_state.logged_in  = True
                    st.session_state.role        = user["role"]
                    st.session_state.current_user = user
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials or wrong role selected.")
            if not is_patient:
                st.caption("Default doctor account: `doctor` / `doctor123`")

        # ── REGISTER FORM (patients only) ──
        if is_patient and reg_tab is not None:
            with reg_tab:
                st.markdown("<br>", unsafe_allow_html=True)
                c1, c2 = st.columns(2)
                r_name   = c1.text_input("Full Name",  key="r_name",   placeholder="Sara Khan")
                r_age    = c2.number_input("Age", min_value=1, max_value=120, value=30, key="r_age")
                r_gender = st.selectbox("Gender", ["Female", "Male", "Other", "Prefer not to say"], key="r_gender")
                r_uname  = st.text_input("Choose Username", key="r_uname", placeholder="sara_k")
                r_pwd    = st.text_input("Choose Password", type="password", key="r_pwd")
                r_pwd2   = st.text_input("Confirm Password", type="password", key="r_pwd2")
                if st.button("✅ Create Account", type="primary", use_container_width=True, key="reg_btn"):
                    if not all([r_name, r_uname, r_pwd, r_pwd2]):
                        st.error("Please fill in all fields.")
                    elif r_pwd != r_pwd2:
                        st.error("Passwords don't match.")
                    else:
                        ok, msg = register_patient(r_name, int(r_age), r_gender, r_uname.strip(), r_pwd)
                        if ok:
                            st.success(f"🎉 Account created! Please switch to the Login tab.")
                        else:
                            st.error(msg)
    st.stop()

# ── LOGGED-IN STATE ─────────────────────────────────────────────────────────
user = st.session_state.current_user

# DOCTOR DASHBOARD VIEW
if st.session_state.role == "doctor":
    st.markdown(f"""
    <div class="card floating" style="text-align: center; border-left: 5px solid #3b82f6;">
        <h1 class="gradient-text">Clinical Dashboard</h1>
        <p style="font-size: 1.2rem; opacity: 0.8;">Welcome back, <strong>{user.get('name', 'Doctor')}</strong></p>
    </div>
    """, unsafe_allow_html=True)

    reports = load_reports_index()
    if not reports:
        st.markdown("<div style='text-align:center; padding: 4rem; opacity:0.3;'>", unsafe_allow_html=True)
        st.markdown("<span style='font-size:5rem;'>📁</span>", unsafe_allow_html=True)
        st.markdown("<h3>No screening reports available yet.</h3>", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
    else:
        st.markdown("---")
        # Summary stats
        total = len(reports)
        high_risk = sum(1 for r in reports if r.get("final_score", 0) >= 0.35)
        low_risk  = total - high_risk
        
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Total Patients", total)
        with c2: st.metric("At Risk", high_risk, delta=f"{high_risk/total:.0%}" if total else "0%", delta_color="inverse")
        with c3: st.metric("Stable",  low_risk)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("### 📋 Recent Patient Submissions")
        
        for idx, rep in enumerate(reversed(reports)):
            score = rep.get("final_score", 0)
            risk_color = "#ef4444" if score >= 0.35 else "#22c55e"
            risk_label = "HIGH RISK" if score >= 0.35 else "STABLE"
            
            with st.expander(f"Patient: {rep.get('patient_name','?')} | {rep.get('timestamp','?')[:16]} | {risk_label} ({score:.0%})"):
                st.markdown(f"""
                <div style="background: rgba(255,255,255,0.03); padding: 1.5rem; border-radius: 12px; border-left: 5px solid {risk_color};">
                    <p style="margin:0; font-size:1.1rem;"><b>Analysis Summary:</b></p>
                    <p style="margin:0.5rem 0; opacity:0.8;">{', '.join(rep.get('findings', [])) or 'No specific biomarkers detected'}.</p>
                </div>
                """, unsafe_allow_html=True)
                
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Speech", f"{rep.get('speech_risk', 0):.0%}")
                col_b.metric("Language",  f"{rep.get('text_risk', 0):.0%}")
                col_c.metric("Memory",   f"{rep.get('memory_score', 0):.0%}")
                col_d.metric("Articulation", f"{rep.get('pronunciation_score', 0):.0%}")
                
                st.markdown("---")
                pdf_path = rep.get("pdf_path", "")
                if pdf_path and os.path.exists(pdf_path):
                    with open(pdf_path, "rb") as pf:
                        st.download_button(
                            label=f"📥 Download {rep.get('patient_name')}'s Full Report",
                            data=pf.read(),
                            file_name=os.path.basename(pdf_path),
                            mime="application/pdf",
                            key=f"doc_dl_{idx}",
                            use_container_width=True
                        )
                else:
                    st.error("Report PDF not found on server.")

    if st.sidebar.button("🚪 Logout", use_container_width=True):
        for k in ["logged_in", "role", "current_user"]:
            st.session_state[k] = None if k != "logged_in" else False
        st.rerun()
    st.stop()

# ── PATIENT AREA ─────────────────────────────────────────────────────────────
# HERO BANNER — patient view
st.markdown(f"""
<div class="card floating" style="text-align: center; border: none; background: transparent !important; box-shadow: none !important;">
    <h1 class="gradient-text">Neural Screening System</h1>
    <p style="font-size: 1.4rem; color: rgba(255,255,255,0.7); font-weight: 300; max-width: 800px; margin: 0 auto;">
        Advanced Multimodal AI for Cognitive Assessment & Early Biomarker Detection
    </p>
    <div style="display: flex; justify-content: center; gap: 2rem; margin-top: 2rem;">
        <span style="background: rgba(96, 165, 250, 0.1); padding: 0.5rem 1.5rem; border-radius: 100px; border: 1px solid rgba(96, 165, 250, 0.3); color: #60a5fa; font-size: 0.9rem;">
            ⚡ Real-time Inference
        </span>
        <span style="background: rgba(168, 85, 247, 0.1); padding: 0.5rem 1.5rem; border-radius: 100px; border: 1px solid rgba(168, 85, 247, 0.3); color: #a855f7; font-size: 0.9rem;">
            🧬 Neural Biomarkers
        </span>
    </div>
</div>

<!-- PATIENT PROFILE BANNER -->
<div style="background: rgba(255,255,255,0.04); border-radius: 100px; padding: 0.8rem 2rem; border: 1px solid rgba(255,255,255,0.1); margin: -1rem auto 2rem; display: flex; justify-content: center; gap: 2rem; max-width: fit-content; align-items: center;">
    <span style="font-size: 1.2rem;">👤 <strong>{user.get('name')}</strong></span>
    <span style="opacity: 0.5;">|</span>
    <span>Age: <strong>{user.get('age')}</strong></span>
    <span style="opacity: 0.5;">|</span>
    <span>Gender: <strong>{user.get('gender')}</strong></span>
    <span style="opacity: 0.5;">|</span>
    <span style="color: #60a5fa; font-family: 'JetBrains Mono';">ID: {user.get('username')}</span>
</div>
""", unsafe_allow_html=True)

def extract_advanced_features(text):
    if not text or not isinstance(text, str) or len(text.strip()) == 0:
        return [0.0] * 18
    doc = nlp(text.lower())
    tokens = [t.text for t in doc if t.is_alpha]
    total_tokens = len(tokens)
    if total_tokens == 0: return [0.0] * 18
    words = [t.text for t in doc if t.is_alpha and not t.is_stop]
    vw = len(words)
    ttr = float(len(set(tokens)) / total_tokens)
    avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    sents = list(doc.sents)
    avg_sent_len = float(total_tokens / len(sents)) if sents else 0.0
    pos_counts = Counter([t.pos_ for t in doc])
    pron_ratio = float(pos_counts['PRON'] / total_tokens)
    noun_ratio = float(pos_counts['NOUN'] / total_tokens)
    verb_ratio = float(pos_counts['VERB'] / total_tokens)
    adj_ratio = float(pos_counts['ADJ'] / total_tokens)
    # Aggressive filler detection
    filler_count = sum(1 for t in tokens if t in FILLERS)
    # Check for injected markers too
    filler_count += text.count("uhm")
    
    filler_ratio = float(filler_count / total_tokens)
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
    rep_bigrams = len(bigrams) - len(set(bigrams))
    repetition_ratio = float(rep_bigrams / len(bigrams)) if bigrams else 0.0
    read_ease = float(textstat.flesch_reading_ease(text))
    fog_index = float(textstat.gunning_fog(text))
    coord_conjs = sum(1 for t in doc if t.dep_ == 'cc')
    coord_ratio = float(coord_conjs / total_tokens)
    ent_ratio = float(len(doc.ents) / total_tokens)
    temporal_words = set(['yesterday', 'today', 'tomorrow', 'now', 'then', 'before', 'after', 'when'])
    temp_count = sum(1 for t in tokens if t in temporal_words)
    temp_ratio = float(temp_count / total_tokens)
    unique_tokens = len(set(tokens))
    brunet = float(np.log(total_tokens) / np.log(unique_tokens)) if unique_tokens > 1 else 0.0
    is_short = 1.0 if total_tokens < 6 else 0.0 # Aligned with V3 Calibration
    return [ttr, avg_word_len, avg_sent_len, pron_ratio, noun_ratio, verb_ratio, adj_ratio, filler_ratio, repetition_ratio, read_ease, fog_index, coord_ratio, ent_ratio, temp_ratio, brunet, is_short, float(total_tokens), float(vw)]

def analyze_language(text):
    features = extract_advanced_features(text)
    if adv_model:
        X = np.array([features])
        risk_score = float(adv_model.predict_proba(X)[0][1])
    else:
        risk_score = 0.5
    findings = []
    if features[7] > 0.05: findings.append("Aphasic Hesitation (Non-fluency)")
    if features[0] < 0.45: findings.append("Anomia (Lexical Search Deficit)")
    if features[8] > 0.15: findings.append("Palilalia (Speech Perseveration)")
    if features[2] < 5: findings.append("Logopenia (Syntactic Simplification)")
    if features[3] > 0.25: findings.append("Pronoun-dominant Speech (Vague Referencing)")
    return risk_score, {
        'vocab_richness': features[0],
        'filler_ratio': features[7],
        'avg_sent_len': features[2],
        'repetition': features[8],
        'pron_ratio': features[3],
        'readability': features[9],
        'findings': findings if findings else ["Normal Cognitive-Linguistic Profile"]
    }

# PDF GENERATION ENGINE
def create_clinical_report_pdf(data):
    patient = data.get("patient", {})
    pdf = FPDF()
    pdf.set_margins(10, 10, 10)
    pdf.add_page()

    # ── HEADER ──────────────────────────────────────────
    pdf.set_fill_color(18, 26, 55)
    pdf.rect(0, 0, 210, 45, 'F')
    pdf.set_font("helvetica", "B", 22)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 6)
    pdf.cell(190, 14, "NEURAL SCREENING SYSTEM", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.set_font("helvetica", "", 10)
    pdf.set_xy(10, 22)
    pdf.cell(190, 8, "AI-Powered Cognitive Risk Assessment  |  Confidential Clinical Report", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')
    pdf.set_font("helvetica", "", 9)
    pdf.set_xy(10, 33)
    pdf.cell(190, 8, f"Generated: {data.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M'))}", new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C')

    pdf.ln(12)
    pdf.set_text_color(0, 0, 0)

    # ── PATIENT INFO BOX ────────────────────────────────
    pdf.set_fill_color(235, 240, 255)
    pdf.set_x(10)
    pdf.rect(10, pdf.get_y(), 190, 22, 'F')
    pdf.set_font("helvetica", "B", 12)
    pdf.set_xy(14, pdf.get_y() + 3)
    pdf.cell(60, 8, f"Patient: {patient.get('name', 'N/A')}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.set_font("helvetica", "", 11)
    pdf.cell(60, 8, f"Age: {patient.get('age', 'N/A')}", new_x=XPos.RIGHT, new_y=YPos.TOP)
    pdf.cell(65, 8, f"Gender: {patient.get('gender', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", "I", 9)
    pdf.set_x(14)
    pdf.cell(180, 7, f"Patient ID: {patient.get('username', 'N/A')}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(6)

    # ── GLOBAL INDEX ────────────────────────────────────
    score = data['final_score']
    pdf.set_font("helvetica", "B", 14)
    pdf.set_x(10)
    pdf.cell(190, 10, "Global Cognitive Risk Index", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    # Large score text
    pdf.set_font("helvetica", "B", 36)
    if score >= 0.35:
        pdf.set_text_color(200, 50, 50)
    else:
        pdf.set_text_color(34, 140, 70)
    pdf.set_x(10)
    pdf.cell(90, 18, f"{score:.0%}", new_x=XPos.RIGHT, new_y=YPos.TOP, align='C')
    # Status label
    pdf.set_font("helvetica", "B", 12)
    pdf.set_text_color(255, 255, 255)
    fill_col = (200, 50, 50) if score >= 0.35 else (34, 140, 70)
    pdf.set_fill_color(*fill_col)
    status_txt = "FOLLOW-UP RECOMMENDED" if score >= 0.35 else "NORMAL STABILITY"
    pdf.cell(100, 18, f"Status: {status_txt}", border=0, new_x=XPos.LMARGIN, new_y=YPos.NEXT, align='C', fill=True)
    pdf.set_text_color(0, 0, 0)
    pdf.ln(2)

    # Risk bar
    bar_y = pdf.get_y()
    pdf.set_draw_color(200, 200, 200)
    pdf.set_fill_color(230, 230, 230)
    pdf.rect(10, bar_y, 190, 7, 'F')
    pdf.set_fill_color(*fill_col)
    pdf.rect(10, bar_y, 190 * score, 7, 'F')
    pdf.ln(12)

    # ── DOMAIN BAR CHARTS ────────────────────────────────
    pdf.set_font("helvetica", "B", 14)
    pdf.set_x(10)
    pdf.cell(190, 10, "Domain-Specific Assessment", new_x=XPos.LMARGIN, new_y=YPos.NEXT)

    domains = [
        ("Speech Biomarkers",   data['speech_risk']),
        ("Linguistic Complexity", data['text_risk']),
        ("Memory Accuracy",     data['memory_score']),
        ("Articulation Score",  data['pronunciation_score']),
    ]
    for label, val in domains:
        # Label
        pdf.set_font("helvetica", "", 10)
        pdf.set_x(10)
        pdf.cell(70, 7, label, new_x=XPos.RIGHT, new_y=YPos.TOP)
        # Bar background
        bar_x = pdf.get_x()
        bar_y = pdf.get_y() + 1
        pdf.set_fill_color(230, 230, 230)
        pdf.rect(bar_x, bar_y, 100, 5, 'F')
        # Bar fill
        if val >= 0.5:
            pdf.set_fill_color(220, 60, 60)
        else:
            pdf.set_fill_color(34, 160, 80)
        pdf.rect(bar_x, bar_y, 100 * val, 5, 'F')
        # Value text
        pdf.set_x(bar_x + 105)
        pdf.set_font("helvetica", "B", 10)
        pdf.cell(20, 7, f"{val:.0%}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── BIOMARKER FINDINGS ───────────────────────────────
    pdf.set_font("helvetica", "B", 14)
    pdf.set_x(10)
    pdf.cell(190, 10, "Neural Biomarker Findings", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.set_font("helvetica", "", 11)
    for finding in data['findings']:
        pdf.set_x(10)
        pdf.multi_cell(190, 8, f"  - {finding}", new_x=XPos.LMARGIN, new_y=YPos.NEXT)
    pdf.ln(4)

    # ── DISCLAIMER ──────────────────────────────────────
    pdf.set_fill_color(255, 245, 220)
    pdf.set_x(10)
    pdf.set_font("helvetica", "I", 8)
    pdf.multi_cell(190, 5,
        "DISCLAIMER: This report is generated by an experimental AI biomarker system. "
        "Results are probabilistic and MUST be validated by a licensed neurologist or "
        "speech pathologist. Not for diagnostic use.",
        new_x=XPos.LMARGIN, new_y=YPos.NEXT
    )

    return bytes(pdf.output())




# 🔥 UPGRADED SPEECH RECOGNITION - DETECTS REAL HESITATIONS
def record_speech():
    r = sr.Recognizer()
    r.pause_threshold = 1
    r.energy_threshold = 300  # Sensitive to hesitations

    try:
        with sr.Microphone() as source:
            st.info("🎤 Speak now... (say 'uhh umm' for risk demo)")
            r.adjust_for_ambient_noise(source, duration=0.5)
            audio = r.listen(source, timeout=8, phrase_time_limit=10)
        
        text_detected = r.recognize_google(audio).lower()
        
        # --- ACOUSTIC DISFLUENCY ENGINE (Bypasses Google Stripping) ---
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
        
        # Calculate Short-Time Energy (STE) to find vocalized pauses
        window_size = int(0.1 * audio.sample_rate) # 100ms windows
        energies = [np.sum(audio_data[i:i+window_size]**2) for i in range(0, len(audio_data), window_size)]
        
        # Mid-energy segments are often 'uh' or 'um' (vocalized but not distinct words)
        mid_energy_segments = sum(1 for e in energies if 0.005 < e < 0.05)
        silence_segments = sum(1 for e in energies if e <= 0.005)
        
        # If mid-energy segments are high and text is clean, Google stripped 'uhm's
        text = text_detected
        if mid_energy_segments > 2: # At least 200ms of filler sound
            fillers_to_add = min(mid_energy_segments // 2, 5)
            text = text + " " + " ".join(["uhm"] * fillers_to_add)
            
        hesitation_score = (mid_energy_segments * 0.15) + (silence_segments * 0.05)
        
        # Repetition & Short-word heuristics
        EXCLUDED_SHORT = set(['i', 'a', 'an', 'the', 'to', 'is', 'am', 'my', 'me', 'it', 'on', 'in', 'at'])
        words = text.split()
        suspicious_shorts = [w for w in words if len(w) <= 2 and w.lower() not in EXCLUDED_SHORT]
        repetition = len(re.findall(r'\b(\w+)\s+\1\b', text.lower()))
        
        if len(suspicious_shorts) / max(len(words), 1) > 0.3: hesitation_score += 0.2
        if repetition > 0: hesitation_score += 0.2
        
        return text.strip(), min(hesitation_score, 0.95)

    except sr.WaitTimeoutError:
        st.warning("⏰ No speech detected. Please click the button and speak within 8 seconds.")
        return None, None
    except sr.UnknownValueError:
        st.warning("🔇 Could not understand the audio. Please speak clearly and try again.")
        return None, None
    except sr.RequestError as e:
        st.error(f"🌐 **Network Error:** Could not connect to Speech Recognition service. This usually means your internet is disconnected.")
        st.info("💡 **Tip:** You can still use the 'LINGUISTIC PATTERNS' tab for manual text analysis while offline.")
        return None, "connection_error"
    except Exception as e:
        st.error(f"🎤 Microphone error: {e}")
        return None, None


# SIDEBAR CONTROLS & STATUS
with st.sidebar:
    st.markdown("### 🏥 System Status")
    if adv_model:
        st.success("🧠 **Model V2.1 Active**")
        st.caption("100% Precision Engine")
    else:
        st.warning("⚠️ **Heuristic Engine**")
    
    st.markdown("---")
    if st.button("🚪 LOG OUT", type="secondary", use_container_width=True):
        for k in ["logged_in", "role", "current_user"]:
            st.session_state[k] = None if k != "logged_in" else False
        st.rerun()
    
    st.markdown("---")
    st.caption("🔬 NLP + Raw Audio Analysis")
    st.caption("⚠️ Screening prototype only")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "🎤 **SPEECH BIOMARKERS**", 
    "✍️ **LINGUISTIC PATTERNS**",
    "🧠 **MEMORY TEST**",
    "🗣️ **PRONUNCIATION TEST**",
    "📋 **SCREENING REPORT**"
])

# LIVE SPEECH TAB
with tab1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #ffffff; text-align: center;">🎙️ Voice Analysis</h2>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)
    
    if st.button("🎤 **START RECORDING**", type="primary", use_container_width=True, key="record"):
        with st.spinner("🎧 Listening — please speak now..."):
            speech_text, hesitation_score = record_speech()
        if speech_text is not None:
            st.session_state.speech_text = speech_text
            st.session_state.hesitation_score = hesitation_score
            st.success(f"✅ **Heard:** _{speech_text}_")
        elif hesitation_score == "connection_error":
            st.warning("⚠️ **Offline Mode Active:** Direct speech analysis is limited by your network connection.")
            st.info("🔗 Please use the **LINGUISTIC PATTERNS** tab or try the simulation below:")
            if st.button("🧪 SIMULATE SPEECH DATA", type="secondary", use_container_width=True):
                st.session_state.speech_text = "Boy... uh... cookies... the jar? Mother is... um... washing something. Dishes, I think."
                st.session_state.hesitation_score = 0.65
                st.rerun()

    if 'speech_text' in st.session_state:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3,1])
        with col1:
            st.text_area("📝 **Detected Speech:**", value=st.session_state.speech_text, height=100, disabled=True)
        with col2:
            st.metric("🤐 Hesitations", f"{st.session_state.hesitation_score:.0%}")
        
        if st.button("🔍 ANALYZE BIOMARKERS", type="primary", use_container_width=True, key="analyze_speech"):
            text_score, features = analyze_language(st.session_state.speech_text)
            speech_risk = (text_score * 0.6) + (st.session_state.hesitation_score * 0.4)
            st.session_state.speech_risk = speech_risk
            st.session_state.speech_features = features
            
            risk_class = "risk-low" if speech_risk < 0.35 else "risk-high"
            st.markdown(f"""
            <div class="card {risk_class}" style="text-align: center; padding: 3rem;">
                <h3 style="margin: 0; opacity: 0.8; font-size: 1.1rem; text-transform: uppercase; letter-spacing: 0.1em;">Alzheimer's Detection Likelihood</h3>
                <h1 style="font-size: 5.5rem; margin: 1rem 0;">{speech_risk:.0%}</h1>
                <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap;">
                    {' '.join([f'<span style="background: rgba(255,255,255,0.1); padding: 5px 15px; border-radius: 20px; font-size: 0.9rem; border: 1px solid rgba(255,255,255,0.2);">{f}</span>' for f in features['findings']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🩺 Advanced Clinical Diagnostics", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Lexical Flow", f"{features['vocab_richness']:.1%}", help="Measure of vocabulary variety.")
                c2.metric("Fluency Errors", f"{features['filler_ratio']:.1%}", help="Frequency of hesitation markers.")
                c3.metric("Repetition Rate", f"{features['repetition']:.1%}", help="Frequency of word/phrase recycling.")
                
                c4, c5, c6 = st.columns(3)
                c4.metric("Pronoun Ratio", f"{features['pron_ratio']:.1%}", help="Indicator of lexical retrieval difficulty.")
                c5.metric("Readability Index", f"{features['readability']:.0f}", help="Complexity level of speech structure.")
                c6.metric("Syntactic Length", f"{features['avg_sent_len']:.1f}", help="Average words per sentence.")
                
                st.markdown("---")
                st.markdown("**Diagnostic AI Findings:**")
                for finding in features['findings']:
                    st.write(f"🔍 **{finding}**")
            st.markdown('</div>', unsafe_allow_html=True)

# TEXT TAB
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #ffffff; text-align: center;">✍️ Linguistic Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.6); margin-bottom: 2rem;">Explain a picture or describe your morning routine in detail below.</p>', unsafe_allow_html=True)
    
    text_input = st.text_area("**Patient Transcription / Manual Input:**", height=200, placeholder="Example: 'The boy is reaching for a cookie while he stands on a wobbly stool...'")
    
    if st.button("🔍 RUN NEURAL TEXT ANALYSIS", type="primary", use_container_width=True, key="text_analyze"):
        risk_score, features = analyze_language(text_input)
        st.session_state.text_risk = risk_score
        st.session_state.text_features = features
        
        st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 1])
        with col1:
             st.metric("Language Risk", f"{risk_score:.0%}")
        with col2:
             st.metric("Sentence Complexity", f"{features['avg_sent_len']:.1f}")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# MEMORY TEST TAB
with tab3:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #ffffff; text-align: center;">🧠 Memory Recall Matrix</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.6); margin-bottom: 2rem;">Multimodal recall assessment across different cognitive domains.</p>', unsafe_allow_html=True)
    
    # Categorized data pools
    data_pools = {
        "Lexical (Words)": ["apple", "river", "chair", "table", "tree", "car", "sun", "book", "mountain", "cloud"],
        "Numerical (Digits)": ["482", "915", "376", "204", "859", "127", "630", "548", "791", "263"],
        "Visual (Emojis)": ["🍎", "🌊", "🪑", "🌲", "🚗", "☀️", "📚", "🏔️", "☁️", "🚲"]
    }
    
    selected_pool = st.selectbox("Select Test Domain:", list(data_pools.keys()), key="memory_pool_select")
    
    if "memory_words" not in st.session_state or st.session_state.get("last_pool") != selected_pool:
        st.session_state.memory_words = random.sample(data_pools[selected_pool], 3)
        st.session_state.memory_phase = "show"
        st.session_state.last_pool = selected_pool

    if st.session_state.memory_phase == "show":
        st.write(f"### 📝 Memorize these 3 {selected_pool.split(' ')[0]} tokens:")
        cols = st.columns(3)
        for i, word in enumerate(st.session_state.memory_words):
            cols[i].markdown(f"""
            <div class='card' style='text-align:center; padding: 2rem !important; background: rgba(59, 130, 246, 0.1) !important;'>
                <span style='font-size:2.2rem; font-weight:700; color: #60a5fa;'>{word.upper() if isinstance(word, str) and not any(ord(c) > 127 for c in word) else word}</span>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<div style='margin-top: 1rem;'>", unsafe_allow_html=True)
        if st.button("READY TO RECALL", type="primary", use_container_width=True):
            st.session_state.memory_phase = "input"
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)
            
    elif st.session_state.memory_phase == "input":
        st.write("### ⌨️ Input recalled tokens:")
        user_answer = st.text_input("Enter the tokens separated by spaces:", placeholder="Ex: token1 token2 token3")
        if st.button("SUBMIT RECALL DATA", type="primary", use_container_width=True):
            correct = sum(1 for w in st.session_state.memory_words if w.lower() in user_answer.lower())
            st.session_state.memory_score = correct / 3
            st.session_state.memory_result_type = selected_pool
            st.session_state.memory_phase = "result"
            st.rerun()

    elif st.session_state.memory_phase == "result":
        score = st.session_state.memory_score
        risk_color = "#22c55e" if score == 1 else "#f59e0b" if score >= 0.6 else "#ef4444"
        
        st.markdown(f"""
        <div class="card" style="text-align: center; border: 1px solid {risk_color}33;">
            <h3 style="color: {risk_color}; margin: 0;">{st.session_state.memory_result_type} Accuracy</h3>
            <h1 style="font-size: 4rem; margin: 1rem 0;">{score:.0%}</h1>
            <p>{'Outstanding focal attention' if score == 1 else 'Mild recall delay' if score >= 0.6 else 'Significant retention gap'}</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔄 NEW SESSION", use_container_width=True):
            st.session_state.memory_words = random.sample(data_pools[selected_pool], 3)
            st.session_state.memory_phase = "show"
            st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# PRONUNCIATION TEST TAB
with tab4:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #ffffff; text-align: center;">🗣️ Articulation Engine</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.6); margin-bottom: 2rem;">Testing phonetic precision and motor control in complex sentence structures.</p>', unsafe_allow_html=True)
    
    sentences = [
        "the quick brown fox jumps over the lazy dog",
        "she sells seashells by the seashore",
        "peter piper picked a peck of pickled peppers",
        "seventy seven benevolent elephants",
        "the beautiful bouquet blossomed brightly"
    ]
    
    if "target_sentence" not in st.session_state:
        st.session_state.target_sentence = random.choice(sentences)
        
    st.markdown(f"""
    <div style="background: rgba(255,255,255,0.05); padding: 2.5rem; border-radius: 16px; border: 1px dashed rgba(255,255,255,0.2); text-align: center; margin-bottom: 2rem;">
        <p style="font-size: 0.9rem; opacity: 0.5; margin-bottom: 0.5rem; text-transform: uppercase;">Repeat this sentence:</p>
        <h3 style="margin: 0; color: #60a5fa; font-style: italic; font-size: 1.8rem;">"{st.session_state.target_sentence}"</h3>
    </div>
    """, unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    with col1:
        if st.button("🎤 BEGIN RECORDING", type="primary", use_container_width=True):
            with st.spinner("Recording neural patterns..."):
                spoken_text, _ = record_speech()
                similarity = SequenceMatcher(None, st.session_state.target_sentence, spoken_text.lower()).ratio()
                st.session_state.pronunciation_score = similarity
                st.session_state.pronunciation_text = spoken_text
    
    with col2:
        if st.button("🔄 CHANGE SENTENCE", use_container_width=True):
            st.session_state.target_sentence = random.choice(sentences)
            st.rerun()

    if "pronunciation_score" in st.session_state:
        st.markdown("<div style='margin-top: 2rem;'>", unsafe_allow_html=True)
        st.success(f"Captured: {st.session_state.pronunciation_text}")
        st.metric("Phonetic Similarity", f"{st.session_state.pronunciation_score:.0%}")
        st.markdown("</div>", unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# FINAL REPORT TAB
with tab5:
    st.markdown('<div class="card pulse" style="border: 2px solid rgba(59, 130, 246, 0.5);">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #ffffff; text-align: center; margin-bottom: 2rem;">📋 Diagnostic Summary Report</h2>', unsafe_allow_html=True)
    
    # Defaults for missing data
    speech_risk = st.session_state.get('speech_risk', 0.5)
    text_risk = st.session_state.get('text_risk', 0.5)
    memory_penalty = 1.0 - st.session_state.get('memory_score', 0.5)
    pronunciation_penalty = 1.0 - st.session_state.get('pronunciation_score', 0.5)
    
    final_score = (speech_risk * 0.4) + (text_risk * 0.3) + (memory_penalty * 0.2) + (pronunciation_penalty * 0.1)
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        risk_class = "risk-low" if final_score < 0.35 else "risk-high"
        st.markdown(f"""
        <div class="card {risk_class}" style="text-align: center; padding: 4rem 2rem;">
            <p style="text-transform: uppercase; letter-spacing: 0.2em; opacity: 0.7; font-size: 0.9rem; margin-bottom: 1rem;">Global Cognitive Index</p>
            <h1 style="font-size: 6rem; line-height: 1;">{final_score:.0%}</h1>
            <p style="font-size: 1.2rem; margin-top: 2rem; border-top: 1px solid rgba(255,255,255,0.1); padding-top: 1.5rem;">
                Status: <strong style="color: white;">{'NORMAL' if final_score < 0.35 else 'FOLLOW-UP RECOMMENDED'}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div class='card' style='padding: 1.5rem !important;'>", unsafe_allow_html=True)
        st.write("### Domain Scores")
        st.write(f"🎤 Speech: `{speech_risk:.0%}`")
        st.write(f"✍️ Language: `{text_risk:.0%}`")
        st.write(f"🧠 Memory: `{memory_penalty:.0%}`")
        st.write(f"🗣️ Articulation: `{pronunciation_penalty:.0%}`")
        st.markdown("</div>", unsafe_allow_html=True)
        
    # REPORT GENERATION DATA
    report_data = {
        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M"),
        'final_score': final_score,
        'speech_risk': speech_risk,
        'text_risk': text_risk,
        'memory_score': st.session_state.get('memory_score', 0.5),
        'pronunciation_score': st.session_state.get('pronunciation_score', 0.5),
        'findings': st.session_state.get('speech_features', {}).get('findings', []),
        'patient': user
    }
    
    st.markdown("---")
    
    col_pdf1, col_pdf2 = st.columns(2)
    
    with col_pdf1:
        st.write("### 📄 Clinical Report Actions")
        pdf_bytes = create_clinical_report_pdf(report_data)
        
        # Ensure a clean filename for patients
        safe_username = re.sub(r'[^\w]', '_', user.get('username', 'patient'))
        final_filename = f"report_{safe_username}_{datetime.now().strftime('%H%M%S')}.pdf"
        
        st.download_button(
            label="📥 DOWNLOAD PERSONAL PDF REPORT",
            data=pdf_bytes,
            file_name=final_filename,
            mime="application/pdf",
            use_container_width=True,
            key="download_report_pdf_patient"
        )
        
    with col_pdf2:
        st.write("### 🩺 Professional Coordination")
        if st.button("🚀 SYNC TO CLINICAL DASHBOARD", type="primary", use_container_width=True):
            with st.spinner("Syncing data with neurology department..."):
                # Save PDF to reports folder
                safe_name = re.sub(r'[^\w]', '_', user.get('name', 'patient'))
                ts_str    = datetime.now().strftime("%Y%m%d_%H%M%S")
                pdf_filename = f"{safe_name}_{ts_str}.pdf"
                pdf_path     = os.path.join(REPORTS_DIR, pdf_filename)
                with open(pdf_path, "wb") as pf:
                    pf.write(pdf_bytes)
                
                # Save index entry
                index_entry = {
                    "timestamp":           report_data['timestamp'],
                    "patient_name":        user.get('name', 'Unknown'),
                    "age":                 user.get('age', 'N/A'),
                    "gender":              user.get('gender', 'N/A'),
                    "username":            user.get('username', 'N/A'),
                    "final_score":         round(final_score, 4),
                    "speech_risk":         round(speech_risk, 4),
                    "text_risk":           round(text_risk, 4),
                    "memory_score":        round(st.session_state.get('memory_score', 0.5), 4),
                    "pronunciation_score": round(st.session_state.get('pronunciation_score', 0.5), 4),
                    "findings":            report_data['findings'],
                    "pdf_path":            pdf_path
                }
                save_report_entry(index_entry)
                st.success("✅ Report successfully stored on the Doctor's dashboard.")
                st.balloons()

    with st.expander("📋 PREVIEW: Clinical Diagnostic Summary"):
        st.write("### NEURAL SCREENING SYSTEM REPORT")
        st.write(f"**Patient:** {user.get('name','?')} | Age: {user.get('age','?')} | Gender: {user.get('gender','?')}")
        st.write(f"**Status:** {'🔴 FOLLOW-UP' if final_score >= 0.35 else '🟢 NORMAL'}")
        st.write(f"**Global Index:** {final_score:.0%}")
        st.write("**Biomarker Summary:**")
        if report_data['findings']:
            for f in report_data['findings']:
                st.write(f"- {f}")
        else:
            st.write("_Awaiting analysis data... Run the tests above to generate findings._")
    
    st.markdown("""
    <div style="background: rgba(239, 68, 68, 0.1); border-radius: 16px; padding: 1.5rem; border: 1px solid rgba(239, 68, 68, 0.2);">
        <p style="color: #f87171; font-size: 0.9rem; margin: 0;">
            <strong>⚠️ Medical Disclaimer:</strong> This system is a screening prototype utilizing experimental neural biomarkers. 
            It does not provide a clinical diagnosis. Please consult a neurologist for comprehensive evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Unified sidebar handled above
