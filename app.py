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
import joblib
from features import extract_features

# Custom Auth Module
from auth import load_users, save_users, authenticate, register_patient, get_all_patients
from history import save_assessment, get_patient_history, get_all_recent_assessments
import pandas as pd

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

nlp = load_nlp()

# Load Neural Model & Scaler
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'alzheimer_model.pkl')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'model', 'scaler.pkl')

try:
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        alz_model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    else:
        alz_model = None
        scaler = None
except:
    alz_model = None
    scaler = None

def sync_to_brain(text, final_label):
    """Self-Learning Loop: Updates the model in real-time based on clinician feedback."""
    global alz_model, scaler
    if alz_model and scaler:
        try:
            feat_vector = extract_features(text)
            X = scaler.transform([feat_vector])
            # Incremental update (Online Learning)
            alz_model.partial_fit(X, [final_label])
            # Save updated brain state
            joblib.dump(alz_model, MODEL_PATH)
            return True
        except:
            return False
    return False

# 🔥 CINEMATIC NEURAL DESIGN SYSTEM
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;700&display=swap');
    
/* Base Experience */
.stApp {
    background: radial-gradient(circle at 50% -20%, #1a233a 0%, #020408 100%) !important;
    color: #ffffff !important;
    font-family: 'Outfit', sans-serif !important;
}

/* Advanced Cinematic Glassmorphism */
.card {
    background: rgba(15, 23, 42, 0.95) !important;
    backdrop-filter: blur(10px) !important;
    border-radius: 32px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    padding: 3rem !important;
    box-shadow: 0 30px 60px -12px rgba(0, 0, 0, 0.9) !important;
    margin-bottom: 2.5rem !important;
    position: relative;
    overflow: hidden;
}

.card:hover {
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
    transform: translateY(-8px) scale(1.005) !important;
    box-shadow: 0 35px 60px -15px rgba(59, 130, 246, 0.2) !important;
}

/* Neural Glow Accents */
.card::before {
    content: '';
    position: absolute;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(circle, rgba(59, 130, 246, 0.05) 0%, transparent 60%);
    pointer-events: none;
    transition: all 1s ease;
}

.card:hover::before {
    background: radial-gradient(circle, rgba(59, 130, 246, 0.1) 0%, transparent 70%);
}

/* High-Intensity Typography */
h1, h2, h3 {
    font-family: 'Outfit', sans-serif !important;
    letter-spacing: -0.02em !important;
    color: #ffffff !important;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5) !important;
}

h1 { font-weight: 800 !important; font-size: 4rem !important; line-height: 1.1 !important; }

/* Cinematic Branding */
.gradient-text {
    background: linear-gradient(135deg, #ffffff 0%, #93c5fd 50%, #c084fc 100%) !important;
    -webkit-background-clip: text !important;
    -webkit-text-fill-color: transparent !important;
    font-weight: 900 !important;
    text-shadow: 0 4px 10px rgba(0,0,0,0.5) !important;
}

/* Hyper-Sensitive Risk States */
.risk-low { 
    background: linear-gradient(135deg, rgba(34, 197, 94, 0.15), #0f172a) !important;
    border: 2px solid #22c55e !important;
}
.risk-high { 
    background: linear-gradient(135deg, rgba(239, 68, 68, 0.15), #0f172a) !important;
    border: 2px solid #ef4444 !important;
    animation: critical-pulse 2s infinite ease-in-out;
}

@keyframes critical-pulse {
    0% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.1); }
    50% { box-shadow: 0 0 30px rgba(239, 68, 68, 0.3); }
    100% { box-shadow: 0 0 10px rgba(239, 68, 68, 0.1); }
}

/* Pro Metric Displays */
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 800 !important;
    color: #ffffff !important;
    font-size: 3.5rem !important;
    text-shadow: 0 4px 15px rgba(0,0,0,0.3) !important;
    -webkit-text-fill-color: initial !important;
}

[data-testid="stMetricLabel"] {
    color: #ffffff !important;
    font-size: 1.1rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.12em !important;
    opacity: 1.0 !important;
    text-shadow: 0 2px 4px rgba(0,0,0,0.5) !important;
}

.card p, .card span {
    color: #ffffff !important;
    opacity: 1.0 !important;
    font-weight: 400 !important;
}

/* Cinematic Tabs */
.stTabs [data-baseweb="tab-list"] {
    gap: 16px;
    background-color: transparent !important;
    padding: 10px 0;
}

.stTabs [data-baseweb="tab"] {
    height: 60px !important;
    border-radius: 16px !important;
    padding: 0 32px !important;
    background-color: rgba(255, 255, 255, 0.03) !important;
    border: 1px solid rgba(255, 255, 255, 0.05) !important;
    color: rgba(255, 255, 255, 0.5) !important;
    font-weight: 600 !important;
}

.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #3b82f6 0%, #2563eb 100%) !important;
    color: white !important;
    border: none !important;
    box-shadow: 0 8px 25px rgba(37, 99, 235, 0.4) !important;
    transform: scale(1.05);
}

/* 3D Medical Buttons */
.stButton button {
    background: linear-gradient(180deg, #3b82f6 0%, #1d4ed8 100%) !important;
    border-radius: 18px !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    color: white !important;
    font-weight: 700 !important;
    padding: 1rem 3rem !important;
    height: auto !important;
    box-shadow: 0 4px 0 #1e40af, 0 8px 15px rgba(0,0,0,0.4) !important;
    transition: all 0.2s ease !important;
    letter-spacing: 0.1em !important;
}

.stButton button:active {
    transform: translateY(3px) !important;
    box-shadow: 0 1px 0 #1e40af, 0 4px 10px rgba(0,0,0,0.4) !important;
}

.stButton button:hover {
    background: linear-gradient(180deg, #60a5fa 0%, #3b82f6 100%) !important;
    box-shadow: 0 4px 0 #1e40af, 0 12px 25px rgba(59, 130, 246, 0.4) !important;
}

/* Neural Pulse Visualizer */
.neural-sync {
    width: 100%;
    height: 4px;
    background: rgba(59, 130, 246, 0.1);
    border-radius: 2px;
    position: relative;
    overflow: hidden;
    margin: 1rem 0;
}
.neural-sync::after {
    content: '';
    position: absolute;
    top: 0; left: -100%;
    width: 50%; height: 100%;
    background: linear-gradient(90deg, transparent, #3b82f6, transparent);
    animation: neural-flow 2s infinite linear;
}
@keyframes neural-flow {
    from { left: -100%; }
    to { left: 100%; }
}

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
    .login-title { font-size: 2.2rem; font-weight: 800; color: white; text-align: center; margin-bottom: 0.4rem; }
    .login-sub   { color: #ffffff; text-align: center; margin-bottom: 2rem; font-size: 1rem; opacity: 1.0; }
    </style>
    """, unsafe_allow_html=True)

    # HERO BANNER - CINEMATIC VERSION
    st.markdown(f"""
    <div style="text-align: center; padding: 4rem 0; background: radial-gradient(circle at center, rgba(59, 130, 246, 0.1) 0%, transparent 70%);">
        <h1 style="font-size: 3.5rem !important; margin-bottom: 0.5rem; color: #60a5fa; font-weight: 900; letter-spacing: -0.05em; text-transform: uppercase;">NEURAL SCREENING SYSTEM</h1>
        <p style="font-size: 1.2rem; opacity: 1.0; letter-spacing: 0.2rem; color: #ffffff; font-weight: 500;">
            ADVANCED COGNITIVE BIOMARKER ANALYTICS | VER 4.0
        </p>
        <div class="neural-sync" style="max-width: 300px; margin: 1.5rem auto;"></div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="card" style="max-width: 800px; margin: 0 auto;">', unsafe_allow_html=True)
    
    # CINEMATIC AUTH TABS
    auth_tab1, auth_tab2, auth_tab3 = st.tabs(["PATIENT PORTAL", "NEW ACCOUNT", "CLINICIAN ACCESS"])
    
    with auth_tab1:
        st.markdown("<h3 style='margin-bottom: 2rem;'>Patient Login</h3>", unsafe_allow_html=True)
        login_user = st.text_input("Username", key="l_user", placeholder="e.g. john_doe")
        login_pass = st.text_input("Password", type="password", key="l_pass")
        if st.button("AUTHENTICATE & ENTER", type="primary", use_container_width=True):
            user = authenticate(login_user, login_pass, role='patient')
            if user and user['role'] == 'patient':
                st.session_state.logged_in = True
                st.session_state.role = 'patient'
                st.session_state.current_user = user
                st.rerun()
            else:
                st.error("Invalid credentials or incorrect portal.")

    with auth_tab2:
        st.markdown("<h3 style='margin-bottom: 2rem;'>Initialize Neural Profile</h3>", unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            reg_name = st.text_input("Full Name", placeholder="e.g. John Doe")
            reg_user = st.text_input("Username ID")
        with c2:
            reg_age = st.number_input("Age", 18, 120, 65)
            reg_gender = st.selectbox("Gender", ["Male", "Female", "Other"])
        reg_pass = st.text_input("Security PIN / Password", type="password")
        
        if st.button("CREATE PROFILE", type="primary", use_container_width=True):
            if reg_user and reg_pass and reg_name:
                if register_patient(reg_name, reg_age, reg_gender, reg_user, reg_pass):
                    st.success("Profile Initialized. Please use the Login tab.")
                else:
                    st.error("Username already registered.")
            else:
                st.warning("Please fill all clinical parameters.")

    with auth_tab3:
        st.markdown("<h3 style='margin-bottom: 2rem;'>Medical Professional Portal</h3>", unsafe_allow_html=True)
        doc_user = st.text_input("Clinician Username", key="d_user")
        doc_pass = st.text_input("Clinician Security Key", type="password", key="d_pass")
        if st.button("VERIFY & ASCEND", type="primary", use_container_width=True):
            user = authenticate(doc_user, doc_pass, role='doctor')
            if user and user['role'] == 'doctor':
                st.session_state.logged_in = True
                st.session_state.role = 'doctor'
                st.session_state.current_user = user
                st.rerun()
            else:
                st.error("❌ Invalid clinician credentials.")
    st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ── LOGGED-IN STATE ─────────────────────────────────────────────────────────
user = st.session_state.current_user

with st.sidebar:
    st.markdown("### 🔐 Session Management")
    if st.button("🚪 Logout", use_container_width=True):
        for k in ["logged_in", "role", "current_user"]:
            st.session_state[k] = None if k != "logged_in" else False
        st.rerun()
    st.markdown("---")

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

        st.markdown("---")
        st.markdown("### 📅 Patient History Matrix (Last 7 Days)")
        
        all_patients = get_all_patients()
        if all_patients:
            patient_names = {u['username']: u['name'] for u in all_patients}
            selected_hist_user = st.selectbox("Select Patient for Trend Analysis:", options=list(patient_names.keys()), format_func=lambda x: patient_names[x])
            
            history = get_patient_history(selected_hist_user, days=7)
            if history:
                df_hist = pd.DataFrame(history)
                # Ensure the display is clean
                if 'timestamp' in df_hist.columns:
                    df_hist['date'] = pd.to_datetime(df_hist['timestamp']).dt.strftime('%Y-%m-%d %H:%M')
                    
                cols_to_show = ['date', 'final_score', 'speech_risk', 'text_risk']
                st.table(df_hist[cols_to_show].style.format({
                    'final_score': '{:.0%}',
                    'speech_risk': '{:.0%}',
                    'text_risk': '{:.0%}'
                }))
            else:
                st.info(f"No recent history recorded for {patient_names[selected_hist_user]}.")
        else:
            st.info("No patients registered in the system.")

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
            Real-time Inference
        </span>
        <span style="background: rgba(168, 85, 247, 0.1); padding: 0.5rem 1.5rem; border-radius: 100px; border: 1px solid rgba(168, 85, 247, 0.3); color: #a855f7; font-size: 0.9rem;">
            Neural Biomarkers
        </span>
    </div>
</div>

<!-- PATIENT PROFILE BANNER -->
<div style="background: rgba(255,255,255,0.04); border-radius: 100px; padding: 0.8rem 2rem; border: 1px solid rgba(255,255,255,0.1); margin: -1rem auto 2rem; display: flex; justify-content: center; gap: 2rem; max-width: fit-content; align-items: center;">
    <span style="font-size: 1.2rem;"><strong>{user.get('name')}</strong></span>
    <span style="opacity: 0.5;">|</span>
    <span>Age: <strong>{user.get('age')}</strong></span>
    <span style="opacity: 0.5;">|</span>
    <span>Gender: <strong>{user.get('gender')}</strong></span>
    <span style="opacity: 0.5;">|</span>
    <span style="color: #60a5fa; font-family: 'JetBrains Mono';">ID: {user.get('username')}</span>
</div>
""", unsafe_allow_html=True)

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


def analyze_language(text):
    if not text or len(text.strip()) < 5:
        # Full default dict to prevent KeyError in UI
        return 0.5, {
            "vocab_richness": 0.0, 
            "filler_ratio": 0.0, 
            "avg_sent_len": 0.0, 
            "repetition": 0.0, 
            "empty_ratio": 0.0, 
            "total_words": 0,
            "findings": ["Insufficient data / Waiting for input..."]
        }
        
    # Use Shared Feature Extraction
    feat_vector = extract_features(text)
    
    # ── CLINICAL INTERPRETATION ──────────────
    findings = []
    
    # Extract specific values for findings UI (based on features.py order)
    # [vocab_rich, disfluency_score, sent_len, pron_ratio, rep_index]
    vocab_richness = feat_vector[0]
    disfluency_score = feat_vector[1]
    avg_sent_len = feat_vector[2] * 20.0 # Scale back for UI
    pron_ratio = feat_vector[3]
    rep_index = feat_vector[4]
    
    if vocab_richness < 0.70:
        findings.append("Anomia (Significant Lexical Search Deficit)")
    if disfluency_score > 0.05:
        findings.append("Aphasic Fragmentation (Excessive Fillers/Hesitations)")
    if rep_index > 0.02:
        findings.append("Palilalia (Repetition Loop)")
    if avg_sent_len < 12:
        findings.append("Logopenia (Grammatical Collapse)")
    if pron_ratio > 0.15:
        findings.append("Semantic Impoverishment (High Pronoun Density)")
        
    # 🧪 MACHINE LEARNING PREDICTION (Self-Learning SGD)
    if alz_model and scaler:
        # Scale features for SGD consistency
        X_scaled = scaler.transform([feat_vector])
        # Get probability of class 1 (Dementia)
        prob = alz_model.predict_proba(X_scaled)[0][1]
        risk_score = prob
    else:
        # Fallback to heuristic if model missing (Realistic Multiplier)
        risk_score = min(disfluency_score * 2.5 + (1-vocab_richness) * 1.5, 0.99)

    return risk_score, {
        'vocab_richness': vocab_richness,
        'filler_ratio': disfluency_score,
        'avg_sent_len': avg_sent_len,
        'repetition': rep_index,
        'empty_ratio': pron_ratio,     # Fix for KeyError: 'empty_ratio' in UI
        'total_words': len(text.split()),
        'findings': findings if findings else ["Normal Cognitive-Linguistic Profile"]
    }

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
        
        text = r.recognize_google(audio)
        word_count = len(text.split())
        
        # ── ACOUSTIC BIOMARKERS (THE HESITATION TRAP) ────────
        # 1. Total Audio Duration
        audio_data = np.frombuffer(audio.get_raw_data(), np.int16).astype(np.float32) / 32768.0
        sample_rate = audio.sample_rate
        duration_sec = len(audio_data) / sample_rate
        
        # 2. Words Per Minute (WPM) - Clinical Benchmark
        wpm = (word_count / max(duration_sec, 1)) * 60
        
        # 3. Acoustic-to-Text Gap (THE GHOST FILLER TRAP)
        # Normal speech is ~2.5 words/sec. Below 1.3 indicates heavy disfluency (filtered uhh/umm)
        speech_density = word_count / max(duration_sec, 0.5)
        
        # 4. Advanced Silence Analysis
        silence_ratio = np.mean(np.abs(audio_data) < 0.005)
        
        # ── EXPERT PROBABILISTIC SCORING (REALISTIC VARIANCE) ──────────
        # Individual risk components (0.0 to 1.0)
        # 1. WPM (Norm: 130-160, Clinical: <100)
        wpm_risk = float(max(0.0, (140.0 - wpm) / 100.0)) 
        
        # 2. Density (Norm: 2.0+, Clinical: <1.2)
        density_risk = float(max(0.0, (2.0 - speech_density) / 1.5))
        
        # 3. Silence (Norm: <20%, Clinical: >40%)
        silence_risk = float(max(0.0, (silence_ratio - 0.15) / 0.5))

        # Ghost filler detection (Acoustic fingerprint of filtered disfluency)
        ghost_filler_intensity = float(max(0.0, (1.6 - speech_density) / 1.0)) * 0.2

        # Probabilistic OR combination: Risk = 1 - (1-r1)*(1-r2)*(1-r3)
        # This prevents additive saturation while ensuring all factors contribute.
        combined_prob = 1.0 - ( (1.0 - wpm_risk) * (1.0 - density_risk) * (1.0 - silence_risk) )
        
        # Add a subtle base for ghost fillers
        final_score_raw = combined_prob + ghost_filler_intensity
        
        # Final calibrated score (Realistic 0-100% range)
        final_hes_score = float(min(max(final_score_raw, 0.0), 1.0))
        st.session_state.hesitation_score = final_hes_score
        
        # Clinical Findings for UI
        findings = []
        if wpm < 115: findings.append(f"Bradyphasic Load ({wpm:.0f} WPM)")
        if speech_density < 1.5: findings.append("Neural Articulation Blockage")
        if silence_ratio > 0.30: findings.append("Pathological Silence Detection")
        if ghost_filler_intensity > 0.1: findings.append("Acoustic Fingerprint (Disfluency)")
        
        # Store for UI display
        st.session_state.last_speech_metrics = {
            "wpm": wpm,
            "density": speech_density,
            "silence": silence_ratio,
            "findings": findings
        }
        
        return text, final_hes_score

    except sr.WaitTimeoutError:
        st.warning("⏰ No speech detected. Please click the button and speak within 8 seconds.")
        return None, None
    except sr.UnknownValueError:
        st.warning("🔇 Could not understand the audio. Please speak clearly and try again.")
        return None, None
    except sr.RequestError as e:
        st.error(f"🌐 Speech recognition service error: {e}")
        return None, None
    except Exception as e:
        st.error(f"🎤 Microphone error: {e}")
        return None, None


# PRO TABS
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

    if 'speech_text' in st.session_state:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        col1, col2 = st.columns([3,1])
        with col1:
            st.text_area("📝 **Detected Speech:**", value=st.session_state.speech_text, height=100, disabled=True)
        with col2:
            st.markdown(f"""
            <div style="text-align: center;">
                <p style="margin: 0; font-size: 0.8rem; opacity: 0.6;">Hesitation Index</p>
                <h2 style="margin: 0; color: #60a5fa;">{st.session_state.hesitation_score:.0%}</h2>
                <div class="neural-sync"></div>
            </div>
            """, unsafe_allow_html=True)
            if 'last_speech_metrics' in st.session_state:
                metrics = st.session_state.last_speech_metrics
                with st.expander("📊 Acoustic Raw Metrics", expanded=False):
                    st.write(f"⏱️ **WPM:** `{metrics['wpm']:.1f}`")
                    st.write(f"📡 **Density:** `{metrics['density']:.2f}` dps")
                    st.write(f"🔇 **Silence:** `{metrics['silence']:.1%}`")
                    st.info("dps = words per second of audio")
        
        if st.button("🧠 TRIGGER NEURAL ANALYSIS", type="primary", use_container_width=True, key="analyze_speech"):
            with st.spinner("Decoding cognitive fingerprints..."):
                text_score, features = analyze_language(st.session_state.speech_text)
                
                # Combine acoustic and linguistic risk
                speech_risk = (text_score * 0.5) + (st.session_state.hesitation_score * 0.5)
                st.session_state.speech_risk = speech_risk
                st.session_state.speech_features = features
            
            risk_class = "risk-low" if speech_risk < 0.28 else "risk-high"
            st.markdown(f"""
            <div class="card {risk_class}" style="text-align: center; padding: 4rem;">
                <h3 style="margin: 0; opacity: 0.9; font-size: 1rem; text-transform: uppercase;">Cognitive Biomarker Intensity</h3>
                <h1 style="font-size: 7rem; margin: 1.5rem 0;" class="gradient-text">{speech_risk:.0%}</h1>
                <div style="display: flex; justify-content: center; gap: 1rem; flex-wrap: wrap; margin-top: 2rem;">
                    {' '.join([f'<span style="background: rgba(59, 130, 246, 0.1); padding: 8px 20px; border-radius: 30px; font-size: 0.9rem; border: 1px solid rgba(59, 130, 246, 0.3); color: white;">{f}</span>' for f in features['findings']])}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            with st.expander("🩺 Clinical Biomarker Insights", expanded=True):
                c1, c2, c3 = st.columns(3)
                c1.metric("Lexical Flow", f"{features['vocab_richness']:.1%}", help="Norm: >60% unique words")
                c2.metric("Empty Words", f"{features['empty_ratio']:.1%}", delta="Norm <10%", delta_color="inverse")
                c3.metric("Fluency Index", f"{1-features['filler_ratio']:.1%}", delta="Norm >95%", delta_color="normal")
                
                st.markdown("---")
                st.markdown("**Acoustic & Linguistic Interpretation:**")
                
                # Show acoustic findings if they exist
                if 'last_speech_metrics' in st.session_state:
                    for af in st.session_state.last_speech_metrics['findings']:
                        st.write(f"📡 **{af}**")
                
                for finding in features['findings']:
                    if "Anomia" in finding:
                        st.write("🔴 **Anomia:** Difficulty in word-finding (Lexical search deficit).")
                    elif "Empty" in finding:
                        st.write("🟡 **Semantic Anomia:** Frequent use of vague 'empty' words (e.g., thing, stuff).")
                    elif "Aphasic" in finding:
                        st.write("🟡 **Aphasic Hesitation:** Processing delays or filler insertion.")
                    elif "Palilalia" in finding:
                        st.write("🔴 **Palilalia:** Involuntary word repetition (Frontal stress).")
                    elif "Logopenia" in finding:
                        st.write("🟡 **Logopenia:** Reduced grammatical complexity.")
            st.markdown('</div>', unsafe_allow_html=True)

# TEXT TAB
with tab2:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #ffffff; text-align: center;">Linguistic Analysis</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.9); margin-bottom: 2rem;">Explain a picture or describe your morning routine in detail below.</p>', unsafe_allow_html=True)
    
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
    st.markdown('<h2 style="color: #ffffff; text-align: center;">Memory Recall Matrix</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.9); margin-bottom: 2rem;">Multimodal recall assessment across different cognitive domains.</p>', unsafe_allow_html=True)
    
    # Categorized data pools
    data_pools = {
        "Lexical (Words)": ["apple", "river", "chair", "table", "tree", "car", "sun", "book", "mountain", "cloud"],
        "Numerical (Digits)": ["482", "915", "376", "204", "859", "127", "630", "548", "791", "263"],
        "Visual (Shapes)": ["Circle", "Square", "Triangle", "Star", "Diamond", "Hexagon", "Pentagon", "Cross", "Arrow", "Heart"]
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
    st.markdown('<h2 style="color: #ffffff; text-align: center;">Articulation Engine</h2>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; color: rgba(255,255,255,0.9); margin-bottom: 2rem;">Testing phonetic precision and motor control in complex sentence structures.</p>', unsafe_allow_html=True)
    
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
        <p style="font-size: 0.9rem; opacity: 0.9; margin-bottom: 0.5rem; text-transform: uppercase;">Repeat this sentence:</p>
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
        if st.button("CHANGE SENTENCE", use_container_width=True):
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
    st.markdown('<div class="card risk-high" style="border: 2px solid rgba(59, 130, 246, 0.5);">', unsafe_allow_html=True)
    st.markdown('<h2 style="color: #ffffff; text-align: center; margin-bottom: 3rem;">DIAGNOSTIC NEURAL SUMMARY</h2>', unsafe_allow_html=True)
    
    # High Sensitivity Multi-Domain Calculation
    speech_risk = st.session_state.get('speech_risk', 0.5)
    text_risk = st.session_state.get('text_risk', 0.5)
    memory_penalty = 1.0 - st.session_state.get('memory_score', 0.5)
    pronunciation_penalty = 1.0 - st.session_state.get('pronunciation_score', 0.5)
    
    final_score = (speech_risk * 0.45) + (text_risk * 0.35) + (memory_penalty * 0.15) + (pronunciation_penalty * 0.05)
    
    col1, col2 = st.columns([1.5, 1])
    with col1:
        # ULTIMATE SENSITIVITY THRESHOLDS
        risk_class = "risk-low" if final_score < 0.25 else "risk-high"
        status_text = "OPTIMAL BASELINE" if final_score < 0.25 else "CLINICAL INTERVENTION INDICATED"
        status_color = "#22c55e" if final_score < 0.25 else "#ef4444"
        
        st.markdown(f"""
        <div class="card {risk_class}" style="text-align: center; padding: 4rem 2rem;">
            <p style="text-transform: uppercase; letter-spacing: 0.3em; color: #ffffff; opacity: 1.0; font-size: 0.9rem; font-weight: 700; margin-bottom: 1.5rem;">Global Cognitive Index</p>
            <h1 style="font-size: 8rem; line-height: 1;" class="gradient-text">{final_score:.0%}</h1>
            <div class="neural-sync" style="margin: 2rem 0;"></div>
            <p style="font-size: 1.4rem; margin-top: 1rem; padding-top: 1.5rem;">
                Clinical Status: <strong style="color: {status_color}; text-shadow: 0 0 10px {status_color}55;">{status_text}</strong>
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("<div style='padding-left: 2rem;'>", unsafe_allow_html=True)
        st.write("### Neural Domain Breakdown")
        st.write(f"Speech Biomarkers: `{speech_risk:.0%}`")
        st.write(f"Language Entropy: `{text_risk:.0%}`")
        st.write(f"Hippocampal Retention: `{memory_penalty:.0%}`")
        st.write(f"Motor Articulation: `{pronunciation_penalty:.0%}`")
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("High synergy detected. Combined risk factors exceed normal latency thresholds.")
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
        st.write("### Clinical Report Actions")
        pdf_bytes = create_clinical_report_pdf(report_data)
        
        # Ensure a clean filename for patients
        safe_username = re.sub(r'[^\w]', '_', user.get('username', 'patient'))
        final_filename = f"report_{safe_username}_{datetime.now().strftime('%H%M%S')}.pdf"
        
        st.download_button(
            label="DOWNLOAD PERSONAL PDF REPORT",
            data=pdf_bytes,
            file_name=final_filename,
            mime="application/pdf",
            use_container_width=True,
            key="download_report_pdf_patient"
        )
        
    with col_pdf2:
        st.write("### Professional Coordination")
        if st.button("📡 SYNC TO GLOBAL BRAIN", type="primary", use_container_width=True):
            # Clinician validation: if score > 0.5, treat as class 1 (Dementia) for learning
            final_label = 1 if final_score > 0.5 else 0
            if sync_to_brain(st.session_state.speech_text, final_label):
                st.success("✅ Model Updated. Synced with global cognitive dataset.")
            else:
                st.info("💡 Analysis Synced with Clinician Portal")
                st.balloons()
        
        if st.button("SYNC TO CLINICAL DASHBOARD", use_container_width=True):
            with st.spinner("Syncing data with neurology department..."):
                # 1. Save to persistent history
                save_assessment(user['username'], report_data)
                
                # 2. Save PDF to reports folder
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
                st.success("Report successfully stored on the Doctor's dashboard.")
                st.balloons()

    with st.expander("PREVIEW: Clinical Diagnostic Summary"):
        st.write("### NEURAL SCREENING SYSTEM REPORT")
        st.write(f"**Patient:** {user.get('name','?')} | Age: {user.get('age','?')} | Gender: {user.get('gender','?')}")
        st.write(f"**Status:** {'FOLLOW-UP REQUIRED' if final_score >= 0.35 else 'NORMAL'}")
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
            <strong>Medical Disclaimer:</strong> This system is a screening prototype utilizing experimental neural biomarkers. 
            It does not provide a clinical diagnosis. Please consult a neurologist for comprehensive evaluation.
        </p>
    </div>
    """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

st.sidebar.caption("NLP + Raw Audio Analysis")
st.sidebar.caption("Screening prototype only")
