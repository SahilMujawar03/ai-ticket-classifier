import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime
import requests
from streamlit_lottie import st_lottie

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="AI Ticket Classifier",
    page_icon="🤖",
    layout="wide",
)

# ------------------ LOTTIE ANIMATION LOADER ------------------
from streamlit_lottie import st_lottie
import requests

def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

# Working animations
hero_anim = load_lottie_url("https://lottie.host/6a5447a6-b010-4f6d-9d9a-a66f61e63029/ibGWpgtmG9.json")
sidebar_anim = load_lottie_url("https://lottie.host/49e2ca61-a653-4a92-9d85-1b3b0fe16db2/Ly6rCasVnF.json")
loading_anim = load_lottie_url("https://lottie.host/30cb2f87-5694-48ce-a2d1-f2f6c9b4e60d/n1w3yF7p9o.json")
# ------------------ STYLE ------------------
st.markdown(
    """
    <style>
    .big-title {
        font-size: 42px;
        font-weight: 900;
        text-align: center;
        margin-bottom: -20px;
    }
    .sub-header {
        font-size: 20px;
        text-align: center;
        margin-bottom: 30px;
        color: #555;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ TITLE + HERO ANIMATION ------------------
st.markdown("<div class='big-title'>🤖 AI Ticket Classifier</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Smart classification of IT support tickets – Active Directory, Network, Hardware, Email, Firewall, MDM, Printer & Security.</div>",
    unsafe_allow_html=True,
)

# Hero animation centered
st_lottie(hero_anim, height=230, key="hero")

# ------------------ SIDEBAR ------------------
st.sidebar.title("ℹ️ About this app")
st.sidebar.write(
    "This tool uses a Machine Learning model (Random Forest + TF-IDF) trained on real-world IT tickets."
)

# Sidebar animation
st_lottie(sidebar_anim, height=200, key="sidebar")

st.sidebar.markdown("### 🔐 Admin login")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")
admin_input = st.sidebar.text_input("Enter admin password", type="password")

is_admin = admin_input == ADMIN_PASSWORD

# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

# ------------------ HELPERS ------------------
def suggest_fix(category: str) -> str:
    fixes = {
        "Active Directory Issue": "Reset password, unlock AD account, verify group membership.",
        "Hardware Issue": "Check cables, reboot device, reseat hardware, run diagnostics.",
        "Network Issue": "Check WiFi/LAN connection, ping gateway, restart router.",
        "Email Issue": "Verify Outlook profile, mailbox size, SMTP connection.",
        "Firewall Issue": "Verify firewall rule, NAT settings, required ports.",
        "MDM Issue": "Re-enroll device, sync policies, verify compliance.",
        "Printer Issue": "Restart printer spooler, reinstall drivers, check network.",
        "Security Issue": "Run AV scan, isolate device, reset credentials.",
    }
    return fixes.get(category, "No predefined suggestion available.")

def log_prediction(ticket: str, prediction: str, confidence: float):
    row = {
        "timestamp": datetime.datetime.now(),
        "ticket": ticket,
        "prediction": prediction,
        "confidence": confidence,
    }
    df = pd.DataFrame([row])
    df.to_csv("prediction_log.csv", mode="a", header=False, index=False)

# ------------------ MAIN UI ------------------
tab1, tab2, tab3 = st.tabs(["📝 Single Ticket", "📂 Bulk CSV (Admin)", "📊 History (Admin)"])

# --- TAB 1: SINGLE TICKET ---
with tab1:
    st.header("Classify a single IT support ticket")

    ticket_text = st.text_area("Ticket Description", height=140)

    if st.button("Predict Category"):
        if ticket_text.strip():

            # Show loading animation
            with st.spinner("Processing..."):
                st_lottie(loading_anim, height=140, key="loading")

            X = vectorizer.transform([ticket_text])
            prediction = model.predict(X)[0]
            prob = model.predict_proba(X).max()

            st.success(f"Predicted Category: **{prediction}**")
            st.info(f"Confidence: **{prob*100:.2f}%**")
            st.warning(f"Suggested Fix: {suggest_fix(prediction)}")

            log_prediction(ticket_text, prediction, prob)

        else:
            st.error("Please enter a ticket description.")

# --- TAB 2: BULK CSV (ADMIN) ---
with tab2:
    if not is_admin:
        st.error("Admin access required.")
    else:
        st.header("Upload CSV for bulk classification")
        csv_file = st.file_uploader("Upload CSV", type=["csv"])

        if csv_file:
            df = pd.read_csv(csv_file)
            df["prediction"] = model.predict(vectorizer.transform(df["ticket"]))
            st.write(df)
            st.success("Bulk prediction completed!")

# --- TAB 3: HISTORY (ADMIN) ---
with tab3:
    if not is_admin:
        st.error("Admin access required.")
    else:
        st.header("Prediction History")
        try:
            log_df = pd.read_csv("prediction_log.csv", names=["timestamp", "ticket", "prediction", "confidence"])
            st.write(log_df.tail(50))
        except:
            st.info("No history found.")

