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

# ------------------ GLOBAL STYLE (BACKGROUND, HOVER, TABS, TRANSITION) ------------------
st.markdown(
    """
    <style>
    /* Animated gradient background */
    .stApp {
        background: linear-gradient(120deg, #0f172a, #1e293b, #0f766e, #1d4ed8);
        background-size: 400% 400%;
        animation: gradientBG 20s ease infinite;
        color: #e5e7eb;
    }

    @keyframes gradientBG {
      0% {background-position: 0% 50%;}
      50% {background-position: 100% 50%;}
      100% {background-position: 0% 50%;}
    }

    /* Fade-in page transition */
    .main, .block-container {
        animation: fadeIn 0.8s ease-in-out;
    }

    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(4px);}
        to {opacity: 1; transform: translateY(0);}
    }

    /* Big title + subtitle */
    .big-title {
        font-size: 42px;
        font-weight: 900;
        text-align: center;
        margin-bottom: -8px;
        color: #e5e7eb;
    }
    .sub-header {
        font-size: 19px;
        text-align: center;
        margin-bottom: 24px;
        color: #cbd5f5;
    }

    /* Card-like look for main content */
    .block-container {
        max-width: 1100px;
        padding-top: 1.5rem;
        padding-bottom: 4rem;
    }

    /* Button hover effects */
    .stButton>button {
        background: linear-gradient(120deg, #22c55e, #16a34a);
        border: none;
        color: white;
        padding: 0.35rem 1.2rem;
        border-radius: 999px;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
        box-shadow: 0 8px 16px rgba(22, 163, 74, 0.35);
    }
    .stButton>button:hover {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 14px 24px rgba(22, 163, 74, 0.5);
        cursor: pointer;
    }

    /* Tabs styling (animated underline + hover) */
    .stTabs [role="tab"] {
        border-radius: 999px;
        padding: 0.4rem 1.4rem;
        margin-right: 0.35rem;
        font-weight: 600;
        color: #e5e7eb;
        background-color: rgba(15, 23, 42, 0.6);
        transition: all 0.2s ease-in-out;
        border: 1px solid rgba(148, 163, 184, 0.4);
    }

    .stTabs [role="tab"]:hover {
        transform: translateY(-1px);
        border-color: #38bdf8;
        box-shadow: 0 6px 16px rgba(56, 189, 248, 0.35);
    }

    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #2563eb, #0ea5e9);
        color: white;
        box-shadow: 0 10px 24px rgba(37, 99, 235, 0.45);
        border-color: transparent;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #020617, #0b1120);
        border-right: 1px solid rgba(148, 163, 184, 0.4);
    }
    section[data-testid="stSidebar"] h1, section[data-testid="stSidebar"] h2, section[data-testid="stSidebar"] h3 {
        color: #e5e7eb;
    }

    /* Improve dataframe contrast */
    .dataframe th, .dataframe td {
        color: #020617 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ LOTTIE ANIMATION LOADER ------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def safe_lottie(animation, height: int, key: str):
    """Render a Lottie animation without crashing app if it fails."""
    try:
        if animation:
            st_lottie(animation, height=height, key=key)
    except Exception:
        pass

# Working animations
hero_anim = load_lottie_url("https://lottie.host/6a5447a6-b010-4f6d-9d9a-a66f61e63029/ibGWpgtmG9.json")
sidebar_anim = load_lottie_url("https://lottie.host/49e2ca61-a653-4a92-9d85-1b3b0fe16db2/Ly6rCasVnF.json")
loading_anim = load_lottie_url("https://lottie.host/30cb2f87-5694-48ce-a2d1-f2f6c9b4e60d/n1w3yF7p9o.json")

# ------------------ TITLE + HERO ANIMATION ------------------
st.markdown("<div class='big-title'>🤖 AI Ticket Classifier</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Smart classification of IT support tickets – Active Directory, Network, Hardware, Email, Firewall, MDM, Printer & Security.</div>",
    unsafe_allow_html=True,
)

safe_lottie(hero_anim, height=230, key="hero")

# ------------------ SIDEBAR ------------------
st.sidebar.title("ℹ️ About this app")
st.sidebar.write(
    "This tool uses a Machine Learning model (Random Forest + TF-IDF)\n"
    "trained on real-world IT support tickets to automatically\n"
    "categorize new requests."
)

safe_lottie(sidebar_anim, height=180, key="sidebar")

st.sidebar.markdown("### 🔐 Admin login")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")
admin_input = st.sidebar.text_input("Enter admin password", type="password", placeholder="Enter admin password")

is_admin = admin_input == ADMIN_PASSWORD
if admin_input and not is_admin:
    st.sidebar.error("Incorrect password.")
elif is_admin and admin_input:
    st.sidebar.success("Admin mode enabled ✅")

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

# ------------------ MAIN UI (TABS) ------------------
tab1, tab2, tab3 = st.tabs(["📝 Single Ticket", "📂 Bulk CSV (Admin)", "📊 History (Admin)"])

# --- TAB 1: SINGLE TICKET ---
with tab1:
    st.subheader("Classify a single IT support ticket")

    ticket_text = st.text_area("Ticket Description", height=160)

    if st.button("Predict Category"):
        if ticket_text.strip():
            # Animated "page transition" feeling: spinner + loading animation
            with st.spinner("Analyzing ticket with AI..."):
                safe_lottie(loading_anim, height=120, key="loading")

                X = vectorizer.transform([ticket_text])
                prediction = model.predict(X)[0]
                prob = float(model.predict_proba(X).max())

            st.success(f"Predicted Category: **{prediction}**")
            st.info(f"Confidence: **{prob*100:.2f}%**")
            st.warning(f"Suggested Fix: {suggest_fix(prediction)}")

            log_prediction(ticket_text, prediction, prob)

        else:
            st.error("Please enter a ticket description.")

# --- TAB 2: BULK CSV (ADMIN) ---
with tab2:
    if not is_admin:
        st.error("Admin access required. Enter password in the sidebar.")
    else:
        st.subheader("Upload CSV for bulk classification")
        st.caption("CSV must contain a column named **'ticket'**.")

        csv_file = st.file_uploader("Upload CSV", type=["csv"])

        if csv_file:
            try:
                df = pd.read_csv(csv_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                df = None

            if df is not None:
                if "ticket" not in df.columns:
                    st.error("CSV must contain a 'ticket' column.")
                else:
                    with st.spinner("Running bulk classification..."):
                        X_bulk = vectorizer.transform(df["ticket"])
                        df["prediction"] = model.predict(X_bulk)
                        probs = model.predict_proba(X_bulk)
                        df["confidence"] = probs.max(axis=1)

                    st.success("Bulk prediction completed!")
                    st.dataframe(df.head(50))

                    csv_out = df.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "Download predictions as CSV",
                        data=csv_out,
                        file_name="ticket_predictions.csv",
                        mime="text/csv",
                    )

# --- TAB 3: HISTORY (ADMIN) ---
with tab3:
    if not is_admin:
        st.error("Admin access required. Enter password in the sidebar.")
    else:
        st.subheader("Prediction History")

        try:
            log_df = pd.read_csv(
                "prediction_log.csv",
                names=["timestamp", "ticket", "prediction", "confidence"],
            )
            st.caption(f"Total logged predictions: **{len(log_df)}**")
            st.dataframe(log_df.tail(50))

            st.markdown("### 📊 Predictions per category")
            counts = log_df["prediction"].value_counts()
            st.bar_chart(counts)
        except FileNotFoundError:
            st.info("No history found yet. Make some predictions first.")
