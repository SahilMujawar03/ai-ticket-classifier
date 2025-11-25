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

# ------------------ GLOBAL STYLE ------------------
st.markdown(
    """
    <style>
    /* Full-app animated blue/purple gradient background */
    .stApp {
        background: linear-gradient(135deg, #1d4ed8, #3b82f6, #6366f1);
        background-size: 280% 280%;
        animation: gradientBG 22s ease infinite;
        color: #e5e7eb;
    }
    @keyframes gradientBG {
      0% {background-position: 0% 50%;}
      50% {background-position: 100% 50%;}
      100% {background-position: 0% 50%;}
    }

    /* Smooth fade-in */
    .main, .block-container {
        animation: fadeIn 0.7s ease-out;
    }
    @keyframes fadeIn {
        from {opacity: 0; transform: translateY(4px);}
        to {opacity: 1; transform: translateY(0);}
    }

    /* Center column + remove white boxes */
    .block-container {
        max-width: 1100px;
        padding-top: 1.2rem;
        padding-bottom: 3rem;
        background-color: transparent !important;
    }

    /* Title + subtitle */
    .big-title {
        font-size: 42px;
        font-weight: 900;
        text-align: center;
        margin-bottom: -4px;
        color: #f9fafb;
    }
    .sub-header {
        font-size: 18px;
        text-align: center;
        margin-bottom: 22px;
        color: #e0ebff;
    }

    /* Buttons with hover */
    .stButton>button {
        background: linear-gradient(120deg, #22c55e, #16a34a);
        border: none;
        color: white;
        padding: 0.38rem 1.3rem;
        border-radius: 999px;
        font-weight: 600;
        transition: all 0.18s ease-in-out;
        box-shadow: 0 8px 16px rgba(22, 163, 74, 0.4);
    }
    .stButton>button:hover {
        transform: translateY(-1px) scale(1.02);
        box-shadow: 0 14px 24px rgba(22, 163, 74, 0.6);
        cursor: pointer;
    }

    /* Tabs styling (hover + active) */
    .stTabs [role="tab"] {
        border-radius: 999px;
        padding: 0.4rem 1.4rem;
        margin-right: 0.35rem;
        font-weight: 600;
        color: #e5e7eb;
        background-color: rgba(15, 23, 42, 0.35);
        transition: all 0.18s ease-in-out;
        border: 1px solid rgba(191, 219, 254, 0.6);
    }
    .stTabs [role="tab"]:hover {
        transform: translateY(-1px);
        border-color: #f97316;
        box-shadow: 0 6px 16px rgba(248, 166, 62, 0.55);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(120deg, #2563eb, #0ea5e9);
        color: white;
        box-shadow: 0 10px 24px rgba(15, 23, 42, 0.7);
        border-color: transparent;
    }

    /* Sidebar: transparent over same gradient (no separate dark panel) */
    section[data-testid="stSidebar"] {
        background-color: transparent;
        backdrop-filter: blur(4px);
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] label {
        color: #f9fafb;
    }

    /* Dataframe text darker so it’s readable on white cells */
    .dataframe td, .dataframe th {
        color: #020617 !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ------------------ LOTTIE ANIMATION HELPERS ------------------
def load_lottie_url(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None

def safe_lottie(animation, height: int, key: str):
    """Render a Lottie animation if available; do nothing if it fails."""
    try:
        if animation:
            st_lottie(animation, height=height, key=key)
    except Exception:
        pass

# Developer typing animation (Option B)
dev_anim = load_lottie_url(
    "https://lottie.host/ccafdcc2-bb3c-4cab-9ee5-4858f7dbda1a/0g31xkRS0o.json"
)

# ------------------ TITLE ------------------
st.markdown("<div class='big-title'>🤖 AI Ticket Classifier</div>", unsafe_allow_html=True)
st.markdown(
    "<div class='sub-header'>Smart classification of IT support tickets – Active Directory, Network, Hardware, Email, Firewall, MDM, Printer & Security.</div>",
    unsafe_allow_html=True,
)

# ------------------ SIDEBAR ------------------
st.sidebar.title("ℹ️ About this app")
st.sidebar.write(
    "This tool uses a Machine Learning model (Random Forest + TF-IDF) "
    "trained on real-world IT support tickets to automatically categorize new requests."
)

st.sidebar.markdown("### 🔐 Admin login")
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")
admin_input = st.sidebar.text_input("Enter admin password", type="password")

is_admin = admin_input == ADMIN_PASSWORD
if admin_input and not is_admin:
    st.sidebar.error("Incorrect password.")
elif admin_input and is_admin:
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

# ------------------ MAIN TABS ------------------
tab1, tab2, tab3 = st.tabs(["📝 Single Ticket", "📂 Bulk CSV (Admin)", "📊 History (Admin)"])

# ---------- TAB 1: SINGLE TICKET ----------
with tab1:
    st.subheader("Classify a single IT support ticket")

    ticket_text = st.text_area("Ticket Description", height=150)

    predict_clicked = st.button("Predict Category")

    if predict_clicked:
        if ticket_text.strip():
            with st.spinner("Analyzing ticket with AI..."):
                X = vectorizer.transform([ticket_text])
                prediction = model.predict(X)[0]
                prob = float(model.predict_proba(X).max())

            st.success(f"Predicted Category: **{prediction}**")
            st.info(f"Confidence: **{prob*100:.2f}%**")
            st.warning(f"Suggested Fix: {suggest_fix(prediction)}")

            log_prediction(ticket_text, prediction, prob)
        else:
            st.error("Please enter a ticket description.")

    # 👇 Developer typing animation directly under the Predict button / results
    safe_lottie(dev_anim, height=230, key="dev_under_predict")

# ---------- TAB 2: BULK CSV (ADMIN) ----------
with tab2:
    if not is_admin:
        st.error("Admin access required. Enter password in the sidebar.")
    else:
        st.subheader("Bulk classify tickets from CSV")
        st.caption("Upload a CSV file with a column named **'ticket'** containing ticket descriptions.")

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

# ---------- TAB 3: HISTORY (ADMIN) ----------
with tab3:
    if not is_admin:
        st.error("Admin access required. Enter password in the sidebar.")
    else:
        st.subheader("Prediction history")

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
