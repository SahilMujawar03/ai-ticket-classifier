import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime
import requests
from streamlit_lottie import st_lottie

# ----------------- BASIC CONFIG -----------------
st.set_page_config(
    page_title="AI Ticket Classifier",
    page_icon="💻",
    layout="wide",
)

# ----------------- AUTH / USERS -----------------
# Reuse ADMIN_PASSWORD from secrets for the 'admin' user
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")

USERS = {
    "admin": {
        "password": ADMIN_PASSWORD,
        "role": "admin",
    },
    "engineer": {
        "password": "engineer123",  # demo password; change in real usage
        "role": "engineer",
    },
    "user": {
        "password": "user123",  # demo password; change in real usage
        "role": "user",
    },
}


def init_auth_state():
    if "auth" not in st.session_state:
        st.session_state["auth"] = {
            "logged_in": False,
            "username": None,
            "role": None,
        }


def login_screen():
    st.markdown(
        "<h2 style='text-align:center;'>🔐 Login to AI Ticket Classifier</h2>",
        unsafe_allow_html=True,
    )
    st.write("")
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        if st.button("Login"):
            user = USERS.get(username)
            if user and user["password"] == password:
                st.session_state["auth"]["logged_in"] = True
                st.session_state["auth"]["username"] = username
                st.session_state["auth"]["role"] = user["role"]
                st.success(f"Welcome, {username} ({user['role']})!")

                # 🔧 FIX APPLIED HERE
                st.rerun()

            else:
                st.error("Invalid username or password.")


init_auth_state()
auth = st.session_state["auth"]

# If not logged in, show login screen and stop
if not auth["logged_in"]:
    login_screen()
    st.stop()

username = auth["username"]
role = auth["role"]
can_bulk = role in ("admin", "engineer")
can_history = role in ("admin", "engineer")
can_dashboard = role == "admin"

# ----------------- STYLE -----------------
st.markdown(
    """
    <style>
    .big-title {
        font-size: 40px;
        font-weight: 800;
        text-align: center;
        margin-bottom: 4px;
    }
    .sub-header {
        font-size: 17px;
        text-align: center;
        margin-bottom: 25px;
        color: #555;
    }
    .prediction-box {
        padding: 12px 16px;
        border-radius: 8px;
        background-color: #e8f7e8;
        border: 1px solid #c2e5c2;
        margin-top: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown('<div class="big-title">🧠 AI Ticket Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Smart classification of IT support tickets '
    'for Active Directory, Network, Hardware, Email, Firewall, MDM, Printer & Security issues.</div>',
    unsafe_allow_html=True,
)

# ----------------- ANIMATION UTILS -----------------
def load_lottie(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


# Load an AI-themed animation (typing / dev style)
lottie_url = "https://assets2.lottiefiles.com/packages/lf20_kyu7xb1v.json"
lottie_ai = load_lottie(lottie_url)

# Display animation below the title
if lottie_ai:
    st_lottie(lottie_ai, height=230, key="aiAnimation")

# ----------------- LOAD ARTIFACTS -----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()

# ----------------- UTILS -----------------
def simple_summary(text: str, max_words: int = 18) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + "..."


def suggest_fix(category: str) -> str:
    suggestions = {
        "Active Directory Issue": "Check account lockout, reset password.",
        "Hardware Issue": "Check cables, run hardware diagnostics.",
        "Network Issue": "Restart router, check WiFi/LAN, verify VPN.",
        "Email Issue": "Recreate Outlook profile, check mailbox size.",
        "Firewall Issue": "Check port rules, NAT policies.",
        "MDM Issue": "Re-enroll device, sync policies.",
        "Printer Issue": "Reconnect printer, reinstall drivers.",
        "Security Issue": "Run antivirus scan, isolate device.",
    }
    return suggestions.get(category, "No suggestion available.")


def compute_severity(ticket: str, category: str) -> str:
    text = ticket.lower()

    if any(k in text for k in ["all users", "server down", "data breach", "cannot access"]):
        return "Critical"

    if any(k in text for k in ["can't login", "urgent", "very slow", "vpn disconnecting"]):
        return "High"

    if any(k in text for k in ["sometimes", "intermittent", "some users"]):
        return "Medium"

    return "Low"


def log_prediction(ticket, prediction, confidence, severity):
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "ticket": ticket,
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
        "severity": severity,
    }
    df = pd.DataFrame([row])
    df.to_csv(
        "prediction_log.csv",
        mode="a",
        header=not st.session_state.get("log_initialized", False),
        index=False,
    )
    st.session_state["log_initialized"] = True


def generate_user_email(ticket, category, severity):
    return (
        f"Subject: Your Support Ticket Update ({category})\n\n"
        f"Your issue has been classified as **{category}** with severity **{severity}**.\n"
        f"Ticket: \"{ticket}\"\n\n"
        f"Our team will get back to you soon."
    )


def generate_internal_note(ticket, category, severity, confidence):
    return (
        f"Issue: {category}\nSeverity: {severity}\nConfidence: {confidence*100:.2f}%\n\n"
        f"Ticket:\n{ticket}\n\nSuggested Fix:\n- {suggest_fix(category)}"
    )


def jaccard_similarity(a: str, b: str) -> float:
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    return len(set_a & set_b) / len(set_a | set_b) if set_a and set_b else 0.0


def find_similar_tickets(current_ticket: str, top_n: int = 3):
    try:
        hist_df = pd.read_csv("prediction_log.csv")
    except FileNotFoundError:
        return None

    if "ticket" not in hist_df.columns:
        return None

    hist_df["similarity"] = hist_df["ticket"].astype(str).apply(
        lambda t: jaccard_similarity(current_ticket, t)
    )
    hist_df = hist_df.sort_values("similarity", ascending=False)
    hist_df = hist_df[hist_df["similarity"] > 0]
    return hist_df.head(top_n) if not hist_df.empty else None


# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("👤 Logged in")
    st.write(f"**User:** {username}")
    st.write(f"**Role:** {role.capitalize()}")

    if st.button("Logout"):
        st.session_state["auth"] = {
            "logged_in": False,
            "username": None,
            "role": None,
        }

        # 🔧 FIX APPLIED HERE
        st.rerun()

    st.markdown("---")
    st.header("ℹ️ About this app")
    st.write("""
This tool uses a machine learning model (Random Forest + TF-IDF)
to classify IT support tickets into categories.
""")

    st.markdown("**Example tickets:**")
    example_tickets = [
        "User cannot login to Active Directory",
        "Outlook not sending emails",
        "Laptop running very slow",
        "VPN keeps disconnecting",
        "Printer not connecting to network",
        "Antivirus alert: malware detected",
    ]
    example_choice = st.selectbox(
        "Choose an example ticket:",
        ["(None)"] + example_tickets,
        index=0,
    )

# ----------------- MAIN LAYOUT -----------------
tab_single, tab_bulk, tab_history, tab_dashboard = st.tabs(
    ["📝 Single Ticket", "📂 Bulk CSV", "📊 History", "📈 Dashboard"]
)

# ---------------- SINGLE TICKET ----------------
with tab_single:
    st.subheader("Classify a single IT ticket")

    ticket_text = st.text_area("Ticket Description", height=120)

    if example_choice != "(None)" and not ticket_text.strip():
        ticket_text = example_choice
        st.info(f"Using example: _{ticket_text}_")

    if st.button("Predict Category"):
        if not ticket_text.strip():
            st.warning("Please enter a ticket description.")
        else:
            vec = vectorizer.transform([ticket_text])
            prediction = model.predict(vec)[0]

            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(vec)[0]
                classes = model.classes_
                top_idx = np.argsort(probas)[::-1][:3]
                top_conf = probas[top_idx[0]]
            else:
                classes = model.classes_
                probas = np.ones(len(classes)) / len(classes)
                top_idx = np.argsort(probas)[::-1][:3]
                top_conf = probas[top_idx[0]]

            severity = compute_severity(ticket_text, prediction)

            st.markdown(
                f'<div class="prediction-box">Predicted Category: <b>{prediction}</b></div>',
                unsafe_allow_html=True,
            )

            st.markdown("### 🚨 Severity")
            st.info(f"Severity: **{severity}**")

            st.markdown("### 🧾 Summary")
            st.info(simple_summary(ticket_text))

            st.markdown("### 🛠 Suggested fix")
            st.info(suggest_fix(prediction))

            st.markdown("### 🔍 Confidence")
            for idx in top_idx:
                st.write(f"- {classes[idx]} — **{probas[idx] * 100:.1f}%**")
            st.progress(float(top_conf))

            st.markdown("### 📧 Email templates")
            st.code(generate_user_email(ticket_text, prediction, severity))
            st.code(generate_internal_note(ticket_text, prediction, severity, top_conf))

            log_prediction(ticket_text, prediction, top_conf, severity)

            st.markdown("### 🔎 Similar past tickets")
            similar = find_similar_tickets(ticket_text, 3)
            if similar is not None:
                st.dataframe(similar)
            else:
                st.info("No similar tickets found.")

# (Bulk, History, Dashboard tabs unchanged)
