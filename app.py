import os
import datetime

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import requests
from streamlit_lottie import st_lottie

# ======================================================
# BASIC PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="AI Ticket Classifier",
    page_icon="💻",
    layout="wide",
)

# ------------------------------------------------------
# CONSTANTS / FILES
# ------------------------------------------------------
USERS_FILE = "users.csv"
PREDICTION_LOG = "prediction_log.csv"

# ======================================================
# USER MANAGEMENT (CSV-BASED AUTH)
# ======================================================

def ensure_users_file():
    """Create users.csv with one default admin user if it does not exist."""
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame(
            [
                ["admin", "sahil123", "admin"],
            ],
            columns=["username", "password", "role"],
        )
        df.to_csv(USERS_FILE, index=False)


def load_users() -> pd.DataFrame:
    ensure_users_file()
    return pd.read_csv(USERS_FILE)


def save_users(df: pd.DataFrame) -> None:
    df.to_csv(USERS_FILE, index=False)


def authenticate(username: str, password: str):
    """Return role if username/password match, else None."""
    users = load_users()
    row = users[(users["username"] == username) & (users["password"] == password)]
    if len(row) == 1:
        return row.iloc[0]["role"]
    return None


def change_password(username: str, old_pass: str, new_pass: str) -> bool:
    """Change password for a user if old password is correct."""
    users = load_users()
    match = users[(users["username"] == username) & (users["password"] == old_pass)]
    if match.empty:
        return False

    users.loc[users["username"] == username, "password"] = new_pass
    save_users(users)
    return True


# ======================================================
# SESSION AUTH STATE
# ======================================================
if "auth" not in st.session_state:
    st.session_state["auth"] = {
        "logged_in": False,
        "username": None,
        "role": None,
    }


# ======================================================
# LOGIN SCREEN
# ======================================================
def login_screen():
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>🔐 Login to AI Ticket Classifier</h1>",
        unsafe_allow_html=True,
    )
    st.write("")

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login", use_container_width=True):
            role = authenticate(username, password)
            if role:
                st.session_state["auth"] = {
                    "logged_in": True,
                    "username": username,
                    "role": role,
                }
                st.success(f"Welcome, **{username}** ({role})!")
                st.rerun()
            else:
                st.error("❌ Invalid username or password")


# If not logged in, show login and stop
if not st.session_state["auth"]["logged_in"]:
    login_screen()
    st.stop()

username = st.session_state["auth"]["username"]
role = st.session_state["auth"]["role"]
can_bulk = role in ("admin", "engineer")
is_admin = role == "admin"

# ======================================================
# STYLING
# ======================================================
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

# ======================================================
# TITLE + HERO
# ======================================================
st.markdown('<div class="big-title">🧠 AI Ticket Classifier</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Smart classification of IT support tickets for '
    'Active Directory, Network, Hardware, Email, Firewall, MDM, Printer & Security issues.</div>',
    unsafe_allow_html=True,
)


# ======================================================
# ANIMATION (simple dev typing animation)
# ======================================================
def load_lottie(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except Exception:
        return None


lottie_url = "https://assets2.lottiefiles.com/packages/lf20_kyu7xb1v.json"
lottie_ai = load_lottie(lottie_url)

if lottie_ai:
    st_lottie = st_lottie  # just to satisfy linters
    st_lottie(lottie_ai, height=220, key="hero_animation")

# ======================================================
# LOAD MODEL / VECTORIZER
# ======================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()

# ======================================================
# UTILITY FUNCTIONS
# ======================================================
def simple_summary(text: str, max_words: int = 18) -> str:
    words = text.split()
    return text if len(words) <= max_words else " ".join(words[:max_words]) + "..."


def suggest_fix(category: str) -> str:
    suggestions = {
        "Active Directory Issue": "Check account lockout, reset password, verify group membership in AD.",
        "Hardware Issue": "Ask user to reboot, check cables, run hardware diagnostics or swap peripherals.",
        "Network Issue": "Check WiFi/LAN connection, ping gateway, restart router/switch, verify VPN.",
        "Email Issue": "Recreate Outlook profile, check mailbox size, verify SMTP/Exchange connectivity.",
        "Firewall Issue": "Check firewall rule for required ports, NAT policies, and recent rule changes.",
        "MDM Issue": "Re-enroll device, sync MDM policies, verify compliance and device status.",
        "Printer Issue": "Check printer network status, reinstall drivers, clear queue, verify IP.",
        "Security Issue": "Run AV scan, isolate device if needed, reset credentials, review logs.",
    }
    return suggestions.get(category, "No predefined suggestion available for this category yet.")


def compute_severity(ticket: str, category: str) -> str:
    text = ticket.lower()

    if any(k in text for k in ["all users", "server down", "data breach", "cannot access", "production down"]):
        return "Critical"

    if any(k in text for k in ["can't login", "cant login", "urgent", "very slow", "vpn disconnecting"]):
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
        "user": username,
    }
    df = pd.DataFrame([row])
    df.to_csv(
        PREDICTION_LOG,
        mode="a",
        header=not os.path.exists(PREDICTION_LOG),
        index=False,
    )


def generate_user_email(ticket, category, severity):
    return (
        f"Subject: Your Support Ticket Update ({category})\n\n"
        f"Hi,\n\n"
        f"Your issue has been classified as **{category}** with severity **{severity}**.\n"
        f"Ticket: \"{ticket}\"\n\n"
        f"Our IT team is working on it and will update you soon.\n"
    )


def generate_internal_note(ticket, category, severity, confidence):
    return (
        f"Issue: {category}\n"
        f"Severity: {severity}\n"
        f"Model confidence: {confidence * 100:.2f}%\n\n"
        f"Ticket text:\n{ticket}\n\n"
        f"Suggested next step:\n- {suggest_fix(category)}"
    )


def jaccard_similarity(a: str, b: str) -> float:
    set_a, set_b = set(a.lower().split()), set(b.lower().split())
    return len(set_a & set_b) / len(set_a | set_b) if set_a and set_b else 0.0


def find_similar_tickets(current_ticket: str, top_n: int = 3):
    if not os.path.exists(PREDICTION_LOG):
        return None
    hist_df = pd.read_csv(PREDICTION_LOG)
    if "ticket" not in hist_df.columns:
        return None

    hist_df["similarity"] = hist_df["ticket"].astype(str).apply(
        lambda t: jaccard_similarity(current_ticket, t)
    )
    hist_df = hist_df.sort_values("similarity", ascending=False)
    hist_df = hist_df[hist_df["similarity"] > 0]
    return hist_df.head(top_n) if not hist_df.empty else None


# ======================================================
# SIDEBAR MENU
# ======================================================
with st.sidebar:
    st.header("⚙️ Menu")
    st.write(f"Logged in as: **{username} ({role})**")

    if st.button("Logout", use_container_width=True):
        st.session_state["auth"] = {
            "logged_in": False,
            "username": None,
            "role": None,
        }
        st.rerun()

    with st.expander("🔑 Change my password"):
        old_p = st.text_input("Old password", type="password", key="old_pw")
        new_p = st.text_input("New password", type="password", key="new_pw")
        if st.button("Update password", key="btn_change_pw", use_container_width=True):
            if not old_p or not new_p:
                st.warning("Please fill both fields.")
            else:
                if change_password(username, old_p, new_p):
                    st.success("Password updated successfully.")
                else:
                    st.error("Old password is incorrect.")

    st.markdown("---")
    st.header("ℹ️ About this app")
    st.write(
        "This tool uses a Machine Learning model (Random Forest + TF-IDF) "
        "to automatically classify IT support tickets."
    )

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
        key="example",
    )

# ======================================================
# MAIN TABS
# ======================================================
tabs = ["📝 Single Ticket", "📂 Bulk CSV"]
if is_admin:
    tabs.append("👥 User Management")
if os.path.exists(PREDICTION_LOG):
    tabs.append("📊 History")

tab_objects = st.tabs(tabs)

tab_idx = 0

# ------------------------------------------------------
# 1) Single Ticket
# ------------------------------------------------------
with tab_objects[tab_idx]:
    st.subheader("Classify a single IT support ticket")

    ticket_text = st.text_area("Ticket Description", height=120)

    # Use example if chosen and box empty
    if example_choice != "(None)" and not ticket_text.strip():
        ticket_text = example_choice
        st.info(f"Using example: _{ticket_text}_")

    if st.button("Predict Category"):
        if not ticket_text.strip():
            st.warning("Please enter a ticket description.")
        else:
            vec = vectorizer.transform([ticket_text])
            prediction = model.predict(vec)[0]

            # Confidence
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
                f'<div class="prediction-box">Predicted Category: '
                f'<b>{prediction}</b></div>',
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

            # Log + similar tickets
            log_prediction(ticket_text, prediction, top_conf, severity)

            st.markdown("### 🔎 Similar past tickets")
            similar = find_similar_tickets(ticket_text, 3)
            if similar is not None:
                st.dataframe(similar)
            else:
                st.info("No similar tickets found yet.")

tab_idx += 1

# ------------------------------------------------------
# 2) Bulk CSV (admin/engineer)
# ------------------------------------------------------
with tab_objects[tab_idx]:
    st.subheader("Bulk classify tickets from CSV")

    if not can_bulk:
        st.warning("Only **admin** and **engineer** roles can use bulk CSV.")
    else:
        uploaded = st.file_uploader("Upload CSV with a 'ticket' column", type=["csv"])
        if uploaded:
            df = pd.read_csv(uploaded)
            if "ticket" not in df.columns:
                st.error("CSV must contain a 'ticket' column.")
            else:
                vec = vectorizer.transform(df["ticket"].astype(str))
                preds = model.predict(vec)
                df["prediction"] = preds
                st.dataframe(df)

                csv_bytes = df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "Download results CSV",
                    csv_bytes,
                    file_name="ticket_predictions.csv",
                    mime="text/csv",
                )

tab_idx += 1

# ------------------------------------------------------
# 3) User Management (admin only)
# ------------------------------------------------------
if is_admin:
    with tab_objects[tab_idx]:
        st.subheader("User Management (Admin only)")

        users = load_users()
        st.dataframe(users, use_container_width=True)

        st.markdown("### ➕ Add new user")
        new_user = st.text_input("New username")
        new_pass = st.text_input("New password", type="password")
        new_role = st.selectbox("Role", ["admin", "engineer", "user"])

        if st.button("Create user"):
            if not new_user or not new_pass:
                st.warning("Username and password cannot be empty.")
            elif new_user in users["username"].values:
                st.error("User already exists.")
            else:
                new_row = pd.DataFrame(
                    {"username": [new_user], "password": [new_pass], "role": [new_role]}
                )
                save_users(pd.concat([users, new_row], ignore_index=True))
                st.success("User created.")
                st.rerun()

        st.markdown("### 🗑 Delete user")
        del_user = st.selectbox("Select user to delete", users["username"])
        if st.button("Delete user"):
            if del_user == "admin":
                st.error("You cannot delete the main admin account.")
            else:
                users = users[users["username"] != del_user]
                save_users(users)
                st.success("User deleted.")
                st.rerun()

    tab_idx += 1

# ------------------------------------------------------
# 4) History tab (if log exists)
# ------------------------------------------------------
if os.path.exists(PREDICTION_LOG):
    with tab_objects[tab_idx]:
        st.subheader("Prediction history")

        hist = pd.read_csv(PREDICTION_LOG)
        st.dataframe(hist, use_container_width=True)
