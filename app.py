import os
import datetime
import time
import smtplib
from email.message import EmailMessage
import sqlite3
from contextlib import contextmanager

import numpy as np
import pandas as pd
import streamlit as st
import joblib
import bcrypt
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

# ======================================================
# DATABASE SETUP (SQLite)
# ======================================================
DB_FILE = "app.db"


@contextmanager
def get_db():
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def init_db():
    """Create tables if they don't exist."""
    with get_db() as db:
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS users (
                username TEXT PRIMARY KEY,
                password TEXT NOT NULL,
                role     TEXT NOT NULL
            )
            """
        )
        db.execute(
            """
            CREATE TABLE IF NOT EXISTS tickets (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp TEXT NOT NULL,
                ticket TEXT NOT NULL,
                prediction TEXT,
                confidence REAL,
                severity TEXT,
                user TEXT
            )
            """
        )


# ======================================================
# CONSTANTS / ENV
# ======================================================
SESSION_TIMEOUT_MINUTES = 30
MAX_LOGIN_ATTEMPTS = 5
LOCKOUT_SECONDS = 300

ENABLE_EMAIL_NOTIFICATIONS = True
ENABLE_SLACK_NOTIFICATIONS = True

EMAIL_SMTP_HOST = os.getenv("SMTP_HOST", "")
EMAIL_SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
EMAIL_SMTP_USER = os.getenv("SMTP_USER", "")
EMAIL_SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
EMAIL_FROM = os.getenv("EMAIL_FROM", EMAIL_SMTP_USER or "")
EMAIL_TO_IT = os.getenv("EMAIL_TO_IT", "")

SLACK_WEBHOOK_URL = os.getenv("SLACK_WEBHOOK_URL", "")

# ======================================================
# PASSWORD / USER HELPERS
# ======================================================
def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def is_hashed_password(pwd: str) -> bool:
    return isinstance(pwd, str) and pwd.startswith("$2")


def verify_password(plain: str, hashed: str) -> bool:
    if not isinstance(hashed, str):
        return False
    try:
        if is_hashed_password(hashed):
            return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))
        # legacy plain-text fallback
        return plain == hashed
    except Exception:
        return False


def ensure_admin_user():
    """Ensure at least one admin exists in the DB."""
    with get_db() as db:
        cur = db.execute("SELECT COUNT(*) AS c FROM users")
        count = cur.fetchone()["c"]
        if count == 0:
            # default admin
            admin_user = "admin"
            admin_pass = "sahil123"
            db.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (admin_user, hash_password(admin_pass), "admin"),
            )


def load_users() -> pd.DataFrame:
    with get_db() as db:
        df = pd.read_sql_query("SELECT username, password, role FROM users", db)
    return df


def authenticate(username: str, password: str):
    with get_db() as db:
        cur = db.execute(
            "SELECT username, password, role FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
    if row and verify_password(password, row["password"]):
        return row["role"]
    return None


def change_password(username: str, old_pass: str, new_pass: str) -> bool:
    with get_db() as db:
        cur = db.execute(
            "SELECT password FROM users WHERE username = ?",
            (username,),
        )
        row = cur.fetchone()
        if not row or not verify_password(old_pass, row["password"]):
            return False
        new_hash = hash_password(new_pass)
        db.execute(
            "UPDATE users SET password = ? WHERE username = ?",
            (new_hash, username),
        )
    return True


def create_user(username: str, password: str, role: str) -> bool:
    try:
        with get_db() as db:
            db.execute(
                "INSERT INTO users (username, password, role) VALUES (?, ?, ?)",
                (username, hash_password(password), role),
            )
        return True
    except sqlite3.IntegrityError:
        return False


def reset_user_password(username: str, new_pass: str) -> None:
    with get_db() as db:
        db.execute(
            "UPDATE users SET password = ? WHERE username = ?",
            (hash_password(new_pass), username),
        )


def delete_user(username: str) -> None:
    with get_db() as db:
        db.execute("DELETE FROM users WHERE username = ?", (username,))


# Initialize DB and admin
init_db()
ensure_admin_user()

# ======================================================
# SESSION AUTH STATE
# ======================================================
if "auth" not in st.session_state:
    st.session_state["auth"] = {
        "logged_in": False,
        "username": None,
        "role": None,
        "login_time": None,
        "last_active": None,
    }

if "login_security" not in st.session_state:
    st.session_state["login_security"] = {"attempts": 0, "locked_until": None}


def check_session_timeout():
    auth = st.session_state.get("auth", {})
    if not auth.get("logged_in"):
        return
    last_active = auth.get("last_active")
    if not last_active:
        return
    try:
        last_dt = datetime.datetime.fromisoformat(last_active)
    except Exception:
        return
    if datetime.datetime.now() - last_dt > datetime.timedelta(
        minutes=SESSION_TIMEOUT_MINUTES
    ):
        st.session_state["auth"] = {
            "logged_in": False,
            "username": None,
            "role": None,
            "login_time": None,
            "last_active": None,
        }
        st.warning("Your session has expired due to inactivity. Please log in again.")


# ======================================================
# LOGIN UI
# ======================================================
def login_screen():
    st.markdown(
        "<h1 style='text-align:center; margin-bottom:0;'>🔐 Login to AI Ticket Classifier</h1>",
        unsafe_allow_html=True,
    )
    st.write("")
    security = st.session_state.get(
        "login_security", {"attempts": 0, "locked_until": None}
    )
    now_ts = time.time()
    locked_until = security.get("locked_until")
    if locked_until and now_ts < locked_until:
        remaining = int(locked_until - now_ts)
        minutes = remaining // 60
        seconds = remaining % 60
        st.error(
            f"Too many failed login attempts. Please try again in {minutes}m {seconds}s."
        )
        return

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username_input = st.text_input("Username")
        password_input = st.text_input("Password", type="password")
        if st.button("Login", use_container_width=True):
            role = authenticate(username_input, password_input)
            if role:
                st.session_state["login_security"] = {
                    "attempts": 0,
                    "locked_until": None,
                }
                st.session_state["auth"] = {
                    "logged_in": True,
                    "username": username_input,
                    "role": role,
                    "login_time": datetime.datetime.now().isoformat(
                        timespec="seconds"
                    ),
                    "last_active": datetime.datetime.now().isoformat(
                        timespec="seconds"
                    ),
                }
                st.success(f"Welcome, **{username_input}** ({role})!")
                st.rerun()
            else:
                attempts = int(security.get("attempts", 0)) + 1
                if attempts >= MAX_LOGIN_ATTEMPTS:
                    st.session_state["login_security"] = {
                        "attempts": 0,
                        "locked_until": now_ts + LOCKOUT_SECONDS,
                    }
                    st.error(
                        "❌ Too many failed attempts. Login has been locked for a short time."
                    )
                else:
                    st.session_state["login_security"] = {
                        "attempts": attempts,
                        "locked_until": None,
                    }
                    st.error("❌ Invalid username or password")


check_session_timeout()
if not st.session_state["auth"]["logged_in"]:
    login_screen()
    st.stop()

st.session_state["auth"]["last_active"] = datetime.datetime.now().isoformat(
    timespec="seconds"
)

username = st.session_state["auth"]["username"]
role = st.session_state["auth"]["role"]
is_admin = role == "admin"
can_bulk = role in ("admin", "engineer")

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
    "Active Directory, Network, Hardware, Email, Firewall, MDM, Printer & Security issues.</div>",
    unsafe_allow_html=True,
)


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
    st_lottie(lottie_ai, height=220, key="hero_animation")

# ======================================================
# MODEL / VECTORIZER
# ======================================================
@st.cache_resource
def load_artifacts():
    try:
        model = joblib.load("model.pkl")
        vectorizer = joblib.load("vectorizer.pkl")
        return model, vectorizer, None
    except Exception as e:
        return None, None, str(e)


model, vectorizer, model_error = load_artifacts()

# ======================================================
# TICKET / LOGIC UTILITIES
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
    return suggestions.get(
        category, "No predefined suggestion available for this category yet."
    )


def compute_severity(ticket: str, category: str) -> str:
    text = ticket.lower()
    if any(
        k in text
        for k in ["all users", "server down", "data breach", "cannot access", "production down"]
    ):
        return "Critical"
    if any(
        k in text for k in ["can't login", "cant login", "urgent", "very slow", "vpn disconnecting"]
    ):
        return "High"
    if any(k in text for k in ["sometimes", "intermittent", "some users"]):
        return "Medium"
    return "Low"


def log_prediction(ticket, prediction, confidence, severity):
    with get_db() as db:
        db.execute(
            """
            INSERT INTO tickets (timestamp, ticket, prediction, confidence, severity, user)
            VALUES (?, ?, ?, ?, ?, ?)
            """,
            (
                datetime.datetime.now().isoformat(timespec="seconds"),
                ticket,
                prediction,
                float(confidence),
                severity,
                username,
            ),
        )


def generate_user_email(ticket, category, severity):
    return (
        f"Subject: Your Support Ticket Update ({category})\n\n"
        f"Hi,\n\n"
        f"Your issue has been classified as {category} with severity {severity}.\n"
        f'Ticket: "{ticket}"\n\n'
        "Our IT team is working on it and will update you soon.\n"
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
    return len(set_a & set_b) / len(set_b | set_a) if set_a and set_b else 0.0


def find_similar_tickets(current_ticket: str, top_n: int = 3):
    with get_db() as db:
        df = pd.read_sql_query("SELECT * FROM tickets", db)
    if df.empty or "ticket" not in df.columns:
        return None
    df["similarity"] = df["ticket"].astype(str).apply(
        lambda t: jaccard_similarity(current_ticket, t)
    )
    df = df.sort_values("similarity", ascending=False)
    df = df[df["similarity"] > 0]
    return df.head(top_n) if not df.empty else None


# ======================================================
# NOTIFICATIONS (EMAIL + SLACK)
# ======================================================
def send_email_notification(subject: str, body: str, to_address: str = None):
    if not ENABLE_EMAIL_NOTIFICATIONS:
        return False, "Email notifications disabled"

    to_address = to_address or EMAIL_TO_IT
    if not (EMAIL_SMTP_HOST and EMAIL_SMTP_USER and EMAIL_SMTP_PASSWORD and to_address):
        return False, "Email SMTP configuration missing"

    try:
        msg = EmailMessage()
        msg["Subject"] = subject
        msg["From"] = EMAIL_FROM or EMAIL_SMTP_USER
        msg["To"] = to_address
        msg.set_content(body)

        with smtplib.SMTP(EMAIL_SMTP_HOST, EMAIL_SMTP_PORT) as server:
            server.starttls()
            server.login(EMAIL_SMTP_USER, EMAIL_SMTP_PASSWORD)
            server.send_message(msg)

        return True, ""
    except Exception as e:
        return False, str(e)


def send_slack_notification(text: str):
    if not ENABLE_SLACK_NOTIFICATIONS:
        return False, "Slack notifications disabled"

    if not SLACK_WEBHOOK_URL:
        return False, "Slack webhook URL not configured"

    try:
        resp = requests.post(SLACK_WEBHOOK_URL, json={"text": text}, timeout=5)
        if 200 <= resp.status_code < 300:
            return True, ""
        return False, f"HTTP {resp.status_code}: {resp.text}"
    except Exception as e:
        return False, str(e)


def notify_critical_or_high(ticket: str, category: str, severity: str, confidence: float, created_by: str):
    if severity not in ("Critical", "High"):
        return

    subject = f"[{severity}] {category} ticket from {created_by}"
    body = (
        f"Severity : {severity}\n"
        f"Category : {category}\n"
        f"Confidence: {confidence * 100:.2f}%\n"
        f"Raised by: {created_by}\n\n"
        f"Ticket text:\n{ticket}\n"
    )

    email_ok, email_err = send_email_notification(subject, body)

    slack_text = (
        f"*{severity}* {category} ticket from *{created_by}* \n"
        f"> {simple_summary(ticket, max_words=25)}\n"
        f"_Model confidence: {confidence * 100:.1f}%_"
    )
    slack_ok, slack_err = send_slack_notification(slack_text)

    if is_admin:
        if email_ok:
            st.success("📧 Email notification sent to IT team.")
        elif email_err and ENABLE_EMAIL_NOTIFICATIONS:
            st.warning(f"Email notification failed: {email_err}")

        if slack_ok:
            st.success("💬 Slack notification sent.")
        elif slack_err and ENABLE_SLACK_NOTIFICATIONS:
            st.warning(f"Slack notification failed: {slack_err}")


# ======================================================
# AI HELP ASSISTANT
# ======================================================
def generate_ai_help(ticket_text: str, category: str, severity: str) -> str:
    text = ticket_text.lower()
    base_fix = suggest_fix(category)
    extra = []

    if "printer" in text:
        extra.extend(
            [
                "- Make sure the printer is powered on and connected to Wi-Fi/LAN.",
                "- Open **Devices & Printers** and check if the printer shows as *Online*.",
                "- Try removing the printer and adding it again.",
            ]
        )
    if "vpn" in text:
        extra.extend(
            [
                "- Confirm your internet works without VPN.",
                "- Check VPN username/password are correct.",
                "- Try disconnect → reconnect, or restart the VPN client.",
            ]
        )
    if "wifi" in text or "wi-fi" in text:
        extra.extend(
            [
                "- Check if other devices can join the same Wi-Fi.",
                "- Forget the network and reconnect.",
                "- Move closer to the router or access point.",
            ]
        )
    if "outlook" in text or "email" in text:
        extra.extend(
            [
                "- Close and reopen Outlook.",
                "- Check if mailbox is full.",
                "- Try sending a test email to yourself.",
            ]
        )
    if "slow" in text or "performance" in text:
        extra.extend(
            [
                "- Close unused apps and browser tabs.",
                "- Restart the device if it hasn’t been restarted recently.",
                "- Check Task Manager for apps using high CPU or RAM.",
            ]
        )

    tips = [f"- {base_fix}"] + extra if base_fix else extra
    tips_str = "\n".join(dict.fromkeys(tips))  # remove duplicates

    message = (
        "### 🤖 Smart Help Summary\n"
        f"- Predicted area: **{category}**\n"
        f"- Estimated severity: **{severity}**\n\n"
        "### ✅ Suggested steps\n"
        f"{tips_str or '- No specific suggestions yet. Please contact IT support.'}\n\n"
        "If these steps don’t solve your issue, please raise a ticket so the IT team can help you further."
    )
    return message


# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.header("⚙️ Menu")
    st.write(f"Logged in as: **{username} ({role})**")

    if st.button("Logout", use_container_width=True):
        st.session_state["auth"] = {
            "logged_in": False,
            "username": None,
            "role": None,
            "login_time": None,
            "last_active": None,
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
# TABS
# ======================================================
tabs = ["📝 Single Ticket", "🤖 AI Help Assistant", "📂 Bulk CSV"]
if is_admin:
    tabs.append("👥 User Management")
tabs.append("👤 User Insights")
tabs.append("📊 History")

tab_objects = st.tabs(tabs)
tab_idx = 0

# ------------------------------------------------------
# 1) Single Ticket
# ------------------------------------------------------
with tab_objects[tab_idx]:
    st.subheader("Classify a single IT support ticket")

    ticket_text = st.text_area("Ticket Description", height=120)

    if example_choice != "(None)" and not ticket_text.strip():
        ticket_text = example_choice
        st.info(f"Using example: _{ticket_text}_")

    if st.button("Predict Category"):
        if model is None or vectorizer is None:
            msg = "Model artifacts could not be loaded. Please contact the administrator."
            if model_error:
                msg += f" (Details: {model_error})"
            st.error(msg)
        elif not ticket_text.strip():
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
                f'<div class="prediction-box">Predicted Category: '
                f"<b>{prediction}</b></div>",
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
            notify_critical_or_high(ticket_text, prediction, severity, top_conf, username)

            st.markdown("### 🔎 Similar past tickets")
            similar = find_similar_tickets(ticket_text, 3)
            if similar is not None:
                st.dataframe(similar)
            else:
                st.info("No similar tickets found yet.")

tab_idx += 1

# ------------------------------------------------------
# 2) AI Help Assistant
# ------------------------------------------------------
with tab_objects[tab_idx]:
    st.subheader("AI Help Assistant for Employees")

    help_text = st.text_area(
        "Describe your IT issue in your own words (e.g., 'My printer is not connecting to the network')",
        height=120,
        key="ai_help_text",
    )

    if st.button("Get Smart Help"):
        if model is None or vectorizer is None:
            st.error("The AI model is not available at the moment.")
        elif not help_text.strip():
            st.warning("Please describe your issue first.")
        else:
            vec = vectorizer.transform([help_text])
            prediction = model.predict(vec)[0]
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(vec)[0]
                top_conf = probas[np.argmax(probas)]
            else:
                top_conf = 1.0 / len(model.classes_)
            severity = compute_severity(help_text, prediction)
            msg = generate_ai_help(help_text, prediction, severity)
            st.markdown(msg)

tab_idx += 1

# ------------------------------------------------------
# 3) Bulk CSV (admin/engineer)
# ------------------------------------------------------
with tab_objects[tab_idx]:
    st.subheader("Bulk classify tickets from CSV")

    if not can_bulk:
        st.warning("Only **admin** and **engineer** roles can use bulk CSV.")
    elif model is None or vectorizer is None:
        msg = "Bulk prediction is unavailable because the model could not be loaded."
        if model_error:
            msg += f" (Details: {model_error})"
        st.error(msg)
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

                df["severity"] = [
                    compute_severity(t, c)
                    for t, c in zip(df["ticket"].astype(str), df["prediction"])
                ]

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
# 4) User Management (admin only)
# ------------------------------------------------------
with tab_objects[tab_idx]:
    if not is_admin:
        st.warning("Only admin users can manage accounts.")
    else:
        st.subheader("User Management (Admin only)")

        users_df = load_users()
        st.dataframe(users_df, use_container_width=True)

        st.markdown("### ➕ Add new user")
        new_user = st.text_input("New username")
        new_pass = st.text_input("New password", type="password")
        new_role = st.selectbox("Role", ["admin", "engineer", "user"])

        if st.button("Create user"):
            if not new_user or not new_pass:
                st.warning("Username and password cannot be empty.")
            else:
                created = create_user(new_user, new_pass, new_role)
                if created:
                    st.success("User created.")
                    st.experimental_rerun()
                else:
                    st.error("User already exists.")

        st.markdown("### 🔁 Reset user password")
        if not users_df.empty:
            reset_user_name = st.selectbox(
                "Select user to reset", users_df["username"], key="reset_user_sel"
            )
            reset_new_pw = st.text_input(
                "New password for selected user", type="password", key="reset_new_pw"
            )
            if st.button("Reset password for user"):
                if not reset_new_pw:
                    st.warning("Please enter a new password.")
                else:
                    reset_user_password(reset_user_name, reset_new_pw)
                    st.success(f"Password reset for user '{reset_user_name}'.")
                    st.experimental_rerun()

        st.markdown("### 🗑 Delete user")
        if not users_df.empty:
            del_user_name = st.selectbox(
                "Select user to delete", users_df["username"], key="del_user_sel"
            )
            if st.button("Delete user"):
                if del_user_name == "admin":
                    st.error("You cannot delete the main admin account.")
                else:
                    delete_user(del_user_name)
                    st.success("User deleted.")
                    st.experimental_rerun()

tab_idx += 1

# ------------------------------------------------------
# 5) User Insights
# ------------------------------------------------------
with tab_objects[tab_idx]:
    st.subheader("User Insights")

    with get_db() as db:
        hist = pd.read_sql_query("SELECT * FROM tickets", db)

    if hist.empty:
        st.info("No prediction history yet.")
    else:
        users_list = sorted(hist["user"].dropna().unique().tolist())
        if not users_list:
            st.info("No user information stored with tickets yet.")
        else:
            selected_user = st.selectbox("Select a user", users_list)

            user_hist = hist[hist["user"] == selected_user]

            st.markdown(f"### Overview for **{selected_user}**")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Total tickets", len(user_hist))
            with col_b:
                st.metric("Unique categories", user_hist["prediction"].nunique())
            with col_c:
                critical_count = (user_hist["severity"] == "Critical").sum()
                st.metric("Critical tickets", int(critical_count))

            st.markdown("### Recent tickets")
            st.dataframe(user_hist.sort_values("timestamp", ascending=False))

            st.markdown("### Category breakdown")
            st.write(user_hist["prediction"].value_counts())

            st.markdown("### Severity breakdown")
            st.write(user_hist["severity"].value_counts())

            st.markdown("### Repeated tickets (same description)")
            dup = (
                user_hist.groupby("ticket")
                .size()
                .reset_index(name="count")
                .sort_values("count", ascending=False)
            )
            repeated = dup[dup["count"] > 1]
            if repeated.empty:
                st.info("No repeated tickets for this user.")
            else:
                st.dataframe(repeated)

tab_idx += 1

# ------------------------------------------------------
# 6) History + Dashboard
# ------------------------------------------------------
with tab_objects[tab_idx]:
    st.subheader("📊 Prediction history & dashboard")

    with get_db() as db:
        hist = pd.read_sql_query("SELECT * FROM tickets", db)

    if hist.empty:
        st.info("No prediction history yet.")
    else:
        total_tickets = len(hist)
        avg_conf = (
            float(hist["confidence"].mean()) if "confidence" in hist.columns else 0.0
        )
        critical_count = (
            int((hist["severity"] == "Critical").sum())
            if "severity" in hist.columns
            else 0
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total tickets", total_tickets)
        with col2:
            st.metric("Avg confidence", f"{avg_conf * 100:.1f}%")
        with col3:
            st.metric("Critical tickets", critical_count)

        st.markdown("### Tickets by category")
        if "prediction" in hist.columns:
            cat_counts = hist["prediction"].value_counts().to_frame(name="count")
            st.bar_chart(cat_counts)
        else:
            st.info("No 'prediction' column found in history.")

        st.markdown("### Tickets by severity")
        if "severity" in hist.columns:
            sev_counts = hist["severity"].value_counts().to_frame(name="count")
            st.bar_chart(sev_counts)
        else:
            st.info("No 'severity' column found in history.")

        st.markdown("### Tickets over time")
        if "timestamp" in hist.columns:
            dates = hist["timestamp"].astype(str).str.slice(0, 10)
            date_counts = dates.value_counts().sort_index().to_frame(name="count")
            st.line_chart(date_counts)
        else:
            st.info("No 'timestamp' column found in history.")

        st.markdown("### Raw history table")
        st.dataframe(hist, use_container_width=True)
