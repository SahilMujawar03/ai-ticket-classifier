import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime

from streamlit_lottie import st_lottie   # animation
import requests                          # load animation JSON

# ----------------- BASIC CONFIG -----------------
st.set_page_config(
    page_title="AI Ticket Classifier",
    page_icon="💻",
    layout="wide",
)

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
    """Load a Lottie animation from a URL."""
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
ADMIN_PASSWORD = st.secrets.get("ADMIN_PASSWORD", "admin123")


def simple_summary(text: str, max_words: int = 18) -> str:
    """Very simple 'AI-style' summary: first N words."""
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words]) + "..."


def suggest_fix(category: str) -> str:
    suggestions = {
        "Active Directory Issue": "Check account lockout, reset password, verify group membership in AD.",
        "Hardware Issue": "Ask user to reboot, check cables, run hardware diagnostics or swap peripherals.",
        "Network Issue": "Check WiFi/LAN connection, ping gateway, restart router/switch, verify VPN.",
        "Email Issue": "Recreate Outlook profile, check mailbox size, verify SMTP/Exchange connection.",
        "Firewall Issue": "Check firewall rule for required ports, NAT policies, and recent rule changes.",
        "MDM Issue": "Re-enrol device, sync MDM policies, verify compliance and certificate status.",
        "Printer Issue": "Check printer network status, reinstall drivers, clear print queue, test print.",
        "Security Issue": "Run full AV scan, isolate device if needed, reset credentials, check logs.",
    }
    return suggestions.get(category, "No predefined suggestion available for this category.")


def compute_severity(ticket: str, category: str) -> str:
    """
    Simple rule-based severity:
    Critical / High / Medium / Low based on keywords + category.
    """
    text = ticket.lower()

    # Critical: everything is down / all users / security
    critical_keywords = [
        "all users", "everyone", "server down", "system down",
        "cannot access", "can't access", "completely down",
        "data breach", "ransomware", "virus outbreak"
    ]
    if any(k in text for k in critical_keywords):
        return "Critical"

    # High: login issues, repeated disconnects, major impact
    high_keywords = [
        "unable to login", "unable to log in", "can't login", "cant login",
        "vpn disconnecting", "keeps disconnecting", "frequent disconnect",
        "high priority", "urgent", "very slow", "not working at all"
    ]
    if any(k in text for k in high_keywords) or category in [
        "Active Directory Issue", "Network Issue", "Security Issue"
    ]:
        return "High"

    # Medium: intermittent / sometimes / some users / annoying but not full stop
    medium_keywords = [
        "sometimes", "intermittent", "occasionally", "some users",
        "delay", "latency", "partially", "not consistent"
    ]
    if any(k in text for k in medium_keywords):
        return "Medium"

    # Default: Low
    return "Low"


def log_prediction(ticket: str, prediction: str, confidence: float, severity: str):
    """Append prediction info to a CSV log (with severity)."""
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


def generate_user_email(ticket: str, category: str, severity: str) -> str:
    """Generate a simple email template to send to the end user."""
    return (
        f"Subject: Update on your support request ({category})\n\n"
        f"Hi,\n\n"
        f"Thank you for raising this issue. Our system has classified your request as "
        f"**{category}** with a severity level of **{severity}**.\n\n"
        f"Ticket summary:\n"
        f"\"{ticket}\"\n\n"
        f"Our IT team is now reviewing this and will update you with the next steps as soon as possible.\n\n"
        f"Best regards,\n"
        f"IT Support Team"
    )


def generate_internal_note(ticket: str, category: str, severity: str, confidence: float) -> str:
    """Generate an internal IT note for engineers."""
    return (
        f"Issue classification: {category}\n"
        f"Severity: {severity}\n"
        f"Model confidence: {confidence*100:.2f}%\n\n"
        f"Original ticket:\n"
        f"{ticket}\n\n"
        f"Suggested starting actions:\n"
        f"- {suggest_fix(category)}\n"
        f"- Check relevant logs/monitoring tools for this category.\n"
        f"- Update ticket with findings and next steps."
    )


def jaccard_similarity(a: str, b: str) -> float:
    """Very simple text similarity using Jaccard over word sets."""
    set_a = set(a.lower().split())
    set_b = set(b.lower().split())
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def find_similar_tickets(current_ticket: str, top_n: int = 3):
    """Return up to top_n similar past tickets from prediction_log.csv."""
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
    hist_df = hist_df[hist_df["similarity"] > 0]  # only those with some overlap
    return hist_df.head(top_n) if not hist_df.empty else None

# ----------------- SIDEBAR -----------------
with st.sidebar:
    st.header("ℹ️ About this app")
    st.write(
        """
        This tool uses a Machine Learning model (Random Forest + TF-IDF)
        trained on real-world IT support tickets to automatically
        categorize new requests.
        """
    )

    st.markdown("**Example tickets:**")
    example_tickets = [
        "User cannot login to Active Directory",
        "Outlook not sending or receiving emails",
        "Laptop is very slow and overheating",
        "VPN keeps disconnecting when user works from home",
        "Printer not connecting to network in office",
        "Antivirus alert: malware detected on workstation",
    ]
    example_choice = st.selectbox(
        "Choose an example ticket:",
        ["(None)"] + example_tickets,
        index=0,
    )

    st.markdown("---")
    st.subheader("🔐 Admin login")
    password = st.text_input("Password", type="password", placeholder="Enter admin password")
    is_admin = password == ADMIN_PASSWORD
    if password and not is_admin:
        st.error("Incorrect password.")
    elif is_admin:
        st.success("Admin mode enabled.")

# ----------------- MAIN LAYOUT -----------------
tab_single, tab_bulk, tab_history, tab_dashboard = st.tabs(
    ["📝 Single Ticket", "📂 Bulk CSV (Admin)", "📊 History (Admin)", "📈 Dashboard (Admin)"]
)

# ---------- SINGLE TICKET TAB ----------
with tab_single:
    st.subheader("Classify a single IT support ticket")

    ticket_text = st.text_area("Ticket Description", height=120)

    # If user selected an example and text area is still empty, prefill it
    if example_choice != "(None)" and not ticket_text.strip():
        ticket_text = example_choice
        st.info(f"Using example ticket: _{ticket_text}_")

    if st.button("Predict Category", key="predict_single"):
        if not ticket_text.strip():
            st.warning("Please enter a ticket description or choose an example from the sidebar.")
        else:
            # Vectorize
            vec = vectorizer.transform([ticket_text])

            # Main prediction
            prediction = model.predict(vec)[0]

            # Probabilities
            if hasattr(model, "predict_proba"):
                probas = model.predict_proba(vec)[0]
                classes = model.classes_
                top_idx = np.argsort(probas)[::-1][:3]
                top_conf = probas[top_idx[0]]
            else:
                # If model has no predict_proba
                classes = model.classes_
                probas = np.ones(len(classes)) / len(classes)
                top_idx = np.argsort(probas)[::-1][:3]
                top_conf = probas[top_idx[0]]

            # Compute severity
            severity = compute_severity(ticket_text, prediction)

            # Show prediction
            st.markdown(
                f'<div class="prediction-box">Predicted Category: '
                f'<b>{prediction}</b></div>',
                unsafe_allow_html=True,
            )

            st.markdown("### 🚨 Severity")
            st.info(f"Severity level: **{severity}**")

            # Summary & suggestion
            st.markdown("### 🧾 Quick summary")
            st.info(simple_summary(ticket_text))

            st.markdown("### 🛠 Suggested fix")
            st.info(suggest_fix(prediction))

            # Confidence section
            st.markdown("### 🔍 Confidence (top 3)")
            for idx in top_idx:
                st.write(f"- {classes[idx]} — **{probas[idx] * 100:.1f}%**")

            # Progress bar for top confidence
            st.markdown("### 📈 Confidence bar")
            st.progress(float(top_conf))

            # NEW: Email templates
            st.markdown("### 📧 Email templates")
            user_email = generate_user_email(ticket_text, prediction, severity)
            internal_note = generate_internal_note(ticket_text, prediction, severity, top_conf)

            with st.expander("📨 Email to user"):
                st.code(user_email, language="markdown")

            with st.expander("📝 Internal IT note"):
                st.code(internal_note, language="markdown")

            # Log it (now with severity)
            log_prediction(ticket_text, prediction, top_conf, severity)

            # NEW: Similar past tickets
            st.markdown("### 🔎 Similar past tickets")
            similar = find_similar_tickets(ticket_text, top_n=3)
            if similar is not None:
                st.dataframe(
                    similar[["timestamp", "ticket", "prediction", "severity", "similarity"]]
                )
            else:
                st.info("No similar tickets found in history yet.")

            with st.expander("Show raw model info"):
                st.write("Classes learned by the model:")
                st.write(list(model.classes_))

# ---------- BULK CSV TAB (ADMIN) ----------
with tab_bulk:
    st.subheader("Bulk classify tickets from CSV")

    if not is_admin:
        st.warning("Admin access required. Enter password in the sidebar to use this feature.")
    else:
        st.write("Upload a CSV file with a column named **'ticket'** containing ticket texts.")

        uploaded_file = st.file_uploader("Upload CSV", type=["csv"], key="bulk_uploader")
        if uploaded_file is not None:
            try:
                bulk_df = pd.read_csv(uploaded_file)
            except Exception as e:
                st.error(f"Failed to read CSV: {e}")
                bulk_df = None

            if bulk_df is not None:
                if "ticket" not in bulk_df.columns:
                    st.error("CSV must contain a column named 'ticket'.")
                else:
                    st.success(f"Loaded {len(bulk_df)} tickets.")
                    st.dataframe(bulk_df.head())

                    if st.button("Run Bulk Classification"):
                        vec_bulk = vectorizer.transform(bulk_df["ticket"])
                        preds = model.predict(vec_bulk)

                        # If model has predict_proba, get max confidence
                        if hasattr(model, "predict_proba"):
                            prob_bulk = model.predict_proba(vec_bulk)
                            max_conf = prob_bulk.max(axis=1)
                        else:
                            max_conf = np.ones(len(preds)) / len(model.classes_)

                        result_df = bulk_df.copy()
                        result_df["predicted_category"] = preds
                        result_df["confidence"] = max_conf
                        # Severity for each ticket
                        result_df["severity"] = [
                            compute_severity(t, c) for t, c in zip(result_df["ticket"], preds)
                        ]

                        st.markdown("### ✅ Bulk results")
                        st.dataframe(result_df.head(20))

                        # Download button
                        csv_out = result_df.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "Download results as CSV",
                            data=csv_out,
                            file_name="ticket_predictions.csv",
                            mime="text/csv",
                        )

# ---------- HISTORY TAB (ADMIN) ----------
with tab_history:
    st.subheader("Prediction history")

    if not is_admin:
        st.warning("Admin access required. Enter password in the sidebar to view history.")
    else:
        try:
            hist_df = pd.read_csv("prediction_log.csv")
            st.write(f"Total logged predictions: **{len(hist_df)}**")
            st.dataframe(hist_df.tail(50))

            # Simple stats by category
            st.markdown("### 📊 Predictions per category")
            if "prediction" in hist_df.columns:
                counts = hist_df["prediction"].value_counts()
                st.bar_chart(counts)

            # Severity distribution
            if "severity" in hist_df.columns:
                st.markdown("### 🚨 Severity distribution")
                sev_counts = hist_df["severity"].value_counts()
                st.bar_chart(sev_counts)
        except FileNotFoundError:
            st.info("No history found yet. Make some predictions first.")

# ---------- DASHBOARD TAB (ADMIN) ----------
with tab_dashboard:
    st.subheader("📈 Ticket Analytics Dashboard")

    if not is_admin:
        st.warning("Admin access required. Enter password in the sidebar to view dashboard.")
    else:
        try:
            dash_df = pd.read_csv("prediction_log.csv")

            # Ensure columns
            if "timestamp" in dash_df.columns:
                dash_df["date"] = pd.to_datetime(dash_df["timestamp"]).dt.date

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total tickets", len(dash_df))
            with col2:
                if "prediction" in dash_df.columns:
                    st.metric("Unique categories", dash_df["prediction"].nunique())
            with col3:
                if "severity" in dash_df.columns:
                    st.metric("Distinct severities", dash_df["severity"].nunique())

            st.markdown("### 📊 Tickets per category")
            if "prediction" in dash_df.columns:
                cat_counts = dash_df["prediction"].value_counts()
                st.bar_chart(cat_counts)

            st.markdown("### 🚨 Tickets per severity")
            if "severity" in dash_df.columns:
                sev_counts = dash_df["severity"].value_counts().reindex(
                    ["Low", "Medium", "High", "Critical"]
                ).fillna(0)
                st.bar_chart(sev_counts)

            if "date" in dash_df.columns:
                st.markdown("### ⏱ Tickets over time")
                time_counts = dash_df.groupby("date").size()
                st.line_chart(time_counts)

        except FileNotFoundError:
            st.info("No data yet. Make some predictions to see analytics.")
