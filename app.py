import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime

from streamlit_lottie import st_lottie   # 🔹 NEW: for animation
import requests                          # 🔹 NEW: to load animation JSON

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

# Load an AI-themed animation
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

def log_prediction(ticket: str, prediction: str, confidence: float):
    """Append prediction info to a CSV log."""
    row = {
        "timestamp": datetime.datetime.now().isoformat(timespec="seconds"),
        "ticket": ticket,
        "prediction": prediction,
        "confidence": round(float(confidence), 4),
    }
    df = pd.DataFrame([row])
    df.to_csv("prediction_log.csv", mode="a", header=not st.session_state.get("log_initialized", False), index=False)
    st.session_state["log_initialized"] = True

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
tab_single, tab_bulk, tab_history = st.tabs(
    ["📝 Single Ticket", "📂 Bulk CSV (Admin)", "📊 History (Admin)"]
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

            # Show prediction
            st.markdown(
                f'<div class="prediction-box">Predicted Category: '
                f'<b>{prediction}</b></div>',
                unsafe_allow_html=True,
            )

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

            # Log it
            log_prediction(ticket_text, prediction, top_conf)

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
            counts = hist_df["prediction"].value_counts()
            st.bar_chart(counts)
        except FileNotFoundError:
            st.info("No history found yet. Make some predictions first.")
