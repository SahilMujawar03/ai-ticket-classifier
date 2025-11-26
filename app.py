import numpy as np
import pandas as pd
import streamlit as st
import joblib
import datetime
import requests
from streamlit_lottie import st_lottie
import os

# ----------------- BASIC CONFIG -----------------
st.set_page_config(
    page_title="AI Ticket Classifier",
    page_icon="💻",
    layout="wide",
)

# ----------------- USER MANAGEMENT (CSV) -----------------

USER_FILE = "users.csv"

# Create default users.csv if missing
if not os.path.exists(USER_FILE):
    df_default = pd.DataFrame([
        ["admin", "admin123", "admin"],
        ["engineer", "engineer123", "engineer"],
        ["user", "user123", "user"]
    ], columns=["username", "password", "role"])
    df_default.to_csv(USER_FILE, index=False)

def load_users():
    return pd.read_csv(USER_FILE)

def save_users(df):
    df.to_csv(USER_FILE, index=False)

# ----------------- AUTH STATE -----------------
def init_auth_state():
    if "auth" not in st.session_state:
        st.session_state["auth"] = {
            "logged_in": False,
            "username": None,
            "role": None,
        }

init_auth_state()

# ----------------- LOGIN SCREEN -----------------
def login_screen():
    st.markdown("<h2 style='text-align:center;'>🔐 Login to AI Ticket Classifier</h2>", unsafe_allow_html=True)
    st.write("")
    
    col1, col2, col3 = st.columns([1,2,1])
    with col2:
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            users = load_users()
            
            match = users[(users["username"] == username) & (users["password"] == password)]
            
            if not match.empty:
                role = match["role"].values[0]

                st.session_state["auth"] = {
                    "logged_in": True,
                    "username": username,
                    "role": role,
                }
                st.success(f"Welcome, {username} ({role})!")
                st.rerun()
            else:
                st.error("Invalid username or password.")

if not st.session_state["auth"]["logged_in"]:
    login_screen()
    st.stop()

username = st.session_state["auth"]["username"]
role = st.session_state["auth"]["role"]

# ----------------- STYLE -----------------
st.markdown("""
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
""", unsafe_allow_html=True)

# ----------------- TITLE -----------------
st.markdown('<div class="big-title">🧠 AI Ticket Classifier</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Smart classification of IT support tickets.</div>', unsafe_allow_html=True)

# ----------------- ANIMATION -----------------
def load_lottie(url: str):
    try:
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    except:
        return None

lottie_url = "https://assets2.lottiefiles.com/packages/lf20_kyu7xb1v.json"
lottie_ai = load_lottie(lottie_url)

if lottie_ai:
    st_lottie(lottie_ai, height=230, key="aiAnimation")

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_artifacts():
    return joblib.load("model.pkl"), joblib.load("vectorizer.pkl")

model, vectorizer = load_artifacts()

# ----------------- TABS -----------------
tab_single, tab_bulk, tab_users = st.tabs(["📝 Single Ticket", "📂 Bulk CSV", "👥 User Management"])

# ----------------- 1. SINGLE TICKET -----------------
with tab_single:
    st.subheader("Classify a single IT ticket")
    ticket_text = st.text_area("Ticket Description", height=120)

    if st.button("Predict Category"):
        if not ticket_text.strip():
            st.warning("Please enter a ticket description.")
        else:
            vec = vectorizer.transform([ticket_text])
            prediction = model.predict(vec)[0]

            st.markdown(f'<div class="prediction-box">Predicted Category: <b>{prediction}</b></div>', unsafe_allow_html=True)

# ----------------- 2. BULK CSV -----------------
with tab_bulk:
    if role not in ["admin", "engineer"]:
        st.warning("Only admin/engineers can use bulk feature")
    else:
        uploaded = st.file_uploader("Upload CSV with a 'ticket' column")
        if uploaded:
            df = pd.read_csv(uploaded)
            df["prediction"] = model.predict(vectorizer.transform(df["ticket"]))
            st.dataframe(df)
            st.download_button("Download results", df.to_csv(index=False), "bulk_results.csv")

# ----------------- 3. USER MANAGEMENT -----------------
with tab_users:
    st.subheader("User Management (Admin Only)")

    if role != "admin":
        st.warning("Only admin can manage users.")
        st.stop()

    users = load_users()
    st.dataframe(users)

    st.markdown("### ➕ Add New User")
    new_user = st.text_input("Username")
    new_pass = st.text_input("Password")
    new_role = st.selectbox("Role", ["admin", "engineer", "user"])

    if st.button("Add User"):
        if new_user.strip() == "" or new_pass.strip() == "":
            st.error("Fields cannot be empty.")
        else:
            users.loc[len(users)] = [new_user, new_pass, new_role]
            save_users(users)
            st.success("User added successfully.")
            st.rerun()

    st.markdown("### 🗑 Delete User")
    del_user = st.selectbox("Select user to delete", users["username"])

    if st.button("Delete User"):
        if del_user == "admin":
            st.error("Admin cannot be deleted.")
        else:
            users = users[users["username"] != del_user]
            save_users(users)
            st.success("User deleted.")
            st.rerun()
