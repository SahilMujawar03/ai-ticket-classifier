import streamlit as st
import pandas as pd
import joblib
import os

# ----------------- LOAD MODEL -----------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_artifacts()

USERS_FILE = "users.csv"

# ----------------- USER SYSTEM -----------------
def load_users():
    if not os.path.exists(USERS_FILE):
        df = pd.DataFrame(columns=["username", "password", "role"])
        df.to_csv(USERS_FILE, index=False)
        return df
    return pd.read_csv(USERS_FILE)

def save_users(df):
    df.to_csv(USERS_FILE, index=False)

def authenticate(username, password):
    users = load_users()
    row = users[(users["username"] == username) & (users["password"] == password)]
    if len(row) == 1:
        return True, row.iloc[0]["role"]
    return False, None

# ----------------- CHANGE PASSWORD -----------------
def change_password(username, old_pass, new_pass):
    users = load_users()
    row = users[(users["username"] == username) & (users["password"] == old_pass)]
    if len(row) == 0:
        return False
    
    users.loc[users["username"] == username, "password"] = new_pass
    save_users(users)
    return True


# ----------------- LOGIN SCREEN -----------------
def login_screen():
    st.title("🔐 Login to AI Ticket Classifier")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        ok, role = authenticate(username, password)
        if ok:
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role
            st.experimental_rerun()
        else:
            st.error("Invalid username or password")


# ----------------- MAIN APP -----------------
def app_home():
    st.title("🧠 AI Ticket Classifier")
    st.write("Smart classification of IT support tickets.")

    tab1, tab2, tab3 = st.tabs([
        "🎫 Single Ticket",
        "📊 Bulk CSV",
        "👥 User Management" if st.session_state["role"] == "admin" else "👤 Change Password"
    ])

    # ------- Single Ticket -------
    with tab1:
        st.subheader("Classify a single IT ticket")
        ticket = st.text_area("Ticket Description")
        
        if st.button("Predict Category"):
            if ticket.strip() == "":
                st.warning("Please enter a ticket description.")
            else:
                X = vectorizer.transform([ticket])
                pred = model.predict(X)[0]
                st.success(f"Predicted Category: **{pred}**")

    # ------- Bulk CSV (Admin + Engineer) -------
    with tab2:
        st.subheader("Upload CSV")
        st.info("Only admin or engineer can use this.")
        
        if st.session_state["role"] in ["admin", "engineer"]:
            file = st.file_uploader("Upload CSV", type=["csv"])
            if file:
                df = pd.read_csv(file)
                df["prediction"] = model.predict(vectorizer.transform(df["ticket"]))
                st.dataframe(df)
        else:
            st.error("You do not have permission for this section.")

    # ------- Admin User Management -------
    if st.session_state["role"] == "admin":
        with tab3:
            st.subheader("Manage Users")
            users = load_users()
            st.dataframe(users)

            st.write("### ➕ Add New User")
            new_user = st.text_input("New Username")
            new_pass = st.text_input("New Password")
            new_role = st.selectbox("Role", ["admin", "engineer", "user"])

            if st.button("Create User"):
                if new_user in users["username"].values:
                    st.error("User already exists!")
                else:
                    new_row = pd.DataFrame({"username":[new_user], "password":[new_pass], "role":[new_role]})
                    save_users(pd.concat([users, new_row], ignore_index=True))
                    st.success("User added successfully!")
                    st.experimental_rerun()

    # ------- User Change Password (non-admin) -------
    else:
        with tab3:
            st.subheader("Change Password")
            old_pass = st.text_input("Old Password", type="password")
            new_pass = st.text_input("New Password", type="password")

            if st.button("Update Password"):
                if change_password(st.session_state["username"], old_pass, new_pass):
                    st.success("Password updated successfully!")
                else:
                    st.error("Incorrect old password.")

    # ------- Logout -------
    st.sidebar.button("Logout", on_click=lambda: logout())


# ----------------- LOGOUT -----------------
def logout():
    for key in ["logged_in", "username", "role"]:
        st.session_state.pop(key, None)
    st.experimental_rerun()


# ----------------- APP ENTRY -----------------
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if not st.session_state["logged_in"]:
    login_screen()
else:
    app_home()
