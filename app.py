import streamlit as st
import pandas as pd
import joblib
import datetime

# ------------------ FILE PATHS ------------------
USERS_FILE = "users.csv"
LOG_FILE = "prediction_log.csv"

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="AI Ticket Classifier", layout="wide")


# ------------------ LOAD MODEL ------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("model.pkl")
    vectorizer = joblib.load("vectorizer.pkl")
    return model, vectorizer


model, vectorizer = load_artifacts()


# ------------------ LOAD USERS ------------------
def load_users():
    try:
        return pd.read_csv(USERS_FILE)
    except:
        return pd.DataFrame(columns=["username", "password", "role"])


def save_users(df):
    df.to_csv(USERS_FILE, index=False)


# ------------------ AUTHENTICATION ------------------
def authenticate(username, password):
    users = load_users()
    match = users[(users["username"] == username) & (users["password"] == password)]

    if len(match) == 1:
        return match.iloc[0]["role"]  # Return actual role from CSV
    return None


# ------------------ LOGIN SCREEN ------------------
def login_screen():
    st.title("🔐 Login to AI Ticket Classifier")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        role = authenticate(username, password)

        if role:
            # Save session
            st.session_state["logged_in"] = True
            st.session_state["username"] = username
            st.session_state["role"] = role

            st.success(f"Logged in as **{username}** ({role})")
            st.experimental_set_query_params()  # SAFE rerun
            st.rerun()
        else:
            st.error("❌ Invalid username or password")


# ------------------ LOGOUT ------------------
def logout():
    st.session_state.clear()
    st.rerun()


# ------------------ SINGLE TICKET PREDICTION ------------------
def predict_ticket(text):
    vector = vectorizer.transform([text])
    prediction = model.predict(vector)[0]
    return prediction


# ------------------ LOG SAVING ------------------
def log_prediction(ticket, prediction):
    row = {
        "timestamp": str(datetime.datetime.now()),
        "ticket": ticket,
        "prediction": prediction,
        "user": st.session_state["username"]
    }

    try:
        df = pd.read_csv(LOG_FILE)
        df = pd.concat([df, pd.DataFrame([row])], ignore_index=True)
    except:
        df = pd.DataFrame([row])

    df.to_csv(LOG_FILE, index=False)


# ------------------ USER MANAGEMENT (Admin) ------------------
def user_management():
    st.header("👥 User Management")

    df = load_users()
    st.dataframe(df)

    st.subheader("➕ Add New User")

    new_user = st.text_input("Username")
    new_pass = st.text_input("Password")
    new_role = st.selectbox("Role", ["admin", "engineer", "user"])

    if st.button("Add User"):
        if new_user.strip() == "" or new_pass.strip() == "":
            st.error("Fill all fields!")
        else:
            df.loc[len(df)] = [new_user, new_pass, new_role]
            save_users(df)
            st.success("User added successfully!")
            st.rerun()

    st.subheader("❌ Delete User")
    del_user = st.selectbox("Select user", df["username"])

    if st.button("Delete User"):
        df = df[df["username"] != del_user]
        save_users(df)
        st.success("User deleted!")
        st.rerun()


# ------------------ MAIN APP ------------------
def main_app():
    st.sidebar.title("⚙️ Menu")
    st.sidebar.write(f"Logged in as: **{st.session_state['username']} ({st.session_state['role']})**")
    if st.sidebar.button("Logout"):
        logout()

    tab1, tab2, tab3 = st.tabs(["📝 Single Ticket", "📊 Bulk CSV", "👥 User Management"])

    # ------------------ TAB 1: Single Ticket ------------------
    with tab1:
        st.header("Classify a single IT ticket")
        text = st.text_area("Ticket Description")

        if st.button("Predict Category"):
            if len(text) < 3:
                st.warning("Enter a valid ticket description.")
            else:
                result = predict_ticket(text)
                st.success(f"Predicted Category: **{result}**")
                log_prediction(text, result)

    # ------------------ TAB 2: Bulk CSV ------------------
    with tab2:
        st.header("Upload CSV for bulk classification")

        file = st.file_uploader("Upload CSV file", type=["csv"])
        if file:
            df = pd.read_csv(file)
            df["Prediction"] = model.predict(vectorizer.transform(df["ticket"]))
            st.dataframe(df)

    # ------------------ TAB 3: User Management (Admin only) ------------------
    with tab3:
        if st.session_state["role"] == "admin":
            user_management()
        else:
            st.error("Access denied — Admins only.")


# ------------------ ROUTER ------------------
if "logged_in" not in st.session_state:
    login_screen()
else:
    main_app()
