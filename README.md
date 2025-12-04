# 🎫 AI Ticket Classifier  
### **Smart IT Support Automation Using Machine Learning + Streamlit + SQLite**

An end-to-end IT support automation platform that classifies tickets using Machine Learning, assists employees with troubleshooting, logs ticket insights, manages users securely, and provides a complete admin dashboard — all inside a single Streamlit web application.

This project demonstrates skills in **Machine Learning**, **NLP**, **Python**, **Streamlit**, **SQLite databases**, **full-stack UI development**, **authentication**, and **AI assistant integration (OpenAI)**.

---

## 🧠 Features Overview

### 🔍 **1. ML-Powered Ticket Classification**
- Predicts IT issue categories:  
  _Outlook, Network, Printer, AD, Hardware, Firewall, Security, MDM, Email_
- Confidence scoring  
- Real-time predictions  
- Model: **TF-IDF + RandomForestClassifier**

---

### 🤖 **2. AI Help Assistant**
Two modes:

#### 🧠 ML Helper *(Offline & Free)*
- Suggests solutions using rule-based + similarity search  
- Runs locally, no API required  
- Ideal for enterprise secure environments  

#### 💬 ChatGPT Helper *(Optional)*
- Connects employees to an AI assistant  
- Uses your OpenAI API key  
- Auto-disabled if no key is provided  

---

### 📁 **3. Bulk CSV Classification**
Upload a CSV → instantly classify hundreds of tickets  
Download results as new CSV  

---

### 👥 **4. User Management (SQLite + bcrypt)**
- Add / remove users  
- Reset passwords  
- Role-based access (admin / user)  
- Passwords securely hashed  

---

### 📊 **5. Admin Analytics Dashboard**
- Ticket trends  
- Category distribution  
- Confidence chart  
- User-wise ticket insights  
- Severity heatmaps (Low / Medium / High)

---

### 🔎 **6. User Insights Engine**
- View a user’s ticket history  
- Detect repeat issues  
- Track severity levels  
- Similar-ticket recommendations  

---

## 🏗 **Project Architecture**

ai-ticket-classifier/
│
├── app.py # Main Streamlit app
├── model.pkl # ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── requirements.txt # Python dependencies
├── tickets_200.csv # Training dataset
├── users.csv # User data (initial)
├── .streamlit/secrets.toml (not in repo)
└── SQLite Database created at runtime


---

## 🛠 **Tech Stack**

- **Python**
- **Streamlit**
- **SQLite** (persistent ticket logging)
- **bcrypt** (secure authentication)
- **Pandas / NumPy**
- **Scikit-learn**
- **OpenAI API (optional)**

---

## 🚀 **Installation & Running**

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SahilMujawar03/ai-ticket-classifier.git
cd ai-ticket-classifier

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run the App
streamlit run app.py

4️⃣ (Optional) Enable ChatGPT Helper

Create file:

.streamlit/secrets.toml


Add:

OPENAI_API_KEY="your-key"

🔐 Default Admin Login
username: admin
password: sahil123

🌐 Deployment (Streamlit Cloud Ready)

This app is built for easy deployment on Streamlit Cloud, including:

🔒 Secure API key management

⚡ Auto-redeploy on push

🌍 Public / private sharing

📊 Cloud logs for debugging

📸 Screenshots

(Add screenshots here to make your project visually impressive)

Example placeholders:

![Home](screenshots/home.png)
![AI Assistant](screenshots/ai_helper.png)
![Admin Dashboard](screenshots/dashboard.png)

📎 GitHub Repository

https://github.com/SahilMujawar03/ai-ticket-classifier

💼 About the Project

This project was created to demonstrate real-world IT automation using machine learning and AI.
It replicates actual enterprise helpdesk workflows:

Automated ticket interpretation

Predictive analytics

User account management

Intelligent helpdesk assistant

Ticket severity detection

Repeat issue analysis

The app is designed to showcase strong engineering capability and is suitable for job portfolios and technical interviews.

⭐ Author

Sahil Mujawar
Aspiring AI Engineer | Python Developer | IT Automation Enthusiast

📬 Want to improve this project?

Pull requests and suggestions are welcome!


---

# ✅ Your README is now:
### ✔ Recruiter-friendly  
### ✔ Clean & professional  
### ✔ Portfolio-ready  
### ✔ Shows all your skills clearly  
### ✔ Makes your project look enterprise-grade  

---

# 🎉 Want the next upgrade?

I can also create:

### 👉 A **LinkedIn post** to showcase this project  
### 👉 A **GitHub project banner image**  
### 👉 A **resume bullet point summary for your CV**  

Just tell me:

**“Create my LinkedIn post”** or  
**“Create resume points for this project.”**
