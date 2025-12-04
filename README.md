# 🎫 AI Ticket Classifier  
### Smart IT Support Automation Using Machine Learning + Streamlit + SQLite

An end-to-end IT support automation platform that classifies tickets using Machine Learning, assists employees with troubleshooting, logs ticket insights, manages users securely, and provides a complete admin dashboard — all inside a single Streamlit application.

---

## 🧠 Features Overview

### 🔍 1. ML-Powered Ticket Classification
- Predicts IT issue categories (Outlook, Network, Printer, AD, Hardware, Firewall, Security, MDM, Email)
- Confidence scoring
- Real-time predictions
- Model: **TF-IDF + RandomForestClassifier**

---

## 🤖 2. AI Help Assistant

### 🧠 ML Helper (Offline & Free)
- Automated troubleshooting suggestions  
- Works without internet  
- Privacy-friendly

### 💬 ChatGPT Helper (Optional)
- Conversational IT assistant  
- Uses OpenAI API  
- Auto-disabled if no key is configured

---

## 📁 3. Bulk CSV Classification
- Upload CSV  
- Classify hundreds of tickets  
- Download results  

---

## 👥 4. User Management (SQLite + bcrypt)
- Add / delete users  
- Reset passwords  
- Role-based access  
- Passwords hashed with bcrypt  

---

## 📊 5. Admin Analytics Dashboard
- Ticket category distribution  
- Confidence analytics  
- Severity heatmap  
- User insights  
- Ticket trends  

---

## 🔎 6. User Insights Engine
- User ticket history  
- Repeated issue detection  
- Severity-level tracking  
- Similar-ticket suggestions  

---

## 🏗 Project Architecture


---

## 🛠 Tech Stack

- **Python**
- **Streamlit**
- **SQLite**
- **bcrypt**
- **Pandas / NumPy**
- **Scikit-learn**
- **OpenAI API (optional)**

---

## 🚀 Installation

### 1️⃣ Clone the Repository
```bash
git clone https://github.com/SahilMujawar03/ai-ticket-classifier.git
cd ai-ticket-classifier

pip install -r requirements.txt

streamlit run app.py

OPENAI_API_KEY = "your-key"

username: admin
password: sahil123


![Home Page](screenshots/home.png)
![AI Helper](screenshots/ai_helper.png)
![Dashboard](screenshots/dashboard.png)
