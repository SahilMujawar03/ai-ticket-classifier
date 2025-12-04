# AI Ticket Classifier – Smart IT Support Automation

An intelligent IT support assistant built with **Machine Learning, SQLite, and Streamlit**, designed to automatically classify support tickets, assist employees with troubleshooting, and provide admin analytics — all in a clean web app.

---

## 🚀 Features

### 🎟 IT Ticket Classification (ML Model)
- Predicts category (Outlook, Network, Printer, AD, Hardware…)
- Shows confidence score
- Uses TF-IDF + Random Forest  
- Fast offline model loading

---

### 🤖 AI Help Assistant

#### 1️⃣ ML Helper (Free & Offline)
- Suggests troubleshooting steps  
- Works without internet or API keys  
- Provides similar ticket history  

#### 2️⃣ ChatGPT Helper (Optional)
- Uses OpenAI GPT if API key is added  
- Disabled safely if no key is configured  
- Designed for enterprise use when needed

---

### 📊 Admin Dashboard
- Ticket trends  
- Top categories  
- Average confidence  
- Severity distribution  
- Tickets over time  

---

### 🧑‍💼 User Management
- Add, delete, and reset user passwords  
- Secure authentication using **bcrypt**  
- SQLite-based user database  

---

### 📁 Bulk CSV Ticket Classification
- Upload CSV  
- Automatically classify multiple tickets  
- Download results with predictions  

---

### 🔍 User Insights
- View user-specific ticket history  
- Detect repeated issues  
- Severity analysis  
- Similar ticket detection engine  

---

## 🗄 SQLite Database Structure

### `users` table
- username (PK)  
- password (hashed)  
- role  

### `tickets` table
- id  
- timestamp  
- ticket text  
- prediction  
- confidence  
- severity  
- username  

---

## 🧠 Machine Learning Model
- TF-IDF Vectorizer  
- RandomForestClassifier  
- Trained on 200+ real-world IT support tickets  
- Covers:  
  - Outlook issues  
  - Network  
  - Printer  
  - AD / Credentials  
  - Hardware  
  - Security  
  - MDM  
  - Firewall  

---

## 🏗 Architecture

Streamlit Web App
│
├── Authentication (SQLite users)
├── ML Model (RandomForest + TF-IDF)
├── Ticket Logging (SQLite)
├── AI Help Assistant (ML/ChatGPT optional)
├── Admin Dashboard
├── Bulk CSV Classifier
└── User Insights Engine

---

## 🛠 Installation

### 1. Clone the repository
git clone https://github.com/SahilMujawar03/ai-ticket-classifier.git
cd ai-ticket-classifier


### 2. Install dependencies
pip install -r requirements.txt


### 3. Run the application
streamlit run app.py


### 4. (Optional) Add OpenAI API Key  
To enable ChatGPT-based assistant:
Create `.streamlit/secrets.toml` and add:
OPENAI_API_KEY = "your-key"


---

## 🔐 Default Admin Login

username: admin
password: admin123


---

## 🌐 Deployment  
This app runs on **Streamlit Cloud**, offering:

- Secure encrypted secrets  
- Public or private sharing  
- Auto-redeployment on push  
- Enterprise-ready ChatGPT assistant  

---

## 🧾 Screenshots  
(Add screenshots here)

---

## 📎 GitHub Repository  
https://github.com/SahilMujawar03/ai-ticket-classifier

---

## ⭐ Author  
Developed by **Sahil Mujawar**

---
