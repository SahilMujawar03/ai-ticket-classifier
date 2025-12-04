# 🎫 AI Ticket Classifier – Smart IT Support Automation

An intelligent IT support assistant built with **Machine Learning, SQLite, and Streamlit**, designed to automatically classify IT support tickets, help employees troubleshoot issues, manage users, and analyze IT trends — all inside one clean web application.

---

## 🚀 Features

### 🎟 IT Ticket Classification (ML Model)
- Predicts issue categories (Outlook, Network, Printer, AD, Hardware, Firewall, MDM, Security)
- Confidence scoring
- Fast and offline ML model (TF-IDF + Random Forest)

---

### 🤖 AI Help Assistant

#### 1️⃣ ML Helper (Free, Offline)
- Suggests automated troubleshooting steps  
- Works without internet  
- Secure & company-friendly  

#### 2️⃣ ChatGPT Helper (Optional)
- Chat-style IT support assistant  
- Requires an OpenAI API key  
- Automatically disabled if no key is provided  

---

### 📊 Admin Dashboard
- Ticket category trends  
- Confidence metrics  
- Issue heatmaps  
- Ticket history timeline  

---

### 👥 User Management
- Add users  
- Delete users  
- Reset passwords  
- Secure login using SQLite + bcrypt  

---

### 📁 Bulk CSV Classifier
- Upload CSV  
- Automatically classify all tickets at once  
- Download results instantly  

---

### 🔍 User Insights Engine
- User-specific ticket history  
- Detect repeated issues  
- Severity and trend analytics  

---

## 🗄 SQLite Database Structure

### 📌 users table
| column    | description                  |
|-----------|------------------------------|
| username  | Primary key                  |
| password  | bcrypt hashed password       |
| role      | admin / user                 |

### 📌 tickets table
| column      | description                       |
|-------------|-----------------------------------|
| id          | Unique ticket ID                  |
| timestamp   | Date & time of prediction         |
| ticket      | Original ticket text              |
| prediction  | ML predicted category             |
| confidence  | ML confidence score               |
| severity    | Auto-assigned severity score      |
| user        | Username who submitted it         |

---

## 🏗 Project Structure

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

### 1️⃣ Clone the repository
```bash
git clone https://github.com/SahilMujawar03/ai-ticket-classifier.git
cd ai-ticket-classifier

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Run the application
streamlit run app.py

4️⃣ (Optional) Enable ChatGPT Helper

Create:

.streamlit/secrets.toml


Add:

OPENAI_API_KEY = "your-key"

🔐 Default Admin Login
username: admin
password: sahil123

🌐 Deployment

The app is ready for Streamlit Cloud, offering:

Secure secrets

Auto-scaling

Public or private access

ChatGPT enterprise-style integration

🧾 Screenshots

(Add screenshots later)

📎 GitHub Repository

https://github.com/SahilMujawar03/ai-ticket-classifier

⭐ Author

Developed by Sahil Mujawar


---

# ✅ **Your README is now fully ready.**

## Next Step:  
Do you want me to also generate a **LinkedIn post** to showcase your project professionally so recruiters notice it?

Example:

✔ Professional  
✔ Includes project highlights  
✔ Recruiter-friendly  
✔ Gets attention  

Just say **"Create my LinkedIn post"**.
