# 🎫 AI Ticket Classifier  
### **Smart IT Support Automation Using Machine Learning + Streamlit + SQLite**

An end-to-end IT support automation platform that classifies tickets using Machine Learning, assists employees with troubleshooting, logs ticket insights, manages users securely, and provides a complete admin dashboard — all inside a single Streamlit web application.

This project demonstrates skills in **Machine Learning**, **NLP**, **Python**, **Streamlit**, **SQLite**, **full-stack UI development**, **authentication**, and **AI assistant integration (OpenAI)**.

---

## 🧠 Features Overview

### 🔍 **1. ML-Powered Ticket Classification**
- Predicts IT issue categories:  
  _Outlook, Network, Printer, AD, Hardware, Firewall, Security, MDM, Email_
- Confidence scoring  
- Real-time predictions  
- Model: **TF-IDF + RandomForestClassifier**

---

## 🤖 **2. AI Help Assistant**

### 🧠 ML Helper *(Offline & Free)*
- Suggests troubleshooting steps  
- Works without internet  
- Ideal for secure enterprise environments  

### 💬 ChatGPT Helper *(Optional)*
- Conversational IT assistant  
- Uses OpenAI API  
- Auto-disabled if no API key is provided  

---

## 📁 **3. Bulk CSV Classification**
Upload CSV → classify → download results.

---

## 👥 **4. User Management (SQLite + bcrypt)**
- Add users  
- Delete users  
- Reset passwords  
- Role-based access (admin/user)  
- Passwords hashed using bcrypt  

---

## 📊 **5. Admin Analytics Dashboard**
- Category distribution  
- Confidence analytics  
- Severity heatmap  
- User insights  
- Ticket trends  

---

## 🔎 **6. User Insights Engine**
- User ticket history  
- Repeated issue detection  
- Severity-level trends  
- Similar-ticket suggestions  

---

## 🏗 Project Architecture

ai-ticket-classifier/
│
├── app.py # Main Streamlit application
├── model.pkl # Trained ML model
├── vectorizer.pkl # TF-IDF vectorizer
├── tickets_200.csv # Training dataset
├── users.csv # Initial user accounts
├── requirements.txt # Dependencies
└── SQLite database created at runtime

yaml
Copy code

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

## 🚀 Installation & Running

### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/SahilMujawar03/ai-ticket-classifier.git
cd ai-ticket-classifier
2️⃣ Install Dependencies
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Application
bash
Copy code
streamlit run app.py
4️⃣ (Optional) Enable ChatGPT Assistant
Create:

bash
Copy code
.streamlit/secrets.toml
Add:

toml
Copy code
OPENAI_API_KEY = "your-key"
🔐 Default Admin Login
makefile
Copy code
username: admin
password: sahil123
🌐 Deployment (Streamlit Cloud Ready)
Secure API secret management

Auto deployment on every Git push

Public and private sharing

Cloud logs for debugging

📸 Screenshots
(Add screenshots here after deployment)

Example:

scss
Copy code
![Home Page](screenshots/home.png)
![AI Helper](screenshots/ai_helper.png)
![Dashboard](screenshots/dashboard.png)
📎 GitHub Repository
https://github.com/SahilMujawar03/ai-ticket-classifier

💼 About This Project
This project replicates real enterprise IT workflows using automation and AI:

Ticket classification

User account management

Predictive analytics

AI troubleshooting

Severity scoring

Repeated issue analysis

A portfolio-grade application demonstrating practical ML + full-stack development.

⭐ Author
Sahil Mujawar
Aspiring AI Engineer | Python Developer | IT Automation Enthusiast
