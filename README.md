# 🎫 AI Ticket Classifier  
### **Smart IT Support Automation Using Machine Learning + Streamlit + SQLite**

An end-to-end IT support automation platform that classifies tickets using Machine Learning, assists employees with troubleshooting, logs ticket insights, manages users securely, and provides a complete admin dashboard — all inside a single Streamlit web application.

This project demonstrates skills in **Machine Learning**, **NLP**, **Python**, **Streamlit**, **SQLite databases**, **full-stack UI development**, **authentication**, and **AI assistant integration (OpenAI)**.

---

## 🧠 Features Overview

### 🔍 **1. ML-Powered Ticket Classification**
- Predicts IT issue categories:  
  *Outlook, Network, Printer, AD, Hardware, Firewall, Security, MDM, Email*
- Confidence scoring  
- Real-time predictions  
- Model: **TF-IDF + RandomForestClassifier**

---

### 🤖 **2. AI Help Assistant**

#### 🧠 ML Helper *(Offline & Free)*
- Suggests automated troubleshooting steps  
- Works without internet  
- Ideal for secure enterprise environments  

#### 💬 ChatGPT Helper *(Optional)*
- Conversational AI assistant  
- Uses OpenAI API  
- Auto-disabled if no API key is provided  

---

### 📁 **3. Bulk CSV Classification**
- Upload CSV  
- Automatically classify hundreds of tickets  
- Download result CSV  

---

### 👥 **4. User Management (SQLite + bcrypt)**
- Add users  
- Delete users  
- Reset passwords  
- Role-based access control (Admin / User)  
- Passwords securely hashed  

---

### 📊 **5. Admin Analytics Dashboard**
- Ticket category distribution  
- Confidence charts  
- Severity heatmaps  
- User ticket insights  
- Trends over time  

---

### 🔎 **6. User Insights Engine**
- User ticket history  
- Detect repeated issues  
- Severity-level tracking  
- Similar-ticket suggestions  

---

## 🏗 **Project Architecture**

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

## 🛠 **Tech Stack**

- **Python**
- **Streamlit**
- **SQLite**
- **bcrypt** (authentication)
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
bash
Copy code
pip install -r requirements.txt
3️⃣ Run the Application
bash
Copy code
streamlit run app.py
4️⃣ (Optional) Enable ChatGPT Assistant
Create this file:

bash
Copy code
.streamlit/secrets.toml
Add the following:

toml
Copy code
OPENAI_API_KEY = "your-key"
🔐 Default Admin Login
makefile
Copy code
username: admin
password: sahil123
🌐 Deployment (Streamlit Cloud Ready)
This app is designed for deployment on Streamlit Cloud, offering:

🔒 Secure secrets management

⚡ Auto-redeployment on every Git push

🌍 Public or private access

🤖 ChatGPT-ready integration

📸 Screenshots
(Add screenshots here for better presentation)

Example:

scss
Copy code
![Home](screenshots/home.png)
![AI Assistant](screenshots/ai_helper.png)
![Admin Dashboard](screenshots/dashboard.png)
📎 GitHub Repository
https://github.com/SahilMujawar03/ai-ticket-classifier

💼 About This Project
This project replicates real-world IT helpdesk workflows using automation and AI:

Ticket classification

User account management

Predictive analytics

AI assistant troubleshooting

Severity scoring

Repeated ticket detection

It is designed as a portfolio-grade project for showcasing engineering and AI skills to employers.

⭐ Author
Sahil Mujawar
Aspiring AI Engineer | Python Developer | IT Automation Enthusiast

🤝 Contributions
Pull requests and suggestions are always welcome!

yaml
Copy code

---

# 🟢 Your README is now 100% professional and ready for GitHub.

### Do you want the next step?

I can now create:

✅ **A perfect LinkedIn post**  
✅ **A resume project section for your CV**  
✅ **A GitHub repository banner image**  

Just tell me:

👉 **“Create my LinkedIn post”**  
or  
👉 **“Write resume points for this project”**






