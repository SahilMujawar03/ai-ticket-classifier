<p align="center">
  <img src="https://raw.githubusercontent.com/SahilMujawar03/ai-ticket-classifier/main/assets/AI%20Ticket%20Classifier%20Banner.png" width="100%" alt="AI Ticket Classifier Banner"/>
</p>

# 🎫 AI Ticket Classifier  
### Smart IT Support Automation Using Machine Learning + Streamlit + SQLite

An end-to-end IT support automation platform that classifies IT tickets using Machine Learning, assists employees with troubleshooting, logs ticket insights, manages users securely, and provides an admin dashboard — all inside a single Streamlit application.

---

## Features Overview

## 1. ML-Powered Ticket Classification
- Predicts IT issue categories: Outlook, Network, Printer, AD, Hardware, Firewall, Security, MDM, Email  
- Shows confidence score  
- Real-time predictions  
- Model used: TF-IDF + RandomForestClassifier

---

## 2. AI Help Assistant

### ML Helper (Offline)
- Suggests troubleshooting steps  
- Works without internet  
- Free and secure  

### ChatGPT Helper (Optional)
- Conversational IT assistant  
- Uses OpenAI API  
- Disabled automatically if no key is set  

---

## 3. Bulk CSV Classification
- Upload a CSV file  
- Classify hundreds of tickets instantly  
- Download results  

---

## 4. User Management (SQLite + bcrypt)
- Add and remove users  
- Reset passwords  
- Role-based access (Admin/User)  
- Secure password hashing  

---

## 5. Admin Analytics Dashboard
- Ticket category distribution  
- Confidence analytics  
- Severity heatmaps  
- User activity insights  

---

## 6. User Insights Engine
- User ticket history  
- Repeated issue detection  
- Severity trend tracking  
- Related-ticket suggestions  

---

## Project Architecture

ai-ticket-classifier/
|
|-- app.py                 (Main Streamlit application)
|-- model.pkl              (Trained ML model)
|-- vectorizer.pkl         (TF-IDF vectorizer)
|-- tickets_200.csv        (Training dataset)
|-- users.csv              (Initial user data)
|-- requirements.txt       (Dependencies)
`-- SQLite database created at runtime

---

## Tech Stack
- Python  
- Streamlit  
- SQLite  
- bcrypt  
- Pandas / NumPy  
- Scikit-learn  
- OpenAI API (optional)

---

## Installation

### 1. Clone the Repository
git clone https://github.com/SahilMujawar03/ai-ticket-classifier.git
cd ai-ticket-classifier

### 2. Install Dependencies
pip install -r requirements.txt

### 3. Run the Application
streamlit run app.py

---

## Optional: Enable ChatGPT Assistant

Create this file:
.streamlit/secrets.toml

Add:
OPENAI_API_KEY = "your-key"

---

## Default Admin Login
username: admin  
password: sahil123  

---

## Deployment (Streamlit Cloud Ready)
- Secure secrets  
- Auto redeployment  
- Public/private sharing  
- Cloud logs  

---

## Screenshots
(Add screenshots after deployment)

![Home Page](screenshots/home.png)  
![AI Helper](screenshots/ai_helper.png)  
![Dashboard](screenshots/dashboard.png)

---

## GitHub Repository
https://github.com/SahilMujawar03/ai-ticket-classifier

---

## Author
Sahil Mujawar  
Aspiring AI Engineer | Python Developer | IT Automation Enthusiast
