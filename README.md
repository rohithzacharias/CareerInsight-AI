<div align="center">

# 🚀 CareerInsight AI

### AI-Powered Student Placement Prediction & Career Analytics Platform

<p>
  <img src="https://img.shields.io/badge/Python-3.x-3776AB?style=for-the-badge&logo=python&logoColor=white">
  <img src="https://img.shields.io/badge/Machine%20Learning-Scikit--Learn-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white">
  <img src="https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white">
  <img src="https://img.shields.io/badge/Power%20BI-Analytics-F2C811?style=for-the-badge&logo=powerbi&logoColor=black">
  <img src="https://img.shields.io/badge/Pandas-Data%20Analysis-150458?style=for-the-badge&logo=pandas&logoColor=white">
</p>

<p>
  <img src="https://img.shields.io/badge/Status-Completed-2ea44f?style=for-the-badge">
  <img src="https://img.shields.io/github/license/rohithzacharias/CareerInsight-AI?style=for-the-badge">
  <img src="https://img.shields.io/github/last-commit/rohithzacharias/CareerInsight-AI?style=for-the-badge">
</p>

**CareerInsight AI** is an interactive machine learning and analytics platform designed to help students understand their placement probability, career opportunities, salary trends, and skill gaps.

</div>

---

## 🌟 Project Highlights

| 🧠 Placement AI | 🎯 Career Matching | 💰 Salary Analytics |
|---|---|---|
| Predict student placement probability | Identify suitable career roles | Explore job salary trends |

| 📚 Skill Gap Analysis | 📊 Power BI | 🖥️ Streamlit |
|---|---|---|
| Find skills that need improvement | Interactive business analytics | Complete interactive web application |

---

# 🎯 What Problem Does CareerInsight AI Solve?

Students often know their academic scores but may not know:

- ❓ What is my current placement probability?
- ❓ Which career role suits my capabilities?
- ❓ What skills should I learn next?
- ❓ How strong is my overall profile?
- ❓ What salary range can I expect for different roles?
- ❓ How can I improve my placement readiness?

**CareerInsight AI brings these insights together in one platform.**

---

# 🧠 Machine Learning

The placement prediction component uses the **Indian Student Placement Dataset 2025**.

### Input Features

The model considers student attributes including:

- Gender
- Age
- Degree
- Branch
- CGPA
- Backlogs
- Internships
- Certifications
- Coding Skills
- Communication Skills
- Aptitude Score
- Projects

### Machine Learning Pipeline

```mermaid
flowchart LR

A["📂 Raw Placement Dataset"] --> B["🔍 Data Inspection"]
B --> C["🧹 Data Cleaning"]
C --> D["📊 Exploratory Data Analysis"]
D --> E["⚙️ Feature Engineering"]
E --> F["📁 Processed Dataset"]
F --> G["🤖 Model Training"]
G --> H["📈 Model Evaluation"]
H --> I["💾 Trained ML Model"]
I --> J["🖥️ Streamlit Application"]
J --> K["🎯 Placement Probability"]
