# 🛡️ A Hybrid Machine Learning Framework for Geospatial Cyber Crime Prediction and Demographic Pattern Analysis

<p align="center">
  <img src="https://img.shields.io/badge/React.js-Frontend-blue?style=for-the-badge&logo=react">
  <img src="https://img.shields.io/badge/Flask-Backend-black?style=for-the-badge&logo=flask">
  <img src="https://img.shields.io/badge/Python-ML-yellow?style=for-the-badge&logo=python">
  <img src="https://img.shields.io/badge/Machine%20Learning-Cybercrime%20Prediction-red?style=for-the-badge">
  <img src="https://img.shields.io/badge/Final%20Year%20Project-2025--26-success?style=for-the-badge">
</p>

---

---

# 🌐 Live Demo

🔗 **[Access the Live Application](https://final-year-project-cyber-crime-analysi.onrender.com/)**

> ⚠️ Note: This app is hosted on Render's free tier, so it may take **30–60 seconds** to spin up on the first load if it's been idle. Please be patient — it'll load fine after that.

---

## 📖 Overview

Cybercrime has become one of the fastest-growing threats in the digital era, involving activities such as:

- Computer Fraud
- Credit Card Fraud
- Cyberstalking
- Telephone Threats
- Identity Theft
- Online Financial Fraud

Traditional crime analysis approaches often fail to capture the complex spatial and temporal relationships present in cybercrime data.

This project proposes a **Hybrid Machine Learning Framework** that combines:

- Machine Learning Classification
- Future Hotspot Prediction
- Geospatial Analysis
- Temporal Analysis
- Interactive Visualization
- Demographic Pattern Exploration

The framework uses the **Chicago Crime Dataset** filtered for cyber-related crimes and provides an intelligent dashboard for crime analytics and predictive decision-making.

---

# 🎯 Objectives

- Analyze cybercrime patterns using data analytics.
- Identify geographical cybercrime hotspots.
- Perform district-wise crime analysis.
- Analyze time-based cybercrime trends.
- Compare multiple machine learning algorithms.
- Forecast future cybercrime hotspots.
- Support data-driven decision making.
- Provide an interactive analytical dashboard.

---

# 🏆 Recognition

This research work was presented at the **8th Regional Science & Technology Congress (2025–26)** organized by the **Department of Science and Technology and Biotechnology, Government of West Bengal**.

🏅 **Outstanding Paper Award – Engineering & Technology Discipline**

---

# 👨‍💻 Project Team

### Department of Computer Science & Engineering
### Siliguri Institute of Technology

| Name | Roll No |
|--------|----------|
| Arnav Biswas | 11900122143 |
| Ayandeep Roy | 11900122117 |
| Suryashis Banerjee | 11900122158 |
| Rimi Dutta | 11900122166 |

### Project Guide

**Dr. Anupam Mukherjee**

---

# 🏗️ System Architecture

```text
Chicago Crime Dataset
          │
          ▼
 Data Preprocessing
          │
          ▼
 Feature Engineering
          │
          ▼
 ┌─────────────────────────┐
 │ Exploratory Analysis    │
 └─────────────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │ Machine Learning Models │
 └─────────────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │ Geospatial Analytics    │
 └─────────────────────────┘
          │
          ▼
 ┌─────────────────────────┐
 │ Future Hotspot Forecast │
 └─────────────────────────┘
          │
          ▼
 React + Flask Dashboard
```

---

# 📊 Dataset Information

### Dataset Source

Chicago Open Data Portal

### Crime Categories

- COMPUTER FRAUD
- CREDIT CARD FRAUD
- CYBERSTALKING
- TELEPHONE THREAT
- VIOLATION GPS MONITORING DEVICE
- FALSE/STOLEN/ALTERED TRP

### Dataset Features

#### Temporal Features

- Year
- Month
- Day
- Hour
- Weekday

#### Spatial Features

- District
- Ward
- Beat
- Latitude
- Longitude

---

# ⚙️ Feature Engineering

The following features were generated and transformed:

- District Encoding
- Month Extraction
- Hour Extraction
- Weekday Extraction
- Weekend Flag
- Latitude Normalization
- Longitude Normalization
- Temporal Pattern Features
- Spatial Pattern Features

---

# 🤖 Machine Learning Models

## Classification Models

The framework compares multiple supervised learning algorithms:

| Model | Purpose |
|---------|---------|
| Naive Bayes | Classification |
| Logistic Regression | Classification |
| Support Vector Machine (SVM) | Classification |
| Decision Tree | Classification |
| Voting Ensemble Classifier | Classification |

---

## Future Hotspot Prediction Models

The forecasting module includes:

| Model |
|---------|
| Linear Regression |
| Support Vector Regression (SVR) |
| Decision Tree Regressor |
| Voting Ensemble Regressor |

---

# 📈 Evaluation Metrics

### Classification Metrics

- Accuracy
- Precision
- Recall
- F1 Score
- Confusion Matrix

### Regression Metrics

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)
- R² Score

---

# 🌍 Geospatial Analysis

The project incorporates advanced geospatial analytics:

### Features

- Cybercrime Heatmaps
- Hotspot Identification
- District-wise Visualization
- Spatial Density Analysis
- Location Intelligence

### Benefits

- High-risk region identification
- Resource allocation support
- Crime concentration analysis
- Geographical trend detection

---

# 📊 Dashboard Modules

## 🏠 Home

Project overview and navigation.

## 📂 Dataset Overview

- Dataset statistics
- Crime distribution
- Data summaries

## 📈 Analysis Module

- District-wise analysis
- Time-wise analysis
- Crime category distribution
- Interactive visualizations

## 🤖 Machine Learning Module

- Model training
- Model comparison
- Performance evaluation

## 🔮 Future Hotspot Prediction

- Crime forecasting
- Risk assessment
- Future hotspot visualization

## ℹ️ About

Project details and documentation.

---

# 🛠️ Technology Stack

## Frontend

- React.js
- JavaScript
- HTML5
- CSS3
- Bootstrap
- Plotly.js
- Leaflet.js

## Backend

- Flask
- Python

## Machine Learning

- Scikit-Learn
- NumPy
- Pandas

## Visualization

- Plotly
- Matplotlib
- Leaflet

---

# 🚀 Installation

## Clone Repository

```bash
git clone https://github.com/yourusername/cybercrime-prediction.git

cd cybercrime-prediction
```

## Project Structure

```text
backend/     Flask API, preprocessing, and ML model code
frontend/    Dashboard HTML, CSS, and browser JavaScript
assets/      Team images, certificate image, and generated Folium map
react_api.py Root launcher kept for `python react_api.py`
```

## Run Locally

```bash
pip install -r requirements.txt
python react_api.py
```

Then open `http://127.0.0.1:5000/`.

💡 Prefer not to install anything? Try the [live demo](https://final-year-project-cyber-crime-analysi.onrender.com/) instead.

# 📷 Project Features

✅ Interactive Dashboard

✅ Geospatial Heatmaps

✅ Cybercrime Classification

✅ District-wise Analysis

✅ Time-wise Analysis

✅ Future Hotspot Prediction

✅ Ensemble Learning

✅ React–Flask Integration

✅ Machine Learning Model Comparison

✅ Crime Forecasting

---

# 🔬 Future Scope

Future enhancements may include:

- Explainable AI (XAI)
- Real-Time Crime Monitoring
- NLP-based Threat Detection
- Social Media Analytics
- Dark Web Monitoring
- Cloud Deployment
- Mobile Dashboard
- Smart City Integration
- AI-powered Alert Systems
- Multi-City Comparative Analysis

---

# 📚 References

[1] Veena K., Meena K., Teekaraman Y., Kuppusamy R., Radhakrishnan A.
*C SVM Classification and KNN Techniques for Cyber Crime Detection*, 2022.

[2] Sahaya Sheela M., Hemanand D., Ranadheer Reddy V.
*Cyber Security System Based on Machine Learning Using Logistic Decision Support Vector*, 2023.

[3] Yeboah-Ofori A.
*Classification of Malware Attacks Using Machine Learning in Decision Tree*, 2020.

[4] Pandey H., Goyal R., Virmani D., Gupta C.
*Ensem_SLDR: Classification of Cybercrime using Ensemble Learning Technique*, 2022.

[5] Cao D. M. et al.
*Advanced Cybercrime Detection: A Comprehensive Study on Supervised and Unsupervised Machine Learning Approaches Using Real-world Datasets*, 2024.

[6] Panigrahy S.
*Geospatial Crime Analytics: A GIS-Based Approach Towards Prediction of Crime Hotspots*, 2021.

[7] Jawla A., Singh M., Hooda N.
*Crime Forecasting using Folium*, 2020.

[8] Ajagbe S. A., Oladipupo M. A., Balogun E. O.
*Crime Belt Monitoring via Data Visualization: A Case Study of Folium*, 2020.

[9] Ahishakiye E., Taremwa D., Omulo E. O., Niyonzima I.
*Crime Prediction Using Decision Tree (J48) Classification Algorithm*, 2017.

---



© This project was developed for academic and research purposes as part of the Bachelor of Technology (B.Tech) degree requirement at Siliguri Institute of Technology.

