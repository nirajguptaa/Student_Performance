# Student Performance Prediction System

A Machine Learning based web application that predicts whether a student will **PASS or FAIL** based on academic features using classification models. The project uses **Scikit-learn** for model training and **Flask** for real-time predictions through a web interface.

---

##  Features

- Predicts student performance using ML classification
- Supports multiple ML models:
  - Logistic Regression
  - Random Forest
- Data preprocessing with:
  - Feature Scaling (StandardScaler)
  - Train–Test Split
- Model evaluation using:
  - Accuracy Score
  - Confusion Matrix
  - Classification Report
- Deployed using Flask for real-time predictions
- Trained model and scaler saved using `joblib`

---

## 🛠️ Tech Stack

- **Programming Language:** Python  
- **Machine Learning:** Scikit-learn  
- **Web Framework:** Flask  
- **Libraries:** Pandas, NumPy, Joblib  
- **Tools:** VS Code, Git  

---

## ⚙️ How It Works

1. **Data Preparation**
   - Input features (e.g. study hours, marks)
   - Target variable (0 = Fail, 1 = Pass)

2. **Preprocessing**
   - Features are scaled using **StandardScaler**
   - Data is split into **training and testing sets**

3. **Model Training**
   - Logistic Regression and Random Forest models are trained
   - Models are evaluated using:
     - Accuracy
     - Confusion Matrix
     - Classification Report

4. **Model Deployment**
   - Best performing model is saved as `model.pkl`
   - Scaler is saved as `scaler.pkl`
   - Flask app loads the model and makes real-time predictions

---

## ▶️ How to Run the Project

### 1️⃣ Install Dependencies
```bash
pip install pandas numpy scikit-learn flask joblib
```

### 2️⃣ Train the Model

```bash
python train_model.py
```

### 3️⃣ Run the Web App

```bash
python app.py
```
### 4️⃣ Open in Browser

```bash
http://127.0.0.1:5000
```
---

### Model Evaluation Example
```
Accuracy: 1.0

Confusion Matrix:
[[1 0]
 [0 1]]

Classification Report:
precision    recall  f1-score   support
0       1.00      1.00      1.00
1       1.00      1.00      1.00
---
✅ Prediction Output
	•	1 → PASS
	•	0 → FAIL

The Flask web interface allows users to enter input values and get instant predictions.

💡 Key Learnings
	•	End-to-end ML pipeline implementation
	•	Data preprocessing and feature scaling
	•	Model evaluation and selection
	•	Model serialization using joblib
	•	Real-time deployment using Flask REST API
```





## 🧑‍💻 Developer Information
- **Developer:** Niraj Kumar Gupta
- **Project Type:** AI/ML & Generative AI Enthusiast

