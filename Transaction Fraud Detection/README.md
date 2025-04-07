# 💳 Fraud Detection Using Machine Learning

This project is focused on detecting fraudulent financial transactions using classical machine learning techniques. The solution pipeline includes **data preprocessing**, **model training**, **evaluation**, and a **Streamlit dashboard** to visualize results and test new transactions.

> 📌 The **main analysis and implementation** was carried out in a detailed `fruad_detection_final.ipynb` notebook using **Google Colab**.

---
## 📊 Dataset Description
The dataset contains 6,362,620 transaction records with 11 features, simulating mobile money transfers and fraud behavior. Key columns include:

  * type: Transaction type (PAYMENT, TRANSFER, CASH_OUT, etc.)

  * amount: Transaction amount

  * oldbalanceOrg / newbalanceOrig: Sender’s balance before and after the transaction

  * oldbalanceDest / newbalanceDest: Receiver’s balance before and after

  * isFraud: Indicates whether the transaction is fraudulent (1) or not (0)

There are no missing values, and the dataset is ideal for binary fraud classification tasks.

📁 **Dataset:** [Click here to download the Fraud Detection dataset](https://drive.google.com/uc?export=download&confirm=6gh6&id=1VNpyNkGxHdskfdTNRSjjyNa5qC9u0JyV)
---
## 📈 Project Overview

### ✅ Goals:
- Detect fraudulent transactions with high precision and recall
- Handle severe class imbalance using SMOTE
- Evaluate and compare different machine learning models
- Build a user-friendly Streamlit dashboard for interaction

### 🧠 Models Used:
- **Random Forest (with and without SMOTE)**
- **XGBoost (with and without SMOTE)**

---

## 🧪 Project Workflow

### 1. **Data Preprocessing**
- Missing values handling
- Outlier removal
- Label encoding of categorical features
- Feature scaling

### 2. **Exploratory Data Analysis**
- Class imbalance check
- Correlation matrix
- Distribution plots of transaction features

### 3. **Model Building & Training**
- Train-test split (80/20)
- Applied SMOTE on the training set
- Trained 4 different models:
  - Random Forest
  - Random Forest + SMOTE
  - XGBoost
  - XGBoost + SMOTE
- Evaluated using:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - ROC-AUC Score

### 4. **Evaluation & Visualizations**
- Confusion matrix
- ROC curve
- Feature importance plots

### 5. **Answers to Key Business Questions**
- Included in the final section of the notebook with insights and recommendations.

---

## 📊 Streamlit Dashboard

A lightweight **Streamlit app** is created just for presentation and interaction. It allows users to:
- Select one of the 4 trained models
- View evaluation metrics and charts
- Upload new transactions in CSV format and get fraud predictions
- Manually enter a single transaction and classify it
- View static feature importance

👉 [Live App (Coming Soon)](https://streamlit-app-link-placeholder.com)

---

## 📂 Project Structure
```
fraud-detection-deploy/
├── app.py # Streamlit app code
├── models/
│ ├── rf_model.pkl # Random Forest
│ ├── rf_smote_model.pkl # Random Forest + SMOTE
│ ├── xgb_model.pkl # XGBoost
│ └── xgb_smote_model.pkl # XGBoost + SMOTE
├── data/
│ ├── y_test.csv # True labels for test data
│ └── fraud_test_predictions.csv # Model predictions for test data
├── requirements.txt
└── README.md
```
---
---

## 🔍 Model Comparison Summary

| Model                | Precision (Class 1) | Recall (Class 1) | F1-Score (Class 1) | ROC-AUC | False Positives | False Negatives | Notes                                |
|---------------------|---------------------|------------------|--------------------|---------|------------------|------------------|--------------------------------------|
| Random Forest        | 0.9996              | 0.9963           | 0.9980             | 0.9982  | -                | -                | Very high precision & recall         |
| XGBoost              | 0.95                | 0.94             | 0.94               | 0.9996  | -                | -                | Slightly lower recall                |
| Random Forest + SMOTE| 0.91                | 1.00             | 0.95               | 0.9981  | 242              | 0                | Perfect recall, slight drop in precision |
| XGBoost + SMOTE      | 0.81                | 0.99             | 0.89               | 0.9960  | 582              | 19               | Lower precision, decent recall       |

---

## 🏆 Recommended Final Model: `Random Forest + SMOTE`

### ✅ Why this model?

- **Recall = 1.00** → Catches **all fraudulent transactions** (no false negatives).
- **F1-Score = 0.95** → Maintains a good balance between precision and recall.
- **ROC-AUC = 0.9981** → Excellent overall classification performance.
- **Interpretability** → Easier to explain and deploy compared to complex models like XGBoost.

> ⚠️ **Important Note:** In fraud detection, **recall is critical** — missing a fraud is much costlier than raising a false alarm.

💡 **Pro Tip:**  
If you prioritize **faster inference** or plan extensive **hyperparameter tuning**, you can explore XGBoost with SMOTE.  
But be aware — it has **lower precision and F1-score** than Random Forest.

---
## 🧠 Models Overview

All models were trained in a separate notebook using a large fraud transaction dataset. Below are the models used:

| Model            | SMOTE Applied | Notes                          |
|------------------|----------------|--------------------------------|
| Random Forest     | ❌              | Standard baseline               |
| Random Forest     | ✅              | Oversampling to fix imbalance   |
| XGBoost           | ❌              | Powerful gradient boosting      |
| XGBoost           | ✅              | Best recall and ROC-AUC         |

---

## 📊 Dashboard Features

- 🔘 Model Selection Dropdown
- 📏 Key Metrics Display: Accuracy, Precision, Recall, F1-Score, ROC-AUC
- 📉 Visualizations: ROC Curve, Confusion Matrix
- ⭐ Feature Importance: Static plots
- 📁 Bulk Prediction: Upload CSV
- 🧍 Single Prediction: Manual input

---

## 📥 Run Locally

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/fraud-detection-deploy.git
   cd fraud-detection-deploy
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
## 🧾 Requirements
The requirements.txt includes key libraries like:

  * streamlit
  * pandas
  * numpy
  * scikit-learn
  * xgboost
  * matplotlib
  * seaborn
  * joblib
## 📌 Note
  * The .ipynb notebook used for data preprocessing, model building, evaluation, and answering project questions was developed in Google Colab but it is compatible with jupyter notebook as well.

## 📧 Contact
Yogesh Jadhav
👉 [LinkedIn](https://www.linkedin.com/in/yogesh-jadhav-60548020a/) | [Email](yj43221@gmail.com)

## 📝 License
This project is licensed under the MIT License.

---
