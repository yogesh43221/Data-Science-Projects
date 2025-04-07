import streamlit as st
import pandas as pd
import pickle
import os
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, RocCurveDisplay
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

@st.cache_data
def load_predictions():
    return pd.read_csv("data/fraud_test_predictions.csv")

predictions_df = load_predictions()
y_test = predictions_df["Actual_isFraud"]
y_pred = predictions_df["Predicted_isFraud"]
y_score = predictions_df["Fraud_Probability"]

# Model selection only used for custom predictions
model_option = st.sidebar.selectbox(
    "Select a Model for Custom Prediction",
    [
        "Random Forest",
        "XGBoost",
        "Random Forest with SMOTE",
        "XGBoost with SMOTE"
    ]
)

model_files = {
    "Random Forest": "models/final_rf_model.pkl",
    "XGBoost": "models/final_xgb_model.pkl",
    "Random Forest with SMOTE": "models/final_rf_smote_model.pkl",
    "XGBoost with SMOTE": "models/final_xgb_smote_model.pkl"
}

@st.cache_resource
def load_model(model_path):
    with open(model_path, "rb") as f:
        return pickle.load(f)

model = load_model(model_files[model_option])

# Title
st.title("💳 Fraud Detection Model Dashboard")
st.subheader(f"📊 Model Evaluation (Loaded from CSV)")

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots()
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Not Fraud", "Fraud"], yticklabels=["Not Fraud", "Fraud"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
st.pyplot(fig)

# Classification Report
report = classification_report(y_test, y_pred, output_dict=True)
st.dataframe(pd.DataFrame(report).transpose())

# ROC-AUC Score
roc_auc = roc_auc_score(y_test, y_score)
st.metric(label="ROC-AUC Score", value=round(roc_auc, 5))

# ROC Curve
st.subheader("📈 ROC Curve")
fig_roc, ax_roc = plt.subplots()
RocCurveDisplay.from_predictions(y_test, y_score, ax=ax_roc)
st.pyplot(fig_roc)

# Custom Prediction
st.subheader(f"🧪 Try Prediction using {model_option}")

feature_list = [
    "amount", "oldbalanceOrg", "newbalanceOrig", 
    "oldbalanceDest", "newbalanceDest", 
    "type_CASH_OUT", "type_DEBIT", "type_PAYMENT", "type_TRANSFER", 
    "errorBalanceOrig", "errorBalanceDest"
]

input_data = {}
cols = st.columns(3)
for i, feature in enumerate(feature_list):
    with cols[i % 3]:
        input_data[feature] = st.number_input(feature, value=0.0, format="%.2f")

if st.button("Predict Transaction"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    prob = model.predict_proba(input_df)[0][1] if hasattr(model, "predict_proba") else None
    st.success("Prediction: **FRAUD** 🚨" if prediction == 1 else "Prediction: **NOT FRAUD** ✅")
    if prob is not None:
        st.info(f"Fraud Probability: {prob:.4f}")

# Feature Importance
st.subheader("📌 Feature Importance")
if hasattr(model, "feature_importances_"):
    importances = pd.Series(model.feature_importances_, index=feature_list)
    st.bar_chart(importances.sort_values(ascending=False))
else:
    st.info("Feature importances not available for this model.")

# Upload New Data for Bulk Prediction
st.subheader("📂 Upload CSV for Bulk Prediction")
uploaded_file = st.file_uploader("Upload transaction data CSV", type="csv")
    # Notes for user about CSV format
st.markdown("""
    **📝 Note: CSV file must follow this format:**
    - Must contain **exactly these columns**:
        - `amount`, `oldbalanceOrg`, `newbalanceOrig`, `oldbalanceDest`, `newbalanceDest`,
        - `type_CASH_OUT`, `type_DEBIT`, `type_PAYMENT`, `type_TRANSFER`,
        - `errorBalanceOrig`, `errorBalanceDest`
    - All values should be **numeric** (float or integer).
    - **No extra or missing columns**.
    - **Column names must match exactly** (case-sensitive, no typos).
    - Example:
    ```csv
    amount,oldbalanceOrg,newbalanceOrig,oldbalanceDest,newbalanceDest,type_CASH_OUT,type_DEBIT,type_PAYMENT,type_TRANSFER,errorBalanceOrig,errorBalanceDest
    10000,50000,40000,100000,110000,1,0,0,0,0,0
    2500,7000,4500,2000,4500,0,1,0,0,0,0
    ```
    """)

if uploaded_file is not None:
    try:
        input_csv_df = pd.read_csv(uploaded_file)
        if all(col in input_csv_df.columns for col in feature_list):
            preds = model.predict(input_csv_df)
            probs = model.predict_proba(input_csv_df)[:, 1] if hasattr(model, "predict_proba") else [None] * len(preds)
            input_csv_df["Predicted_isFraud"] = preds
            input_csv_df["Fraud_Probability"] = probs
            st.write("### 📋 Prediction Results:")
            st.dataframe(input_csv_df[["Predicted_isFraud", "Fraud_Probability"] + feature_list])
            csv_download = input_csv_df.to_csv(index=False).encode("utf-8")
            st.download_button("📥 Download Predictions as CSV", data=csv_download, file_name="predictions.csv", mime="text/csv")
        else:
            st.error("❌ CSV missing required feature columns.")
    except Exception as e:
        st.error(f"Error processing file: {e}")
