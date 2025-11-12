import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Loan Eligibility Predictor", layout="wide")
st.title("🏦 Loan Eligibility Prediction App")

# Upload CSV
uploaded_file = st.file_uploader(r"C:\Users\teste\Downloads\train_u6lujuX_CVtuZ9i (1).csv", type=["csv"])

if uploaded_file:
    df = pd.read_csv(r"C:\Users\teste\Downloads\train_u6lujuX_CVtuZ9i (1).csv")
    st.subheader("📊 Raw Dataset")
    st.dataframe(df)

    # Drop Loan_ID
    if 'Loan_ID' in df.columns:
        df.drop('Loan_ID', axis=1, inplace=True)

    # Handle missing values
    for col in ['Gender', 'Married', 'Dependents', 'Self_Employed', 'Credit_History']:
        df[col].fillna(df[col].mode()[0], inplace=True)
    df['LoanAmount'].fillna(df['LoanAmount'].median(), inplace=True)
    df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].median(), inplace=True)

    # Visualizations
    st.subheader("📈 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Loan Status Count**")
        fig1, ax1 = plt.subplots()
        sns.countplot(x='Loan_Status', data=df, ax=ax1)
        st.pyplot(fig1)

    with col2:
        st.markdown("**Applicant Income by Loan Status**")
        fig2, ax2 = plt.subplots()
        sns.boxplot(x='Loan_Status', y='ApplicantIncome', data=df, ax=ax2)
        st.pyplot(fig2)

    st.markdown("**Correlation Heatmap**")
    fig3, ax3 = plt.subplots(figsize=(10, 5))
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap='coolwarm', ax=ax3)
    st.pyplot(fig3)

    # Encoding
    le = LabelEncoder()
    for col in df.select_dtypes('object'):
        df[col] = le.fit_transform(df[col])

    # Split data
    X = df.drop('Loan_Status', axis=1)
    y = df['Loan_Status']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    st.subheader("📉 Model Evaluation")
    st.write(f"**Accuracy:** {accuracy_score(y_test, y_pred):.2f}")
    st.text("Confusion Matrix:")
    st.write(confusion_matrix(y_test, y_pred))
    st.text("Classification Report:")
    st.text(classification_report(y_test, y_pred))

    # Prediction section
    st.subheader("🔮 Predict Loan Approval")

    with st.form("loan_form"):
        Gender = st.selectbox("Gender", ['Male', 'Female'])
        Married = st.selectbox("Married", ['Yes', 'No'])
        Dependents = st.selectbox("Dependents", ['0', '1', '2', '3+'])
        Education = st.selectbox("Education", ['Graduate', 'Not Graduate'])
        Self_Employed = st.selectbox("Self Employed", ['Yes', 'No'])
        ApplicantIncome = st.number_input("Applicant Income", min_value=0)
        CoapplicantIncome = st.number_input("Coapplicant Income", min_value=0.0)
        LoanAmount = st.number_input("Loan Amount", min_value=0.0)
        Loan_Amount_Term = st.number_input("Loan Amount Term", min_value=0.0)
        Credit_History = st.selectbox("Credit History", [1.0, 0.0])
        Property_Area = st.selectbox("Property Area", ['Urban', 'Rural', 'Semiurban'])

        submitted = st.form_submit_button("Predict")

        if submitted:
            input_data = pd.DataFrame({
                'Gender': [Gender],
                'Married': [Married],
                'Dependents': [Dependents],
                'Education': [Education],
                'Self_Employed': [Self_Employed],
                'ApplicantIncome': [ApplicantIncome],
                'CoapplicantIncome': [CoapplicantIncome],
                'LoanAmount': [LoanAmount],
                'Loan_Amount_Term': [Loan_Amount_Term],
                'Credit_History': [Credit_History],
                'Property_Area': [Property_Area]
            })

            for col in input_data.select_dtypes('object'):
                input_data[col] = le.fit_transform(input_data[col])

            prediction = model.predict(input_data)
            result = "✅ Approved" if prediction[0] == 1 else "❌ Not Approved"
            st.success(f"Loan Prediction Result: **{result}**")

else:
    st.info("👈 Upload a dataset to get started.")

