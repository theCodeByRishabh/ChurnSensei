import joblib
import numpy as np
import pandas as pd
import streamlit as st
import os

MODEL_PATH = "models/best_churn_model.pkl"

FEATURE_COLUMNS = [
    "gender",
    "SeniorCitizen",
    "Partner",
    "Dependents",
    "tenure",
    "PhoneService",
    "InternetService",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
    "MonthlyCharges",
    "TotalCharges",
]


@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. "
            f"Please run train_model.py first to train and save the model."
        )
    model = joblib.load(MODEL_PATH)
    return model


def main():
    st.title("📉 ChurnSensei – Customer Retention Prediction")

    st.write(
        """
        Predict whether a telecom customer is likely to **churn** (leave) based on their profile.  
        Fill in the form below and click **Predict**.
        """
    )

    model = load_model()

    st.sidebar.header("About")
    st.sidebar.write(
        """
        **Tech stack:**
        - Python, Pandas, NumPy  
        - Scikit-Learn, XGBoost  
        - Matplotlib (for training plots)  
        - Streamlit (this app)
        """
    )

    st.header("Customer Details")

    with st.form("churn_form"):
        col1, col2 = st.columns(2)

        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            senior = st.selectbox("Senior Citizen (0 = No, 1 = Yes)", [0, 1])
            partner = st.selectbox("Has Partner?", ["Yes", "No"])
            dependents = st.selectbox("Has Dependents?", ["Yes", "No"])
            tenure = st.number_input("Tenure (months)", min_value=0, max_value=1000, value=12, step=1)
            phone_service = st.selectbox("Phone Service", ["Yes", "No"])

        with col2:
            internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
            contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless Billing", ["Yes", "No"])
            payment_method = st.selectbox(
                "Payment Method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer (automatic)",
                    "Credit card (automatic)",
                ],
            )
            monthly_charges = st.number_input("Monthly Charges", min_value=0.0, value=70.0, step=1.0)
            total_charges = st.number_input("Total Charges", min_value=0.0, value=1000.0, step=10.0)

        submitted = st.form_submit_button("Predict")

    if submitted:
        input_data = {
            "gender": gender,
            "SeniorCitizen": int(senior),
            "Partner": partner,
            "Dependents": dependents,
            "tenure": float(tenure),
            "PhoneService": phone_service,
            "InternetService": internet_service,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment_method,
            "MonthlyCharges": float(monthly_charges),
            "TotalCharges": float(total_charges),
        }

        input_df = pd.DataFrame([input_data], columns=FEATURE_COLUMNS)

        pred_proba = model.predict_proba(input_df)[0][1]  
        pred_class = model.predict(input_df)[0]

        churn_prob_percent = pred_proba * 100

        st.subheader("Prediction Result")

        if pred_class == 1:
            st.error("⚠️ This customer is **likely to churn**.")
        else:
            st.success("✅ This customer is **likely to stay**.")

        st.metric(
            label="Churn Probability",
            value=f"{churn_prob_percent:.1f} %",
        )

        prob_float = float(pred_proba)

        st.progress(max(0.0, min(prob_float, 1.0)))

        st.write("_(1.0 = 100% confidence in churn)_")



if __name__ == "__main__":
    main()
