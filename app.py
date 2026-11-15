
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("model.pkl")

st.title("Customer Repurchase Prediction")

st.write(
    "Predict whether a customer is likely to make "
    "a purchase in the next 90 days."
)

total_orders = st.number_input("Total Orders", min_value=1)
total_spend = st.number_input("Total Spend", min_value=0.0)
total_quantity = st.number_input("Total Quantity", min_value=1)
avg_price = st.number_input("Average Price", min_value=0.0)
unique_products = st.number_input("Unique Products", min_value=1)
recency = st.number_input("Days Since Last Purchase", min_value=0)
tenure = st.number_input("Customer Tenure in Days", min_value=0)
avg_order_value = st.number_input("Average Order Value", min_value=0.0)
orders_per_month = st.number_input("Orders Per Month", min_value=0.0)
country = st.text_input("Country", value="United Kingdom")

if st.button("Predict"):

    customer = pd.DataFrame([{
        "total_orders": total_orders,
        "total_spend": total_spend,
        "total_quantity": total_quantity,
        "avg_price": avg_price,
        "unique_products": unique_products,
        "recency": recency,
        "tenure": tenure,
        "avg_order_value": avg_order_value,
        "orders_per_month": orders_per_month,
        "country": country
    }])

    prediction = model.predict(customer)[0]

    probability = model.predict_proba(customer)[0][1]

    st.write("Repurchase Probability:", round(probability * 100, 2), "%")

    if prediction == 1:
        st.success("Customer is predicted to repurchase.")
    else:
        st.warning("Customer is not predicted to repurchase.")
