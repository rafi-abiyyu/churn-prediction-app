import streamlit as st
import pandas as pd
import pickle
import numpy as np

#settingan page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

#load model
@st.cache_resource
def load_model():
    with open('model_xgboost.sav', 'rb') as file:
        model = pickle.load(file)
    return model
model=load_model()

#title
st.title("Customer Churn Prediction ")
st.markdown("---")

#column
col1, col2=st.columns(2)
with col1:
    
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0.0,
        max_value=61.0,
        value=10.0,
        step=1.0,
        help="Number of months the customer has been with the company"
    )
    
    preferred_login_device = st.selectbox(
        "Preferred Login Device",
        options=["Phone", "Computer"],
        help="Device used most frequently to login"
    )
    
    city_tier = st.selectbox(
        "City Tier",
        options=[1, 2, 3],
        help="Tier of the city (1: Metro, 2: Tier-1, 3: Tier-2)"
    )
    
    warehouse_to_home = st.number_input(
        "Warehouse to Home Distance (km)",
        min_value=5.0,
        max_value=127.0,
        value=15.0,
        step=1.0,
        help="Distance from warehouse to customer's home"
    )
    
    preferred_payment_mode = st.selectbox(
        "Preferred Payment Mode",
        options=["Debit Card", "CC", "UPI", "COD", "E wallet"],
        format_func=lambda x: {
            "CC": "Credit Card",
            "COD": "Cash on Delivery"
        }.get(x, x),
        help="Most frequently used payment method"
    )
    
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"],
        help="Customer's gender"
    )
    
    hour_spend_on_app = st.number_input(
        "Hours Spent on App",
        min_value=0.0,
        max_value=5.0,
        value=3.0,
        step=1.0,
        help="Average hours spent on the app per day"
    )
    
    number_of_device_registered = st.number_input(
        "Number of Devices Registered",
        min_value=1,
        max_value=6,
        value=3,
        step=1,
        help="Number of devices registered to the account"
    )

with col2:
    
    prefered_order_cat = st.selectbox(
        "Preferred Order Category",
        options=["Laptop & Accessory", "Phone", "Fashion", "Grocery", "Others"],
        format_func=lambda x: "Mobile Phone" if x == "Phone" else x,
        help="Most frequently ordered product category"
    )
    
    satisfaction_score = st.slider(
        "Satisfaction Score",
        min_value=1,
        max_value=5,
        value=3,
        step=1,
        help="Customer satisfaction rating (1-5)"
    )
    
    marital_status = st.selectbox(
        "Marital Status",
        options=["Single", "Married", "Divorced"],
        help="Customer's marital status"
    )
    
    number_of_address = st.number_input(
        "Number of Addresses",
        min_value=1,
        max_value=22,
        value=2,
        step=1,
        help="Number of addresses saved in the account"
    )
    
    complain = st.selectbox(
        "Complaint Raised",
        options=[0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No",
        help="Whether customer has raised any complaint"
    )
    
    order_amount_hike_from_last_year = st.number_input(
        "Order Amount Hike from Last Year (%)",
        min_value=11.0,
        max_value=26.0,
        value=15.0,
        step=1.0,
        help="Percentage increase in order amount from last year"
    )
    
    coupon_used = st.number_input(
        "Coupons Used",
        min_value=0.0,
        max_value=16.0,
        value=1.0,
        step=1.0,
        help="Number of coupons used in the last month"
    )
    
    order_count = st.number_input(
        "Order Count",
        min_value=1.0,
        max_value=16.0,
        value=1.0,
        step=1.0,
        help="Number of orders placed in the last month"
    )
    
    day_since_last_order = st.number_input(
        "Days Since Last Order",
        min_value=0.0,
        max_value=46.0,
        value=5.0,
        step=1.0,
        help="Number of days since the last order was placed"
    )
    
    cashback_amount = st.number_input(
        "Cashback Amount",
        min_value=0.0,
        max_value=324.99,
        value=150.0,
        step=10.0,
        help="Average cashback amount received"
    )

st.markdown("---")

#prediction
if st.button("Predict", type="primary", use_container_width=True):
    input_data = pd.DataFrame({
        'Tenure': [float(tenure)],
        'PreferredLoginDevice': [preferred_login_device],
        'CityTier': [int(city_tier)],
        'WarehouseToHome': [float(warehouse_to_home)],
        'PreferredPaymentMode': [preferred_payment_mode],
        'Gender': [gender],
        'HourSpendOnApp': [float(hour_spend_on_app)],
        'NumberOfDeviceRegistered': [int(number_of_device_registered)],
        'PreferedOrderCat': [prefered_order_cat],
        'SatisfactionScore': [int(satisfaction_score)],
        'MaritalStatus': [marital_status],
        'NumberOfAddress': [int(number_of_address)],
        'Complain': [int(complain)],
        'OrderAmountHikeFromlastYear': [float(order_amount_hike_from_last_year)],
        'CouponUsed': [float(coupon_used)],
        'OrderCount': [float(order_count)],
        'DaySinceLastOrder': [float(day_since_last_order)],
        'CashbackAmount': [float(cashback_amount)]
    })
    
    #buat probabilitas prediksi
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="Churn Prediction",
                value="Will Churn" if prediction == 1 else "Will Stay",
                delta="High Risk" if prediction == 1 else "Low Risk",
                delta_color="inverse"
            )
        
        with col2:
            st.metric(
                label="Churn Probability",
                value=f"{prediction_proba[1]:.2%}",
                delta=f"{prediction_proba[1] - 0.5:.2%} from baseline"
            )
        
        with col3:
            st.metric(
                label="Retention Probability",
                value=f"{prediction_proba[0]:.2%}",
                delta=f"{0.5 - prediction_proba[0]:.2%} from baseline",
                delta_color="inverse"
            )        
        st.markdown("### Probability Distribution")
        prob_df = pd.DataFrame({
            'Outcome': ['Will Stay', 'Will Churn'],
            'Probability': [prediction_proba[0], prediction_proba[1]]
        })
        st.bar_chart(prob_df.set_index('Outcome'))  
                  
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check the error details above and ensure all inputs are correct.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center'>
    <p>Built with Streamlit | XGBoost Model | Customer Churn Prediction</p>
</div>
""", unsafe_allow_html=True)