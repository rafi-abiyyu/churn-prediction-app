import streamlit as st
import pandas as pd
import pickle
import numpy as np
import plotly.express as px

# Page config
st.set_page_config(
    page_title="Customer Churn Prediction",
    layout="wide"
)

# Define consistent colors
RED_COLOR = "#DC3545"
GREEN_COLOR = "#28A745"

# Custom CSS for white background and styling
st.markdown(f"""
<style>
    .stApp {{
        background-color: white;
    }}
    .main {{
        background-color: white;
    }}
    [data-testid="stAppViewContainer"] {{
        background-color: white;
    }}
    [data-testid="stHeader"] {{
        background-color: white;
    }}
    .result-card {{
        padding: 20px;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin: 10px;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }}
    .churn-card {{
        background-color: {RED_COLOR};
        color: white;
    }}
    .stay-card {{
        background-color: {GREEN_COLOR};
        color: white;
    }}
    .churn-prob-card {{
        background-color: {RED_COLOR};
        color: white;
    }}
    .retention-prob-card {{
        background-color: {GREEN_COLOR};
        color: white;
    }}
    .result-value {{
        font-size: 28px;
        font-weight: bold;
        margin: 10px 0;
    }}
    .result-label {{
        font-size: 14px;
        opacity: 0.9;
    }}
    .risk-badge {{
        display: inline-block;
        padding: 5px 15px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        margin-top: 10px;
        background-color: rgba(255, 255, 255, 0.3);
        color: white;
    }}
</style>
""", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    with open('model_xgboost.sav', 'rb') as file:
        model = pickle.load(file)
    return model

model = load_model()

# Title
st.title("Customer Churn Prediction")
st.markdown("---")

# Columns for input - 9 inputs each for symmetry
col1, col2 = st.columns(2)

with col1:
    tenure = st.number_input(
        "Tenure (months)",
        min_value=0.0,
        max_value=61.0,
        value=10.0,
        step=1.0,
        help="Number of months the customer has been with the company"
    )
    
    preferred_login_device = st.radio(
        "Preferred Login Device",
        options=["Phone", "Computer"],
        horizontal=True,
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
    
    gender = st.radio(
        "Gender",
        options=["Male", "Female"],
        horizontal=True,
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
    
    prefered_order_cat = st.selectbox(
        "Preferred Order Category",
        options=["Laptop & Accessory", "Phone", "Fashion", "Grocery", "Others"],
        format_func=lambda x: "Mobile Phone" if x == "Phone" else x,
        help="Most frequently ordered product category"
    )

with col2:
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
    
    complain_option = st.radio(
        "Complaint Raised",
        options=["No", "Yes"],
        horizontal=True,
        help="Whether customer has raised any complaint"
    )
    complain = 1 if complain_option == "Yes" else 0
    
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

# Prediction
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
    
    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if prediction == 1:
                st.markdown(f"""
                <div class="result-card churn-card">
                    <div class="result-label">Churn Prediction</div>
                    <div class="result-value">Will Churn</div>
                    <div class="risk-badge">High Risk</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="result-card stay-card">
                    <div class="result-label">Churn Prediction</div>
                    <div class="result-value">Will Stay</div>
                    <div class="risk-badge">Low Risk</div>
                </div>
                """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="result-card churn-prob-card">
                <div class="result-label">Churn Probability</div>
                <div class="result-value">{prediction_proba[1]:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown(f"""
            <div class="result-card retention-prob-card">
                <div class="result-label">Retention Probability</div>
                <div class="result-value">{prediction_proba[0]:.2%}</div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Pie Chart for Probability Distribution
        st.markdown("### Probability Distribution")
        
        prob_df = pd.DataFrame({
            'Outcome': ['Will Stay', 'Will Churn'],
            'Probability': [prediction_proba[0], prediction_proba[1]]
        })
        
        fig = px.pie(
            prob_df, 
            values='Probability', 
            names='Outcome',
            color='Outcome',
            color_discrete_map={
                'Will Stay': GREEN_COLOR,
                'Will Churn': RED_COLOR
            },
            hole=0.4
        )
        
        fig.update_traces(
            textposition='inside',
            textinfo='percent+label',
            textfont_size=14
        )
        
        fig.update_layout(
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=-0.2,
                xanchor="center",
                x=0.5
            ),
            paper_bgcolor='white',
            plot_bgcolor='white',
            margin=dict(t=20, b=20, l=20, r=20)
        )
        
        st.plotly_chart(fig, use_container_width=True)
                  
    except Exception as e:
        st.error(f"Error making prediction: {str(e)}")
        st.info("Please check the error details above and ensure all inputs are correct.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Built with Streamlit | XGBoost Model | Customer Churn Prediction</p>
</div>
""", unsafe_allow_html=True)
