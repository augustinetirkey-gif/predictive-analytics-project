import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Sales Predictive Platform", layout="wide")

# Conversion Rate (USD to INR)
USD_TO_INR = 90.55

# Custom Styling
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { border: 1px solid #e0e0e0; padding: 15px; border-radius: 10px; background: white; }
    </style>
    """, unsafe_allow_html=True)

# --- WEEK 1 & 3: DATA LOADING & FEATURE ENGINEERING ---
@st.cache_data
def load_data():
    # Ensure this file exists in your folder!
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    # Extracting features
    df['MONTH'] = df['ORDERDATE'].dt.month
    df['YEAR'] = df['ORDERDATE'].dt.year
    df['QUARTER'] = df['ORDERDATE'].dt.quarter
    # Create an INR column for charts
    df['SALES_INR'] = df['SALES'] * USD_TO_INR
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'cleaned_sales_data.csv' not found.")
    st.stop()

# --- HEADER ---
st.title("🎯 AI-Based Predictive Analytics Platform")
st.subheader("Project: Forecasting Trends & Business Outcomes (In Rupees ₹)")
st.write("---")

# --- NAVIGATION TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Week 1-2: EDA", 
    "📍 Week 3: Feature Engineering", 
    "📍 Week 4: Model Building", 
    "📍 Week 5: Evaluation",
    "📍 Week 6: Deployment"
])

# --- WEEK 1 & 2: EXPLORATORY DATA ANALYSIS ---
with tab1:
    st.header("📊 Exploratory Data Analysis")
    
    # Calculate values in INR
    total_rev_inr = df['SALES_INR'].sum()
    avg_order_inr = df['SALES_INR'].mean()
    
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Historical Revenue (INR)", f"₹{total_rev_inr:,.2f}")
    m2.metric("Average Order Value (INR)", f"₹{avg_order_inr:,.2f}")
    m3.metric("Data Rows Processed", len(df))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue by Product Line (₹)")
        fig_prod = px.bar(df.groupby('PRODUCTLINE')['SALES_INR'].sum().reset_index(), 
                          x='PRODUCTLINE', y='SALES_INR', color='SALES_INR', 
                          labels={'SALES_INR':'Sales (₹)'}, template="plotly_white")
        st.plotly_chart(fig_prod, use_container_width=True)
    with c2:
        st.subheader("Top Markets (Country)")
        fig_geo = px.pie(df.groupby('COUNTRY')['SALES_INR'].sum().reset_index(), 
                         values='SALES_INR', names='COUNTRY', hole=0.4)
        st.plotly_chart(fig_geo, use_container_width=True)

# --- WEEK 3: FEATURE ENGINEERING ---
with tab2:
    st.header("⚙️ Feature Engineering")
    st.info("Converting raw data into mathematical features for the AI model.")
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Target Variable:** `SALES` (Predicted in USD then converted)")
        st.write("**Temporal Features:** `MONTH`, `YEAR`, `QUARTER`")
    with col_b:
        st.write("**Categorical Features:** `PRODUCTLINE`, `COUNTRY` (Encoded)")

# --- WEEK 4: MODEL BUILDING ---
with tab3:
    st.header("🤖 Machine Learning Model")
    features = ['MONTH', 'QUARTER', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']
    X = df[features].copy()
    y = df['SALES'] # Training on base USD for numerical stability
    
    le_dict = {}
    for col in ['PRODUCTLINE', 'COUNTRY']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    st.success("Model trained successfully!")

# --- WEEK 5: EVALUATION ---
with tab4:
    st.header("📉 Performance Metrics (Converted to INR)")
    y_pred = model.predict(X_test)
    
    # Calculate Metrics and convert to INR
    mae_inr = mean_absolute_error(y_test, y_pred) * USD_TO_INR
    rmse_inr = np.sqrt(mean_squared_error(y_test, y_pred)) * USD_TO_INR
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE (Error in ₹)", f"₹{mae_inr:.2f}")
    col2.metric("RMSE (Error in ₹)", f"₹{rmse_inr:.2f}")
    col3.metric("R² Score (Accuracy)", f"{r2:.4f}")

# --- WEEK 6: DEPLOYMENT ---
with tab5:
    st.header("🚀 Strategic Forecasting Tool")
    
    with st.container():
        c_in1, c_in2, c_in3 = st.columns(3)
        with c_in1:
            i_month = st.selectbox("Forecast Month", range(1, 13))
            i_qtr = (i_month-1)//3 + 1
        with c_in2:
            i_prod = st.selectbox("Product Line", df['PRODUCTLINE'].unique())
            i_qty = st.number_input("Target Quantity", value=30)
        with c_in3:
            i_country = st.selectbox("Market Country", df['COUNTRY'].unique())
            i_msrp = st.number_input("Standard MSRP (USD)", value=100)

        if st.button("Generate Outcome Forecast"):
            p_prod = le_dict['PRODUCTLINE'].transform([i_prod])[0]
            p_country = le_dict['COUNTRY'].transform([i_country])[0]
            
            input_arr = np.array([[i_month, i_qtr, i_msrp, i_qty, p_prod, p_country]])
            prediction_usd = model.predict(input_arr)[0]
            prediction_inr = prediction_usd * USD_TO_INR
            
            st.markdown("---")
            st.write(f"### Predicted Transaction Value: :green[**₹{prediction_inr:,.2f}**]")
            
            if prediction_usd > df['SALES'].mean():
                st.success("💡 Recommendation: High-priority deal (Above Average).")
            else:
                st.info("💡 Recommendation: Standard transaction.")
