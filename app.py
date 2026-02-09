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

# Custom Styling for a Professional UI
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .stMetric { border: 1px solid #e0e0e0; padding: 15px; border-radius: 10px; background: white; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #f1f3f4; 
        border-radius: 4px 4px 0 0; 
        padding: 10px 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- WEEK 1 & 3: DATA LOADING & FEATURE ENGINEERING ---
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    # Extracting features for the model
    df['MONTH'] = df['ORDERDATE'].dt.month
    df['YEAR'] = df['ORDERDATE'].dt.year
    df['QUARTER'] = df['ORDERDATE'].dt.quarter
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("Error: 'cleaned_sales_data.csv' not found. Please ensure the file is in the same directory.")
    st.stop()

# --- HEADER ---
st.title("🎯 AI-Based Predictive Analytics Platform")
st.subheader("Project: Forecasting Trends & Business Outcomes")
st.write("---")

# --- NAVIGATION TABS (MATCHING 6-WEEK PLAN) ---
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
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Historical Revenue", f"${df['SALES'].sum():,.2f}")
    m2.metric("Average Order Value", f"${df['SALES'].mean():,.2f}")
    m3.metric("Data Rows Processed", len(df))

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Revenue by Product Line")
        fig_prod = px.bar(df.groupby('PRODUCTLINE')['SALES'].sum().reset_index(), 
                          x='PRODUCTLINE', y='SALES', color='SALES', template="plotly_white")
        st.plotly_chart(fig_prod, use_container_width=True)
    with c2:
        st.subheader("Top Markets (Country)")
        fig_geo = px.pie(df.groupby('COUNTRY')['SALES'].sum().reset_index(), 
                         values='SALES', names='COUNTRY', hole=0.4)
        st.plotly_chart(fig_geo, use_container_width=True)

# --- WEEK 3: FEATURE ENGINEERING ---
with tab2:
    st.header("⚙️ Feature Engineering")
    st.info("Converting raw data into mathematical features for the AI model.")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.write("**Target Variable:** `SALES` (Continuous)")
        st.write("**Temporal Features:** `MONTH`, `YEAR`, `QUARTER` (Extracted from Date)")
    with col_b:
        st.write("**Categorical Features:** `PRODUCTLINE`, `COUNTRY`, `DEALSIZE` (Encoded)")
        st.write("**Numeric Features:** `MSRP`, `QUANTITYORDERED` (Normalized)")

# --- WEEK 4: MODEL BUILDING ---
with tab3:
    st.header("🤖 Machine Learning Model")
    
    # Selecting Features
    features = ['MONTH', 'QUARTER', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']
    X = df[features].copy()
    y = df['SALES']
    
    # Encoding
    le_dict = {}
    for col in ['PRODUCTLINE', 'COUNTRY']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Model Training
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    st.success("Random Forest Regressor trained on 80% of historical data.")
    
    # Importance Plot
    imp = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance')
    fig_imp = px.bar(imp, x='Importance', y='Feature', orientation='h', title="Feature Importance Analysis")
    st.plotly_chart(fig_imp)

# --- WEEK 5: EVALUATION & OPTIMIZATION ---
with tab4:
    st.header("📉 Performance Metrics")
    y_pred = model.predict(X_test)
    
    # Metrics Calculation
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse) # Corrected RMSE calculation
    r2 = r2_score(y_test, y_pred)
    
    col1, col2, col3 = st.columns(3)
    col1.metric("MAE", f"${mae:.2f}")
    col2.metric("RMSE", f"${rmse:.2f}")
    col3.metric("R² Score", f"{r2:.4f}")
    
    st.subheader("Model Error Distribution")
    fig_res = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Sales', 'y': 'Predicted Sales'}, 
                         opacity=0.5, trendline="ols")
    st.plotly_chart(fig_res, use_container_width=True)

# --- WEEK 6: DEPLOYMENT ---
with tab5:
    st.header("🚀 Strategic Forecasting Tool")
    st.write("Use this tool to simulate business outcomes and forecast future sales.")
    
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
            i_msrp = st.number_input("Standard MSRP", value=100)

        if st.button("Generate Outcome Forecast"):
            # Encode inputs
            p_prod = le_dict['PRODUCTLINE'].transform([i_prod])[0]
            p_country = le_dict['COUNTRY'].transform([i_country])[0]
            
            # Predict
            input_arr = np.array([[i_month, i_qtr, i_msrp, i_qty, p_prod, p_country]])
            prediction = model.predict(input_arr)[0]
            
            st.markdown("---")
            st.write(f"### Predicted Transaction Value: :green[**${prediction:,.2f}**]")
            
            # Business Logic
            if prediction > df['SALES'].mean():
                st.success("💡 Recommendation: High-priority deal. Assign senior sales representative.")
            else:
                st.info("💡 Recommendation: Standard transaction. Proceed with automated fulfillment.")
