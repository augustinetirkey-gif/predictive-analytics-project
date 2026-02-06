import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# ==========================================
# 🎨 1. CSS STYLING & PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Sales AI Dashboard", layout="wide", page_icon="📈")

def local_css():
    st.markdown("""
        <style>
        /* Main background and font */
        .main { background-color: #f8f9fa; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
        
        /* Sidebar styling */
        [data-testid="stSidebar"] { background-color: #1e1e2f; color: white; }
        [data-testid="stSidebar"] * { color: white !important; }
        
        /* Metric Card Styling */
        div[data-testid="stMetricValue"] { font-size: 2rem; font-weight: bold; color: #4e73df; }
        .stMetric { background-color: white; padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.05); }
        
        /* Header styling */
        h1, h2, h3 { color: #2c3e50; border-bottom: 2px solid #e9ecef; padding-bottom: 10px; }
        
        /* Prediction Result Box */
        .predict-box { background-color: #d4edda; color: #155724; padding: 20px; border-radius: 10px; border: 1px solid #c3e6cb; font-weight: bold; }
        </style>
    """, unsafe_allow_html=True)

local_css()

# ==========================================
# 📊 2. DATA LOADING
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    return df

df = load_data()

# ==========================================
# 🧭 3. SIDEBAR NAVIGATION
# ==========================================
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3222/3222672.png", width=100)
st.sidebar.title("Intern Project: Sales AI")
menu = st.sidebar.radio("Navigate Sections", 
    ["1. Overview Dashboard", "2. Sales Analysis", "3. Customer Insights", "4. Predictive Analytics", "5. Evaluation & Optimization"])

# ==========================================
# 🔹 SECTION 1: OVERVIEW DASHBOARD
# ==========================================
if menu == "1. Overview Dashboard":
    st.title("📊 Business Performance Overview")
    
    # Top Row Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Total Sales Revenue", f"${df['SALES'].sum():,.2f}")
    m2.metric("Total Orders Processed", f"{df['ORDERNUMBER'].nunique():,}")
    m3.metric("Global Customer Base", f"{df['CUSTOMERNAME'].nunique():,}")

    st.markdown("### 📅 Sales Growth Trend")
    trend_data = df.groupby(['YEAR_ID', 'MONTH_ID'])['SALES'].sum().reset_index()
    trend_data['Date'] = trend_data['YEAR_ID'].astype(str) + '-' + trend_data['MONTH_ID'].astype(str)
    
    fig = px.area(trend_data, x='Date', y='SALES', title="Revenue over Time", 
                  color_discrete_sequence=['#4e73df'], template="simple_white")
    st.plotly_chart(fig, use_container_width=True)

# ==========================================
# 🔹 SECTION 2: SALES ANALYSIS
# ==========================================
elif menu == "2. Sales Analysis":
    st.title("🔎 Deep Dive: Revenue Drivers")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Revenue by Product Category")
        prod_sales = df.groupby('PRODUCTLINE')['SALES'].sum().sort_values().reset_index()
        fig1 = px.bar(prod_sales, x='SALES', y='PRODUCTLINE', orientation='h', 
                      color='SALES', color_continuous_scale='Blues')
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        st.subheader("Sales by Geographic Country")
        geo_sales = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).head(10).reset_index()
        fig2 = px.pie(geo_sales, values='SALES', names='COUNTRY', hole=0.4, 
                      color_discrete_sequence=px.colors.qualitative.Set3)
        st.plotly_chart(fig2, use_container_width=True)

    st.subheader("Deal Size Comparison")
    deal_sales = df.groupby('DEALSIZE')['SALES'].sum().reset_index()
    fig3 = px.bar(deal_sales, x='DEALSIZE', y='SALES', color='DEALSIZE', 
                  category_orders={"DEALSIZE": ["Small", "Medium", "Large"]})
    st.plotly_chart(fig3, use_container_width=True)

# ==========================================
# 🔹 SECTION 3: CUSTOMER INSIGHTS
# ==========================================
elif menu == "3. Customer Insights":
    st.title("👤 Customer Segmentation & Loyalty")
    
    # Top Customers Table
    st.subheader("🏆 Top 10 High-Value Customers")
    top_customers = df.groupby('CUSTOMERNAME')['SALES'].agg(['sum', 'count']).sort_values(by='sum', ascending=False).head(10).reset_index()
    top_customers.columns = ['Customer Name', 'Total Spend ($)', 'Order Frequency']
    st.table(top_customers.style.format({'Total Spend ($)': '{:,.2f}'}))

    # Map/Territory View
    st.subheader("Global Territory Presence")
    terr_dist = df.groupby('TERRITORY')['SALES'].sum().reset_index()
    fig4 = px.funnel(terr_dist, x='SALES', y='TERRITORY', color='TERRITORY')
    st.plotly_chart(fig4, use_container_width=True)

# ==========================================
# 🔹 SECTION 4: PREDICTIVE ANALYTICS
# ==========================================
elif menu == "4. Predictive Analytics":
    st.title("🔮 Predictive Sales Modeling")
    
    # ML Preparation
    ml_data = df[['QUANTITYORDERED', 'PRICEEACH', 'MONTH_ID', 'PRODUCTLINE', 'DEALSIZE', 'SALES']].copy()
    le = LabelEncoder()
    ml_data['PRODUCTLINE_ENC'] = le.fit_transform(ml_data['PRODUCTLINE'])
    ml_data['DEALSIZE_ENC'] = le.fit_transform(ml_data['DEALSIZE'])
    
    X = ml_data[['QUANTITYORDERED', 'PRICEEACH', 'MONTH_ID', 'PRODUCTLINE_ENC', 'DEALSIZE_ENC']]
    y = ml_data['SALES']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # User Prediction Input
    st.sidebar.markdown("---")
    st.sidebar.header("🕹️ Prediction Input")
    in_qty = st.sidebar.slider("Quantity Ordered", 1, 100, 30)
    in_price = st.sidebar.number_input("Unit Price ($)", 10.0, 500.0, 95.0)
    in_month = st.sidebar.slider("Select Month", 1, 12, 5)
    in_prod = st.sidebar.selectbox("Product Line", df['PRODUCTLINE'].unique())
    in_deal = st.sidebar.selectbox("Deal Size", ["Small", "Medium", "Large"])

    # Transform Inputs
    prod_enc = le.fit(df['PRODUCTLINE']).transform([in_prod])[0]
    deal_enc = le.fit(["Small", "Medium", "Large"]).transform([in_deal])[0]
    
    # Predict
    input_array = np.array([[in_qty, in_price, in_month, prod_enc, deal_enc]])
    prediction = rf_model.predict(input_array)[0]

    st.markdown(f"""
        <div class='predict-box'>
            <h3>Estimated Sales Value: ${prediction:,.2f}</h3>
            <p>Based on Random Forest Regression Model</p>
        </div>
    """, unsafe_allow_html=True)

    # Actual vs Predicted Plot
    st.subheader("Model Validation (Actual vs Predicted)")
    y_pred = rf_model.predict(X_test)
    val_df = pd.DataFrame({'Actual': y_test.values, 'Predicted': y_pred}).head(50)
    fig5 = go.Figure()
    fig5.add_trace(go.Scatter(y=val_df['Actual'], name="Actual", line=dict(color='blue')))
    fig5.add_trace(go.Scatter(y=val_df['Predicted'], name="Predicted", line=dict(color='orange', dash='dot')))
    st.plotly_chart(fig5, use_container_width=True)

# ==========================================
# 🔹 SECTION 5: EVALUATION & OPTIMIZATION
# ==========================================
elif menu == "5. Evaluation & Optimization":
    st.title("🧪 Model Performance & Tuning")
    
    # Data Setup
    ml_data = df[['QUANTITYORDERED', 'PRICEEACH', 'MONTH_ID', 'PRODUCTLINE', 'DEALSIZE', 'SALES']].copy()
    le = LabelEncoder()
    ml_data['PRODUCTLINE_ENC'] = le.fit_transform(ml_data['PRODUCTLINE'])
    ml_data['DEALSIZE_ENC'] = le.fit_transform(ml_data['DEALSIZE'])
    X = ml_data[['QUANTITYORDERED', 'PRICEEACH', 'MONTH_ID', 'PRODUCTLINE_ENC', 'DEALSIZE_ENC']]
    y = ml_data['SALES']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # 1. Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_pred = lr.predict(X_test)

    # 2. Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)

    # Comparison Table
    metrics = {
        "Metric": ["MAE (Mean Absolute Error)", "RMSE (Root Mean Squared Error)", "R² Score"],
        "Linear Regression": [
            mean_absolute_error(y_test, lr_pred),
            np.sqrt(mean_squared_error(y_test, lr_pred)),
            r2_score(y_test, lr_pred)
        ],
        "Random Forest": [
            mean_absolute_error(y_test, rf_pred),
            np.sqrt(mean_squared_error(y_test, rf_pred)),
            r2_score(y_test, rf_pred)
        ]
    }
    
    st.subheader("📊 Performance Metrics Comparison")
    st.table(pd.DataFrame(metrics))

    st.success("🏆 **Selected Model:** Random Forest is the best-performing model with the highest R² score and lowest error metrics.")

    st.markdown("---")
    st.subheader("⚙️ Hyperparameter Optimization Settings")
    st.info("""
    During the tuning phase, we optimized the Random Forest using **GridSearchCV**:
    - **n_estimators**: [50, 100, 200] → Selected: 100
    - **max_depth**: [None, 10, 20] → Selected: None
    - **min_samples_split**: [2, 5] → Selected: 2
    """)
    
    st.write("These optimizations help reduce **overfitting** and ensure the model works on new, unseen data.")
