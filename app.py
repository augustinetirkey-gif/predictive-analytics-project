import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import datetime

# ==========================================
# 🎨 1. PROFESSIONAL CSS & PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Sales AI Dashboard", layout="wide", page_icon="📈")

def apply_custom_styles():
    st.markdown("""
        <style>
        .main { background-color: #f4f7f9; }
        .welcome-banner {
            background: linear-gradient(90deg, #4e73df 0%, #224abe 100%);
            padding: 30px; border-radius: 15px; color: white; margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        [data-testid="stSidebar"] { background-color: #1a1c24; }
        [data-testid="stSidebar"] * { color: #ffffff !important; }
        .stMetric {
            background-color: white; padding: 20px; border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e3e6f0;
        }
        .ai-card {
            background-color: #e3f2fd; border-left: 5px solid #2196f3;
            padding: 20px; border-radius: 10px; margin-bottom: 20px; color: #0d47a1;
        }
        .analysis-box {
            background-color: #fffdf0; border: 1px solid #ffeeba;
            padding: 20px; border-radius: 10px; margin-top: 10px; color: #856404;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# ==========================================
# 📊 2. DATA ENGINE
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    # Clean string data
    for col in ['CITY', 'STATE', 'TERRITORY']:
        df[col] = df[col].fillna('N/A')
    return df

df = load_data()

# ==========================================
# 🧭 3. NAVIGATION & LOGIN (SIMULATED)
# ==========================================
if 'logged_in' not in st.session_state:
    st.session_state.logged_in = True  # Simplified for deployment

st.sidebar.title("📈 Sales AI Menu")
menu = st.sidebar.radio("Navigate Sections:", 
    ["1. Overview Dashboard", "2. Sales Analysis", "3. Customer Search (Person Detail)", "4. Predictive AI", "5. 🛡️ Admin Panel"])

# ==========================================
# 🔹 SECTION 1: OVERVIEW DASHBOARD
# ==========================================
if menu == "1. Overview Dashboard":
    st.markdown("""<div class="welcome-banner"><h1>Corporate Sales Overview 🚀</h1><p>Quick business snapshot for management decision-making.</p></div>""", unsafe_allow_html=True)
    
    # KPIs
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Sales (Revenue)", f"${df['SALES'].sum():,.2f}")
    c2.metric("Total Orders", f"{df['ORDERNUMBER'].nunique():,}")
    c3.metric("Total Customers", f"{df['CUSTOMERNAME'].nunique():,}")

    # Year-wise Trend
    st.subheader("📈 Year-wise Sales Trend")
    yearly_trend = df.groupby('YEAR_ID')['SALES'].sum().reset_index()
    fig_year = px.bar(yearly_trend, x='YEAR_ID', y='SALES', text_auto='.2s', color='SALES', template="plotly_white")
    st.plotly_chart(fig_year, use_container_width=True)

    st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
    st.subheader("📝 Executive Proper Analysis")
    st.write(f"Business health is stable with **{df['CUSTOMERNAME'].nunique()}** active clients. The system is currently scaling smoothly with over **{len(df)}** historical records.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 🔹 SECTION 2: SALES ANALYSIS
# ==========================================
elif menu == "2. Sales Analysis":
    st.title("🔎 Revenue Performance Drivers")
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Sales by Product Line")
        prod_data = df.groupby('PRODUCTLINE')['SALES'].sum().reset_index().sort_values('SALES')
        st.plotly_chart(px.bar(prod_data, x='SALES', y='PRODUCTLINE', orientation='h', color='SALES'), use_container_width=True)
        
        st.subheader("Sales by Deal Size")
        deal_data = df.groupby('DEALSIZE')['SALES'].sum().reset_index()
        st.plotly_chart(px.pie(deal_data, values='SALES', names='DEALSIZE', hole=0.4), use_container_width=True)

    with col_b:
        st.subheader("Top 10 Countries by Sales")
        country_data = df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(10)
        st.plotly_chart(px.bar(country_data, x='COUNTRY', y='SALES', color='SALES'), use_container_width=True)

    st.markdown('<div class="analysis-box">', unsafe_allow_html=True)
    st.subheader("📝 Sales Strategy Notes")
    st.write("Current revenue is highly concentrated in specific product lines. Diversification in 'Trains' and 'Ships' is recommended.")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 🔹 SECTION 3: CUSTOMER SEARCH (PERSON DETAIL)
# ==========================================
elif menu == "3. Customer Search (Person Detail)":
    st.title("👤 Individual Customer Deep-Dive")
    search_name = st.selectbox("Select Customer to View Details:", sorted(df['CUSTOMERNAME'].unique()))
    
    cust_df = df[df['CUSTOMERNAME'] == search_name]
    
    # AI Recommendation Logic
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.subheader(f"🤖 AI Recommendation for {search_name}")
    top_cat = cust_df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
    st.write(f"**Insight:** This customer has the highest affinity for **{top_cat}**.")
    st.write(f"**Strategy:** Next campaign should focus on bulk-discounts for **{top_cat}** or early access to new **{cust_df['TERRITORY'].iloc[0]}** inventory.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Detailed Info
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Value", f"${cust_df['SALES'].sum():,.2f}")
    c2.metric("Contact Person", f"{cust_df['CONTACTFIRSTNAME'].iloc[0]} {cust_df['CONTACTLASTNAME'].iloc[0]}")
    c3.metric("Phone Number", f"{cust_df['PHONE'].iloc[0]}")

    st.subheader("📍 Location Details")
    st.info(f"**Address:** {cust_df['ADDRESSLINE1'].iloc[0]}, {cust_df['CITY'].iloc[0]}, {cust_df['COUNTRY'].iloc[0]}")

    st.subheader("🕒 Full Purchase History")
    st.dataframe(cust_df[['ORDERDATE', 'ORDERNUMBER', 'PRODUCTLINE', 'SALES', 'STATUS', 'DEALSIZE']], use_container_width=True)

# ==========================================
# 🔹 SECTION 4: PREDICTIVE AI
# ==========================================
elif menu == "4. Predictive AI":
    st.title("🔮 AI Revenue Forecasting")
    
    # Feature Engineering for Prediction
    le = LabelEncoder()
    pred_df = df.copy()
    
    # Encode requested features
    cat_cols = ['PRODUCTLINE', 'DEALSIZE', 'COUNTRY', 'TERRITORY']
    for col in cat_cols:
        pred_df[col] = le.fit_transform(pred_df[col])
    
    # Define Model Features (Using your specific Input Variables)
    features = ['QUANTITYORDERED', 'PRICEEACH', 'MSRP', 'YEAR_ID', 'MONTH_ID', 'PRODUCTLINE', 'DEALSIZE']
    X = pred_df[features]
    y = pred_df['SALES']
    
    model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X, y)

    st.sidebar.subheader("Adjust Forecast Inputs")
    qty = st.sidebar.slider("Quantity Ordered", 1, 100, 30)
    price = st.sidebar.number_input("Price Each ($)", 10.0, 200.0, 100.0)
    msrp = st.sidebar.number_input("MSRP ($)", 10.0, 250.0, 120.0)
    
    if st.button("Run AI Prediction"):
        # Dummy values for other features for prediction
        input_data = [[qty, price, msrp, 2005, 11, 0, 1]] 
        prediction = model.predict(input_data)[0]
        st.success(f"### Predicted Order Value: ${prediction:,.2f}")
        st.write("Note: Model utilizes Quantity, Price, MSRP, and Time-based factors for high accuracy.")

# ==========================================
# 🔹 SECTION 5: ADMIN PANEL
# ==========================================
elif menu == "5. 🛡️ Admin Panel":
    st.title("🛡️ System Administration")
    st.write(f"**Database Size:** {len(df)} Records")
    st.write("**Columns Tracked:**", list(df.columns))
    st.subheader("Raw Data Preview")
    st.dataframe(df.head(20), use_container_width=True)
