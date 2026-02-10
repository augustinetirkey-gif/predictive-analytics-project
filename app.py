import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder
import datetime

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="PredictiCorp BI Suite", layout="wide", initial_sidebar_state="expanded")

# --- EXECUTIVE THEMING (Custom CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #1f4e79; }
    .css-10trblm { color: #1f4e79; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #f4f7f9; }
    .stTabs [data-baseweb="tab"] { background-color: #ffffff; border-radius: 10px 10px 0 0; border: 1px solid #e1e4e8; padding: 10px 20px; font-weight: bold; color: #5c6c7b; }
    .stTabs [aria-selected="true"] { background-color: #1f4e79 !important; color: white !important; }
    .card { background-color: #ffffff; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 8px solid #1f4e79; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ANALYTICS ENGINE ---
@st.cache_data
def get_processed_data():
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df['YEAR'] = df['ORDERDATE'].dt.year
    df['MONTH_NAME'] = df['ORDERDATE'].dt.month_name()
    df['DAY_NAME'] = df['ORDERDATE'].dt.day_name()
    return df

df_master = get_processed_data()

# --- SIDEBAR: STRATEGIC FILTERS ---
st.sidebar.title("🏢 BI Command Center")
st.sidebar.markdown("**Global Filtering Engine**")
st_year = st.sidebar.multiselect("Fiscal Year", options=sorted(df_master['YEAR'].unique()), default=df_master['YEAR'].unique())
st_country = st.sidebar.multiselect("Active Markets", options=sorted(df_master['COUNTRY'].unique()), default=df_master['COUNTRY'].unique())

# Filtered dataset for charts
df = df_master[(df_master['YEAR'].isin(st_year)) & (df_master['COUNTRY'].isin(st_country))]

# --- MACHINE LEARNING PIPELINE (The "Background" Work) ---
features = ['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']
X = df_master[features].copy()
y = df_master['SALES']
le_dict = {}
for col in ['PRODUCTLINE', 'COUNTRY']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred = model.predict(X_test)

# --- APP LAYOUT ---
st.title("🚀 PredictiCorp Executive Intelligence Suite")
st.caption("Data-Driven Insights for Global Market Strategy")

tabs = st.tabs([
    "📈 Executive Dashboard", 
    "🔮 Revenue Simulator", 
    "🌍 Market Insights", 
    "🧪 Scientific Lab (Week 1-6)"
])

# --- TAB 1: EXECUTIVE DASHBOARD ---
with tabs[0]:
    st.subheader("Performance KPIs")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total Revenue", f"${df['SALES'].sum()/1e6:.2f}M", "+5.2%")
    k2.metric("Avg Order Value", f"${df['SALES'].mean():,.2f}")
    k3.metric("Transaction Volume", f"{len(df):,}")
    k4.metric("Active Regions", f"{df['COUNTRY'].nunique()}")

    st.markdown("---")
    c1, c2 = st.columns([2, 1])
    with c1:
        st.markdown("### Revenue Momentum (Monthly Trend)")
        trend = df.groupby(['YEAR', 'MONTH_ID', 'MONTH_NAME'])['SALES'].sum().reset_index().sort_values(['YEAR', 'MONTH_ID'])
        fig_trend = px.line(trend, x='MONTH_NAME', y='SALES', color='YEAR', markers=True, template="plotly_white")
        st.plotly_chart(fig_trend, use_container_width=True)
    
    with c2:
        st.markdown("### Portfolio Composition")
        fig_pie = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism)
        st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 2: REVENUE SIMULATOR (Decision Making Tool) ---
# --- STEP 1: Create a Simulation Grid for all Countries and Months ---
# We get every unique country in your data
all_countries = df_base['COUNTRY'].unique()
all_months = range(1, 13) # Months 1 to 12

# We define standard inputs for the forecast (Averages)
avg_qty = df_base['QUANTITYORDERED'].mean()
avg_msrp = df_base['MSRP'].mean()
top_product = df_base['PRODUCTLINE'].mode()[0]

# Build the list of scenarios
forecast_scenarios = []
for country in all_countries:
    for month in all_months:
        qtr = (month - 1) // 3 + 1
        forecast_scenarios.append({
            'COUNTRY': country,
            'MONTH_ID': month,
            'QTR_ID': qtr,
            'MSRP': avg_msrp,
            'QUANTITYORDERED': avg_qty,
            'PRODUCTLINE': top_product
        })

# Convert to DataFrame
forecast_df = pd.DataFrame(forecast_scenarios)

# --- STEP 2: Transform the Data for the AI Model ---
# We must encode the 'Words' into 'Numbers' so the AI can read them
predict_df = forecast_df.copy()
predict_df['PRODUCTLINE'] = encoders['PRODUCTLINE'].transform(predict_df['PRODUCTLINE'])
predict_df['COUNTRY'] = encoders['COUNTRY'].transform(predict_df['COUNTRY'])

# --- STEP 3: Generate Predictions for Every Row ---
# The model predicts revenue for every country/month combination at once
forecast_df['PREDICTED_REVENUE'] = ai_model.predict(predict_df[['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']])

# --- STEP 4: Aggregate Results for Decision Making ---
# Now you can tell the business exactly what to expect:

# 1. Total Annual Global Prediction
total_annual_revenue = forecast_df['PREDICTED_REVENUE'].sum()

# 2. Month-wise Prediction (All countries combined)
monthly_forecast = forecast_df.groupby('MONTH_ID')['PREDICTED_REVENUE'].sum()

# 3. Country-wise Prediction (Full year total per country)
country_forecast = forecast_df.groupby('COUNTRY')['PREDICTED_REVENUE'].sum().sort_values(ascending=False)

# --- STEP 5: Visualizing in Streamlit ---
st.write(f"### 🌏 Total Global Forecasted Revenue: ${total_annual_revenue/1e6:.2f}M")

# Show Country Ranking
st.bar_chart(country_forecast)

# Show Monthly Seasonal Trend
st.line_chart(monthly_forecast)

# Show the full table for deep analysis
st.dataframe(forecast_df)
# --- TAB 3: MARKET INSIGHTS ---
with tabs[2]:
    st.header("💡 Business Directives")
    
    col_i1, col_i2 = st.columns(2)
    with col_i1:
        st.markdown("""
        <div class="card">
            <h4>📦 Inventory Optimization</h4>
            <p><b>Insight:</b> Peak demand occurs consistently in Q4. <br>
            <b>Action:</b> Increase Classic Car and Motorcycle inventory by 20% starting September.</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col_i2:
        st.markdown("""
        <div class="card">
            <h4>🌍 Regional Strategy</h4>
            <p><b>Insight:</b> USA and France contribute to 55% of total revenue.<br>
            <b>Action:</b> Pilot a localized loyalty program in the EMEA territory to defend market share.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Geographic Performance Heatmap")
    geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
    fig_map = px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="SALES", color_continuous_scale="Blues")
    st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 4: SCIENTIFIC LAB (The "Journey") ---
with tabs[3]:
    st.header("🧪 Data Science Methodology & Audit Trail")
    st.markdown("This section documents the technical rigor behind the predictions.")
    
    exp1 = st.expander("🛠️ Week 1-2: EDA & Data Foundations")
    with exp1:
        st.write("Ensuring data quality and uncovering hidden distributions.")
        st.dataframe(df_master.describe())
    
    exp2 = st.expander("🛠️ Week 3-4: Feature Engineering & Model Training")
    with exp2:
        st.write("Variables impacting the Random Forest Regressor:")
        importances = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance')
        st.plotly_chart(px.bar(importances, x='Importance', y='Feature', orientation='h'))

    exp3 = st.expander("🛠️ Week 5-6: Quality Metrics & Validation")
    with exp3:
        m1, m2 = st.columns(2)
        m1.metric("Mean Absolute Error", f"${mean_absolute_error(y_test, y_pred):.2f}")
        m2.metric("R-Squared Score", f"{r2_score(y_test, y_pred):.4f}")
        fig_acc = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Sales', 'y': 'Predicted Sales'}, trendline="ols")
        st.plotly_chart(fig_acc, use_container_width=True)
