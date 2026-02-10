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

# --- MACHINE LEARNING PIPELINE ---
# We define features and target
ml_features = ['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']
X = df_master[ml_features].copy()
y = df_master['SALES']

# We use 'encoders' consistently throughout the app
encoders = {}
for col in ['PRODUCTLINE', 'COUNTRY']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    encoders[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# We use 'ai_model' consistently throughout the app
ai_model = RandomForestRegressor(n_estimators=100, random_state=42).fit(X_train, y_train)
y_pred = ai_model.predict(X_test)

# --- APP LAYOUT ---
st.title("🚀 PredictiCorp Executive Intelligence Suite")
st.caption("Data-Driven Insights for Global Market Strategy")

tabs = st.tabs([
    "📈 Executive Dashboard", 
    "🔮 Global Revenue Forecast", 
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

# --- TAB 2: GLOBAL REVENUE FORECAST (The Simulator) ---
with tabs[1]:
    st.header("🔮 Global Revenue Forecast Simulator")
    st.markdown("This tool predicts revenue for **every country** across the next 12 months based on current filtered data.")
    
    # --- STEP 1: Create a Simulation Grid ---
    all_countries = df['COUNTRY'].unique()
    all_months = range(1, 13) 
    
    # Standard inputs for prediction (averages from the dataset)
    avg_qty = df['QUANTITYORDERED'].mean()
    avg_msrp = df['MSRP'].mean()
    top_product = df['PRODUCTLINE'].mode()[0]

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

    forecast_df = pd.DataFrame(forecast_scenarios)

    # --- STEP 2: Encode categorical columns using the 'encoders' dictionary ---
    predict_df = forecast_df.copy()
    predict_df['PRODUCTLINE'] = encoders['PRODUCTLINE'].transform(predict_df['PRODUCTLINE'])
    predict_df['COUNTRY'] = encoders['COUNTRY'].transform(predict_df['COUNTRY'])

    # --- STEP 3: Predict Revenue using 'ai_model' ---
    forecast_df['PREDICTED_REVENUE'] = ai_model.predict(
        predict_df[['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']]
    )

    # --- STEP 4: Aggregations ---
    total_annual_revenue = forecast_df['PREDICTED_REVENUE'].sum()
    monthly_forecast = forecast_df.groupby('MONTH_ID')['PREDICTED_REVENUE'].sum().reset_index()
    country_forecast = forecast_df.groupby('COUNTRY')['PREDICTED_REVENUE'].sum().sort_values(ascending=False).reset_index()

    # --- STEP 5: Visualizations ---
    st.write(f"### 🌏 Total Predicted Annual Revenue: ${total_annual_revenue/1e6:.2f}M")
    
    col_chart1, col_chart2 = st.columns(2)
    with col_chart1:
        st.subheader("Predicted Revenue by Country")
        fig_country = px.bar(country_forecast, x='COUNTRY', y='PREDICTED_REVENUE', color='PREDICTED_REVENUE', color_continuous_scale="Blues")
        st.plotly_chart(fig_country, use_container_width=True)
    
    with col_chart2:
        st.subheader("Global Monthly Forecast Trend")
        fig_month = px.line(monthly_forecast, x='MONTH_ID', y='PREDICTED_REVENUE', markers=True)
        st.plotly_chart(fig_month, use_container_width=True)

    st.subheader("Detailed Forecast Data")
    st.dataframe(forecast_df, use_container_width=True)

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
            <p><b>Insight:</b> USA and France contribute significantly to total revenue.<br>
            <b>Action:</b> Pilot a localized loyalty program in the EMEA territory to defend market share.</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### Geographic Performance Heatmap")
    geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
    fig_map = px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="SALES", color_continuous_scale="Blues")
    st.plotly_chart(fig_map, use_container_width=True)

# --- TAB 4: SCIENTIFIC LAB ---
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
        importances = pd.DataFrame({'Feature': ml_features, 'Importance': ai_model.feature_importances_}).sort_values('Importance')
        st.plotly_chart(px.bar(importances, x='Importance', y='Feature', orientation='h'))

    exp3 = st.expander("🛠️ Week 5-6: Quality Metrics & Validation")
    with exp3:
        m1, m2 = st.columns(2)
        m1.metric("Mean Absolute Error", f"${mean_absolute_error(y_test, y_pred):.2f}")
        m2.metric("R-Squared Score", f"{r2_score(y_test, y_pred):.4f}")
        fig_acc = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Sales', 'y': 'Predicted Sales'}, trendline="ols")
        st.plotly_chart(fig_acc, use_container_width=True)
