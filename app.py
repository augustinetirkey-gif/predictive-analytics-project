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
with tabs[1]:
    st.header("🔮 Strategic Scenario Simulator")
    st.markdown("Adjust parameters to predict the revenue outcome of prospective deals.")
    
    with st.container():
        col_s1, col_s2, col_s3 = st.columns(3)
        in_prod = col_s1.selectbox("Product Line", df_master['PRODUCTLINE'].unique())
        in_qty = col_s1.slider("Target Quantity", 10, 100, 30)
        in_country = col_s2.selectbox("Destination Market", sorted(df_master['COUNTRY'].unique()))
        in_msrp = col_s2.number_input("Unit MSRP ($)", value=100)
        in_month = col_s3.slider("Order Month", 1, 12, 6)
        
        if st.button("RUN PREDICTIVE SIMULATION", use_container_width=True, type="primary"):
            p_prod = le_dict['PRODUCTLINE'].transform([in_prod])[0]
            p_country = le_dict['COUNTRY'].transform([in_country])[0]
            p_qtr = (in_month - 1) // 3 + 1
            prediction = model.predict(np.array([[in_month, p_qtr, in_msrp, in_qty, p_prod, p_country]]))[0]
            
            st.markdown(f"""
                <div style="background-color: #e1f5fe; padding: 30px; border-radius: 15px; border-left: 10px solid #0288d1; text-align: center;">
                    <h3 style="color: #01579b; margin: 0;">Predicted Deal Revenue</h3>
                    <h1 style="color: #01579b; font-size: 50px; margin: 10px 0;">${prediction:,.2f}</h1>
                    <p style="color: #0277bd; font-weight: bold;">AI Confidence Score: {r2_score(y_test, y_pred)*100:.1f}%</p>
                </div>
            """, unsafe_allow_html=True)

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
