import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="PredictiCorp BI Suite", layout="wide", initial_sidebar_state="expanded")

# --- EXECUTIVE THEMING ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #1f4e79; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #f4f7f9; }
    .stTabs [data-baseweb="tab"] { background-color: #ffffff; border-radius: 10px 10px 0 0; border: 1px solid #e1e4e8; padding: 10px 20px; font-weight: bold; color: #5c6c7b; }
    .stTabs [aria-selected="true"] { background-color: #1f4e79 !important; color: white !important; }
    .card { background-color: #ffffff; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 8px solid #1f4e79; }
    
    .welcome-header {
        background: linear-gradient(90deg, #1f4e79 0%, #2c3e50 100%);
        color: white; padding: 60px; border-radius: 20px; text-align: center; margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    .feature-box {
        background: white; padding: 30px; border-radius: 15px; border-bottom: 4px solid #1f4e79; text-align: center; transition: transform 0.3s ease;
    }
    .feature-box:hover { transform: translateY(-10px); }
    </style>
    """, unsafe_allow_html=True)

# --- TEMPLATE GENERATOR ---
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

template_df = pd.DataFrame(columns=['ORDERNUMBER', 'QUANTITYORDERED', 'PRICEEACH', 'SALES', 'ORDERDATE', 'STATUS', 'QTR_ID', 'MONTH_ID', 'YEAR_ID', 'PRODUCTLINE', 'MSRP', 'PRODUCTCODE', 'CUSTOMERNAME', 'COUNTRY', 'TERRITORY', 'DEALSIZE'])
csv_template = convert_df_to_csv(template_df)

# --- SIDEBAR ---
st.sidebar.title("🏢 BI Command Center")
st.sidebar.download_button(label="📥 Download CSV Template", data=csv_template, file_name="sales_data_template.csv", mime="text/csv")
st.sidebar.divider()
uploaded_file = st.sidebar.file_uploader("Upload Sales Data (CSV)", type=["csv"])

if uploaded_file is not None:
    @st.cache_data
    def load_and_process_data(file):
        df = pd.read_csv(file)
        if 'ORDERDATE' in df.columns:
            df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
            df['YEAR'] = df['ORDERDATE'].dt.year
            df['MONTH_NAME'] = df['ORDERDATE'].dt.month_name()
        elif 'YEAR_ID' in df.columns:
            df['YEAR'] = df['YEAR_ID']
        return df

    df_master = load_and_process_data(uploaded_file)
    
    # --- DYNAMIC FILTERS ---
    st.sidebar.subheader("Filter Dashboard View")
    st_year = st.sidebar.multiselect("Fiscal Year", options=sorted(df_master['YEAR'].unique()), default=df_master['YEAR'].unique())
    st_country = st.sidebar.multiselect("Active Markets (Venues)", options=sorted(df_master['COUNTRY'].unique()), default=df_master['COUNTRY'].unique())
    st_items = st.sidebar.multiselect("Specific Items (Product Lines)", options=sorted(df_master['PRODUCTLINE'].unique()), default=df_master['PRODUCTLINE'].unique())
    
    # Global Filtered Dataframe
    df = df_master[
        (df_master['YEAR'].isin(st_year)) & 
        (df_master['COUNTRY'].isin(st_country)) & 
        (df_master['PRODUCTLINE'].isin(st_items))
    ]

    @st.cache_resource
    def train_bi_model(data):
        features = ['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']
        X, y = data[features], data['SALES']
        pipe = Pipeline(steps=[
            ('pre', ColumnTransformer([('cat', OneHotEncoder(handle_unknown='ignore'), ['PRODUCTLINE', 'COUNTRY'])], remainder='passthrough')),
            ('reg', RandomForestRegressor(n_estimators=100, random_state=42))
        ]).fit(X, y)
        return pipe, r2_score(y, pipe.predict(X)) * 100

    bi_pipe, ai_score = train_bi_model(df_master)

    tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Strategic Market Insights", "📅 Demand Forecast", "👥 Customer Analytics"])

    if df.empty:
        st.warning("⚠️ No data available for the current selection. Please adjust your filters in the sidebar.")
    else:
        # --- TAB 1: EXECUTIVE DASHBOARD ---
        with tabs[0]:
            st.subheader("Performance KPIs")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Revenue", f"${df['SALES'].sum()/1e6:.2f}M")
            k2.metric("Avg Order Value", f"${df['SALES'].mean():,.2f}")
            k3.metric("Transaction Volume", f"{len(df):,}")
            k4.metric("Active Regions", f"{df['COUNTRY'].nunique()}")
            st.markdown("---")
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("#### Monthly Sales Trend")
                trend = df.groupby(['YEAR', 'MONTH_ID', 'MONTH_NAME'])['SALES'].sum().reset_index().sort_values(['YEAR', 'MONTH_ID'])
                fig_trend = px.line(trend, x='MONTH_NAME', y='SALES', color='YEAR', markers=True, template="plotly_white")
                st.plotly_chart(fig_trend, use_container_width=True)
            with c2:
                st.markdown("#### Product Line Distribution")
                fig_pie = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # SPECIFIC VENUE BY SPECIFIC ITEM CHART
            st.markdown("#### 📍 Venue & Item Analysis")
            col_bar1, col_bar2 = st.columns(2)
            with col_bar1:
                country_rev = df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
                fig_bar_country = px.bar(country_rev, x='COUNTRY', y='SALES', text_auto='.2s', title="Revenue by Venue", color='SALES', template="plotly_white")
                st.plotly_chart(fig_bar_country, use_container_width=True)
            with col_bar2:
                item_rev = df.groupby('PRODUCTLINE')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
                fig_bar_item = px.bar(item_rev, x='SALES', y='PRODUCTLINE', orientation='h', title="Revenue by Item", color='SALES', template="plotly_white")
                st.plotly_chart(fig_bar_item, use_container_width=True)

            st.markdown("#### 🔍 Variance & Outlier Detection")
            fig_box = px.box(df, x='COUNTRY', y='SALES', color='PRODUCTLINE', template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)

        # --- TAB 2: SIMULATOR ---
        with tabs[1]:
            st.header("🔮 Strategic Scenario Simulator")
            col1, col2, col3 = st.columns(3)
            in_country = col1.selectbox("Target Market (Country)", sorted(df_master['COUNTRY'].unique()))
            valid_products = df_master[df_master['COUNTRY'] == in_country]['PRODUCTLINE'].unique()
            in_prod = col2.selectbox(f"Available Products in {in_country}", valid_products)
            
            ref_data = df_master[df_master['PRODUCTLINE'] == in_prod]
            avg_msrp = float(ref_data['MSRP'].mean())
            st.info(f"💡 Historical Avg Price for {in_prod}: ${avg_msrp:.2f}")
            
            in_qty = col1.slider("Quantity to Sell", 1, 500, 50)
            in_msrp = col2.number_input("Unit Price ($)", value=int(avg_msrp))
            in_month = col3.slider("Order Month", 1, 12, 12)
            
            if st.button("RUN AI SIMULATION", use_container_width=True, type="primary"):
                inp = pd.DataFrame([{'MONTH_ID': in_month, 'QTR_ID': (in_month-1)//3+1, 'MSRP': in_msrp, 'QUANTITYORDERED': in_qty, 'PRODUCTLINE': in_prod, 'COUNTRY': in_country}])
                pred = bi_pipe.predict(inp)[0]
                st.markdown(f"<div style='background-color:#e3f2fd;padding:30px;border-radius:15px;text-align:center;border: 2px solid #1f4e79;'><h1>${pred:,.2f}</h1><p>Projected Revenue</p></div>", unsafe_allow_html=True)

        # --- TAB 3: MARKET INSIGHTS ---
        with tabs[2]:
            st.header("💡 Business Directives")
            geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
            fig_map = px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="SALES", hover_name="COUNTRY", color_continuous_scale='Blues', template="plotly_white")
            st.plotly_chart(fig_map, use_container_width=True)

        # --- TAB 4: DEMAND FORECAST ---
        with tabs[3]:
            st.header("📅 Demand Forecasting")
            forecast_df = df.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
            forecast_df['Target_Forecast'] = forecast_df['SALES'].rolling(window=3).mean().shift(-1)
            fig_forecast = px.line(forecast_df, x='MONTH_ID', y=['SALES', 'Target_Forecast'], markers=True, template="plotly_white", title="3-Month Momentum")
            st.plotly_chart(fig_forecast, use_container_width=True)

        # --- TAB 5: CUSTOMER ANALYTICS ---
        with tabs[4]:
            st.header("👥 Customer Loyalty")
            cust_val = df.groupby('CUSTOMERNAME')['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(10)
            st.plotly_chart(px.bar(cust_val, x='SALES', y='CUSTOMERNAME', orientation='h', template="plotly_white"), use_container_width=True)

else:
    st.markdown("""<div class="welcome-header"><h1>🚀 Welcome to PredictiCorp Intelligence</h1><p>The Global Executive Suite for Data-Driven Market Strategy</p></div>""", unsafe_allow_html=True)
    st.info("👈 Please upload your Sales Data CSV in the sidebar to activate insights.")
