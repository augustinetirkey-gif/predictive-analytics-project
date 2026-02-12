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
import io

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
        
        # Fill missing values in categorical columns to prevent filter errors
        categorical_cols = ['TERRITORY', 'COUNTRY', 'PRODUCTLINE', 'DEALSIZE']
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].fillna("Unknown")
        return df

    df_master = load_and_process_data(uploaded_file)

    # --- UPDATED SIDEBAR FILTERS (FIXED WITH DROPNA & UNIQUE) ---
    st.sidebar.subheader("🎚️ Strategic Filters")
    
    # Helper function to get clean, sorted unique values
    def get_options(column):
        return sorted(df_master[column].unique().tolist())

    st_year = st.sidebar.multiselect("Fiscal Year", options=get_options('YEAR'), default=get_options('YEAR'))
    st_qtr = st.sidebar.multiselect("Quarter", options=get_options('QTR_ID'), default=get_options('QTR_ID'))
    st_month = st.sidebar.multiselect("Month", options=get_options('MONTH_ID'), default=get_options('MONTH_ID'))
    st_country = st.sidebar.multiselect("Active Markets", options=get_options('COUNTRY'), default=get_options('COUNTRY'))
    st_territory = st.sidebar.multiselect("Territory", options=get_options('TERRITORY'), default=get_options('TERRITORY'))
    st_product = st.sidebar.multiselect("Product Line", options=get_options('PRODUCTLINE'), default=get_options('PRODUCTLINE'))
    st_dealsize = st.sidebar.multiselect("Deal Size", options=get_options('DEALSIZE'), default=get_options('DEALSIZE'))
    
    # Revenue Range Filter (Handled with float conversion for reliability)
    min_val = float(df_master['SALES'].min())
    max_val = float(df_master['SALES'].max())
    st_rev_range = st.sidebar.slider("Revenue Range ($)", min_val, max_val, (min_val, max_val))

    # Apply All Filters to df
    df = df_master[
        (df_master['YEAR'].isin(st_year)) & 
        (df_master['QTR_ID'].isin(st_qtr)) &
        (df_master['MONTH_ID'].isin(st_month)) &
        (df_master['COUNTRY'].isin(st_country)) & 
        (df_master['TERRITORY'].isin(st_territory)) &
        (df_master['PRODUCTLINE'].isin(st_product)) &
        (df_master['DEALSIZE'].isin(st_dealsize)) &
        (df_master['SALES'].between(st_rev_range[0], st_rev_range[1]))
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
        st.warning("⚠️ No data available for the current selection. Please adjust your filters.")
    else:
        # --- TAB 1: UPDATED EXECUTIVE DASHBOARD ---
        with tabs[0]:
            total_rev = df['SALES'].sum()
            if len(st_year) == 1:
                prev_year_rev = df_master[df_master['YEAR'] == (st_year[0] - 1)]['SALES'].sum()
                growth = ((total_rev - prev_year_rev) / prev_year_rev * 100) if prev_year_rev > 0 else 0
            else:
                growth = 0

            st.subheader("Performance KPIs")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Revenue", f"${total_rev/1e6:.2f}M", f"{growth:.1f}% vs Prev" if growth != 0 else None)
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
                st.markdown("#### Revenue by Product")
                fig_prod = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism)
                st.plotly_chart(fig_prod, use_container_width=True)
            
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("#### Revenue by Territory")
                terr_rev = df.groupby('TERRITORY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
                fig_terr = px.bar(terr_rev, x='TERRITORY', y='SALES', color='SALES', template="plotly_white")
                st.plotly_chart(fig_terr, use_container_width=True)
            with c4:
                st.markdown("#### Revenue by Deal Size")
                deal_rev = df.groupby('DEALSIZE')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
                fig_deal = px.bar(deal_rev, x='DEALSIZE', y='SALES', color='DEALSIZE', template="plotly_white")
                st.plotly_chart(fig_deal, use_container_width=True)

            c5, c6 = st.columns(2)
            with c5:
                st.markdown("#### Top 10 High-Value Customers")
                top_cust = df.groupby('CUSTOMERNAME')['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(10)
                fig_cust = px.bar(top_cust, x='SALES', y='CUSTOMERNAME', orientation='h', template="plotly_white", color='SALES')
                st.plotly_chart(fig_cust, use_container_width=True)
            with c6:
                st.markdown("#### 🔍 Sales Outlier Detection")
                fig_box = px.box(df, x='PRODUCTLINE', y='SALES', color='PRODUCTLINE', template="plotly_white")
                st.plotly_chart(fig_box, use_container_width=True)

        # TAB 2: Simulator
        with tabs[1]:
            st.header("🔮 Strategic Scenario Simulator")
            col1, col2, col3 = st.columns(3)
            in_country = col1.selectbox("Target Market (Country)", sorted(df_master['COUNTRY'].unique()))
            valid_products = df_master[df_master['COUNTRY'] == in_country]['PRODUCTLINE'].unique()
            in_prod = col2.selectbox(f"Available Products in {in_country}", valid_products)
            
            ref_data = df_master[df_master['PRODUCTLINE'] == in_prod]
            avg_msrp = float(ref_data['MSRP'].mean()) if not ref_data.empty else 100.0
            
            in_qty = col1.slider("Quantity to Sell", 1, 500, 50)
            in_msrp = col2.number_input("Unit Price ($)", value=int(avg_msrp))
            in_month = col3.slider("Order Month", 1, 12, 12)
            
            if st.button("RUN AI SIMULATION & REALITY CHECK", use_container_width=True, type="primary"):
                inp = pd.DataFrame([{'MONTH_ID': in_month, 'QTR_ID': (in_month-1)//3+1, 'MSRP': in_msrp, 'QUANTITYORDERED': in_qty, 'PRODUCTLINE': in_prod, 'COUNTRY': in_country}])
                pred = bi_pipe.predict(inp)[0]
                st.metric("PROJECTED REVENUE", f"${pred:,.2f}")

        # TAB 3: Market Insights
        with tabs[2]:
            st.header("💡 Business Directives")
            top_country = df.groupby('COUNTRY')['SALES'].sum().idxmax()
            top_prod = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
            
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                st.info(f"**Inventory Optimization:** Prioritize **{top_prod}** line.")
            with col_i2:
                st.success(f"**Regional Strategy:** **{top_country}** is the priority market.")

            geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
            fig_map = px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="SALES", template="plotly_white")
            st.plotly_chart(fig_map, use_container_width=True)

        # TAB 4: DEMAND FORECASTING
        with tabs[3]:
            st.header("📅 Demand Forecasting")
            forecast_df = df.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
            forecast_df['Target_Forecast'] = forecast_df['SALES'].rolling(window=3).mean().shift(-1)
            fig_forecast = px.line(forecast_df, x='MONTH_ID', y=['SALES', 'Target_Forecast'], markers=True, template="plotly_white")
            st.plotly_chart(fig_forecast, use_container_width=True)

        # TAB 5: CUSTOMER ANALYTICS
        with tabs[4]:
            st.header("👥 Customer Loyalty")
            cust_val = df.groupby('CUSTOMERNAME')['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(10)
            st.plotly_chart(px.bar(cust_val, x='SALES', y='CUSTOMERNAME', orientation='h', template="plotly_white"), use_container_width=True)

else:
    st.markdown("""<div class="welcome-header"><h1>🚀 Welcome to PredictiCorp Intelligence</h1><p>The Global Executive Suite for Data-Driven Market Strategy</p></div>""", unsafe_allow_html=True)
    st.info("👈 Please upload your Sales Data CSV in the sidebar to activate insights.")
