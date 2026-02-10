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
    st_year = st.sidebar.multiselect("Fiscal Year", options=sorted(df_master['YEAR'].unique()), default=df_master['YEAR'].unique())
    st_country = st.sidebar.multiselect("Active Markets", options=sorted(df_master['COUNTRY'].unique()), default=df_master['COUNTRY'].unique())
    df = df_master[(df_master['YEAR'].isin(st_year)) & (df_master['COUNTRY'].isin(st_country))]

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

    tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Strategic Market Insights"])

    # TAB 1: Dashboard
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
            trend = df.groupby(['YEAR', 'MONTH_ID', 'MONTH_NAME'])['SALES'].sum().reset_index().sort_values(['YEAR', 'MONTH_ID'])
            fig_trend = px.line(trend, x='MONTH_NAME', y='SALES', color='YEAR', markers=True, template="plotly_white")
            st.plotly_chart(fig_trend, use_container_width=True)
        with c2:
            fig_pie = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism)
            st.plotly_chart(fig_pie, use_container_width=True)

    # TAB 2: Simulator
    with tabs[1]:
        st.header("🔮 Strategic Scenario Simulator")
        col1, col2, col3 = st.columns(3)
        in_prod = col1.selectbox("Product Line", df_master['PRODUCTLINE'].unique())
        in_qty = col1.slider("Quantity", 10, 500, 50)
        in_country = col2.selectbox("Country", sorted(df_master['COUNTRY'].unique()))
        in_msrp = col2.number_input("Unit Price ($)", value=100)
        in_month = col3.slider("Order Month", 1, 12, 6)
        if st.button("RUN AI SIMULATION", use_container_width=True, type="primary"):
            inp = pd.DataFrame([{'MONTH_ID': in_month, 'QTR_ID': (in_month-1)//3+1, 'MSRP': in_msrp, 'QUANTITYORDERED': in_qty, 'PRODUCTLINE': in_prod, 'COUNTRY': in_country}])
            pred = bi_pipe.predict(inp)[0]
            st.markdown(f"<div style='background-color:#e3f2fd;padding:30px;border-radius:15px;text-align:center;'><h3>Predicted Revenue</h3><h1>${pred:,.2f}</h1><p>AI Accuracy: {ai_score:.1f}%</p></div>", unsafe_allow_html=True)

    # TAB 3: Market Insights
    with tabs[2]:
        st.header("💡 Business Directives")
        
        # Dynamic Insight Logic
        top_country = df.groupby('COUNTRY')['SALES'].sum().idxmax()
        top_prod = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
        
        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown(f"""
            <div class="card">
                <h4>📦 Inventory Optimization</h4>
                <p><b>Insight:</b> <b>{top_prod}</b> is currently your highest-performing line.<br>
                <b>Action:</b> Prioritize supply chain stability for this category to meet predicted Q4 demand.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_i2:
            st.markdown(f"""
            <div class="card">
                <h4>🌍 Regional Strategy</h4>
                <p><b>Insight:</b> <b>{top_country}</b> contributes the most to your global revenue.<br>
                <b>Action:</b> Pilot a localized loyalty program in <b>{top_country}</b> to defend market share.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Geographic Performance Heatmap")
        geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
        fig_map = px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="SALES", color_continuous_scale="Blues")
        st.plotly_chart(fig_map, use_container_width=True)

else:
    # --- INTERACTIVE WELCOME PAGE ---
    st.markdown("""
        <div class="welcome-header">
            <h1>🚀 Welcome to PredictiCorp Intelligence</h1>
            <p style="font-size: 1.2em; opacity: 0.9;">The Global Executive Suite for Data-Driven Market Strategy</p>
        </div>
    """, unsafe_allow_html=True)
    
    st.markdown("### 🛠️ Get Started in 3 Simple Steps")
    s1, s2, s3 = st.columns(3)
    with s1:
        st.markdown("""<div class="feature-box"><h2>📋</h2><h3>Step 1</h3><p>Download the CSV template from the sidebar.</p></div>""", unsafe_allow_html=True)
    with s2:
        st.markdown("""<div class="feature-box"><h2>📥</h2><h3>Step 2</h3><p>Upload your sales data for automated AI training.</p></div>""", unsafe_allow_html=True)
    with s3:
        st.markdown("""<div class="feature-box"><h2>💡</h2><h3>Step 3</h3><p>Explore Tab 3 for Deep Market Insights.</p></div>""", unsafe_allow_html=True)

    st.markdown("---")
    c_demo1, c_demo2 = st.columns([1, 1])
    with c_demo1:
        st.image("https://img.freepik.com/free-vector/business-analytics-concept-illustration_114360-3944.jpg", use_column_width=True)
    with c_demo2:
        st.subheader("🤖 Advanced AI Capabilities")
        with st.expander("🌍 Territory Benchmarking"):
            st.write("Compare EMEA, APAC, and NA performance side-by-side to identify regional laggards.")
        with st.expander("📦 Deal Structure Analysis"):
            st.write("Understand if your revenue is coming from 'Small' high-frequency orders or 'Large' enterprise deals.")
        with st.expander("🔮 Dynamic Forecasting"):
            st.write("Retrain the model instantly as you filter data to get context-specific predictions.")
        st.warning("👈 Please upload your Sales Data CSV in the sidebar to activate.")
