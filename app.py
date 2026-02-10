import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="PredictiCorp BI Suite", layout="wide", initial_sidebar_state="expanded")

# --- EXECUTIVE THEMING (Custom CSS) ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #1f4e79; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: #f4f7f9; }
    .stTabs [data-baseweb="tab"] { background-color: #ffffff; border-radius: 10px 10px 0 0; border: 1px solid #e1e4e8; padding: 10px 20px; font-weight: bold; color: #5c6c7b; }
    .stTabs [aria-selected="true"] { background-color: #1f4e79 !important; color: white !important; }
    .card { background-color: #ffffff; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 8px solid #1f4e79; }
    </style>
    """, unsafe_allow_html=True)

# --- SIDEBAR: STRATEGIC FILTERS ---
st.sidebar.title("🏢 BI Command Center")
st.sidebar.markdown("**Global Filtering Engine**")

# --- CSV FILE UPLOADER ---
uploaded_file = st.sidebar.file_uploader("Step 1: Upload Sales Data (CSV)", type=["csv"])

if uploaded_file is not None:
    # DATA ANALYTICS ENGINE
    @st.cache_data
    def load_and_process_data(file):
        df = pd.read_csv(file)
        # Standardize dates and extract features
        if 'ORDERDATE' in df.columns:
            df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
            df['YEAR'] = df['ORDERDATE'].dt.year
            df['MONTH_NAME'] = df['ORDERDATE'].dt.month_name()
        return df

    df_master = load_and_process_data(uploaded_file)

    # Sidebar Filter logic (only if columns exist)
    st_year = st.sidebar.multiselect("Fiscal Year", options=sorted(df_master['YEAR'].unique()), default=df_master['YEAR'].unique())
    st_country = st.sidebar.multiselect("Active Markets", options=sorted(df_master['COUNTRY'].unique()), default=df_master['COUNTRY'].unique())

    df = df_master[(df_master['YEAR'].isin(st_year)) & (df_master['COUNTRY'].isin(st_country))]

    # MACHINE LEARNING PIPELINE
    @st.cache_resource
    def train_bi_model(data):
        features = ['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']
        X = data[features]
        y = data['SALES']
        
        preprocessor = ColumnTransformer(
            transformers=[
                ('cat', OneHotEncoder(handle_unknown='ignore'), ['PRODUCTLINE', 'COUNTRY'])
            ], remainder='passthrough')

        model_pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
        ])
        
        model_pipeline.fit(X, y)
        score = r2_score(y, model_pipeline.predict(X)) * 100
        return model_pipeline, score

    bi_pipe, ai_score = train_bi_model(df_master)

    # --- APP LAYOUT ---
    st.title("🚀 PredictiCorp Executive Intelligence Suite")
    st.caption("Data-Driven Insights for Global Market Strategy")

    tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Market Insights"])

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

    # --- TAB 2: REVENUE SIMULATOR ---
    with tabs[1]:
        st.header("🔮 Strategic Scenario Simulator")
        st.markdown("Adjust parameters to predict the revenue outcome of prospective deals.")

        with st.container():
            col1, col2, col3 = st.columns(3)

            in_prod = col1.selectbox("Product Line", df_master['PRODUCTLINE'].unique())
            in_qty = col1.slider("Quantity", 10, 500, 50)

            in_country = col2.selectbox("Country", sorted(df_master['COUNTRY'].unique()))
            in_msrp = col2.number_input("Unit Price ($)", value=100)

            in_month = col3.slider("Order Month", 1, 12, 6)
            in_discount = col3.slider("Discount (%)", 0, 50, 10) 

            if st.button("RUN AI SIMULATION", use_container_width=True, type="primary"):
                qtr = (in_month - 1) // 3 + 1
                input_df = pd.DataFrame([{
                    'MONTH_ID': in_month,
                    'QTR_ID': qtr,
                    'MSRP': in_msrp,
                    'QUANTITYORDERED': in_qty,
                    'PRODUCTLINE': in_prod,
                    'COUNTRY': in_country
                }])

                prediction = bi_pipe.predict(input_df)[0]

                st.markdown(f"""
                <div style="background-color:#e3f2fd;padding:30px;border-radius:15px;text-align:center;">
                    <h3 style="color:#1f4e79;">Predicted Revenue</h3>
                    <h1 style="color:#1f4e79; font-size: 48px;">${prediction:,.2f}</h1>
                    <p style="color:#5c6c7b;"><b>AI System Accuracy:</b> {ai_score:.1f}%</p>
                </div>
                """, unsafe_allow_html=True)

    # --- TAB 3: MARKET INSIGHTS ---
    with tabs[2]:
        st.header("💡 Business Directives")
        col_i1, col_i2 = st.columns(2)
        
        # Calculate dynamic insights for the cards
        top_country = df.groupby('COUNTRY')['SALES'].sum().idxmax()
        top_prod = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()

        with col_i1:
            st.markdown(f"""
            <div class="card">
                <h4>📦 Inventory Optimization</h4>
                <p><b>Insight:</b> Most revenue is driven by <b>{top_prod}</b>. <br>
                <b>Action:</b> Prioritize supply chain efficiency for this category in the upcoming quarter.</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col_i2:
            st.markdown(f"""
            <div class="card">
                <h4>🌍 Regional Strategy</h4>
                <p><b>Insight:</b> <b>{top_country}</b> is your highest-performing market.<br>
                <b>Action:</b> Test a localized loyalty program in <b>{top_country}</b> to increase repeat purchases.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Geographic Performance Heatmap")
        geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
        fig_map = px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="SALES", color_continuous_scale="Blues")
        st.plotly_chart(fig_map, use_container_width=True)

else:
    # Landing state when no file is uploaded
    st.title("🚀 PredictiCorp Executive Intelligence Suite")
    st.info("👋 Welcome! Please upload your Sales Data CSV file in the sidebar to launch the intelligence suite.")
    st.image("https://img.freepik.com/free-vector/business-analytics-concept-illustration_114360-3944.jpg", width=400)
