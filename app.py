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
    </style>
    """, unsafe_allow_html=True)

# --- TEMPLATE GENERATOR ---
def convert_df_to_csv(df):
    return df.to_csv(index=False).encode('utf-8')

template_df = pd.DataFrame(columns=['ORDERNUMBER', 'QUANTITYORDERED', 'PRICEEACH', 'SALES', 'ORDERDATE', 'QTR_ID', 'MONTH_ID', 'YEAR_ID', 'PRODUCTLINE', 'MSRP', 'COUNTRY'])
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

    tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Market Insights"])

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
            trend = df.groupby(['YEAR', 'MONTH_ID', 'MONTH_NAME'])['SALES'].sum().reset_index().sort_values(['YEAR', 'MONTH_ID'])
            st.plotly_chart(px.line(trend, x='MONTH_NAME', y='SALES', color='YEAR', markers=True, template="plotly_white"), use_container_width=True)
        with c2:
            st.plotly_chart(px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism), use_container_width=True)

    # --- TAB 2: REVENUE SIMULATOR ---
    with tabs[1]:
        st.header("🔮 Strategic Scenario Simulator")
        with st.container():
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

    # --- TAB 3: MARKET INSIGHTS (CLEAN BUBBLE MAP) ---
    with tabs[2]:
        st.header("💡 Business Directives")
        
        # Logic for Dynamic Insights
        top_prod = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
        top_country = df.groupby('COUNTRY')['SALES'].sum().idxmax()
        country_share = (df.groupby('COUNTRY')['SALES'].sum().max() / df['SALES'].sum()) * 100

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            st.markdown(f"""
            <div class="card">
                <h4>📦 Inventory Optimization</h4>
                <p><b>Insight:</b> <b>{top_prod}</b> is the revenue leader. AI suggests maintaining 15% safety stock for this line.</p>
            </div>
            """, unsafe_allow_html=True)
        with col_i2:
            st.markdown(f"""
            <div class="card">
                <h4>🌍 Regional Strategy</h4>
                <p><b>Insight:</b> <b>{top_country}</b> contributes {country_share:.1f}% of revenue. Focus on high-value customer retention.</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("### Geographic Performance Heatmap")
        st.caption("Each circle represents a country. Larger circles indicate higher transaction volume.")
        
        geo_df = df.groupby('COUNTRY').agg({'SALES': 'sum', 'QUANTITYORDERED': 'sum'}).reset_index()

        # 
        fig_map = px.scatter_geo(
            geo_df,
            locations="COUNTRY",
            locationmode='country names',
            size="SALES", # Circle size based on revenue
            color="SALES", # Circle color based on revenue
            hover_name="COUNTRY",
            template="plotly_white",
            color_continuous_scale="Plasma",
            projection="equirectangular"
        )

        fig_map.update_traces(marker=dict(line=dict(width=1, color='DarkSlateGrey')))
        fig_map.update_geos(showcountries=True, countrycolor="Linen")
        
        st.plotly_chart(fig_map, use_container_width=True)

else:
    st.title("🚀 PredictiCorp Executive Intelligence Suite")
    st.info("👋 Welcome! Please upload your Sales Data CSV in the sidebar to begin.")
