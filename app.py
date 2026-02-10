import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="Universal BI Intelligence", layout="wide")

# --- EXECUTIVE THEMING ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    .stMetric { background-color: #ffffff; padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.05); border-top: 5px solid #1f4e79; }
    .card { background-color: #ffffff; padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.05); margin-bottom: 20px; border-left: 8px solid #1f4e79; }
    </style>
    """, unsafe_allow_html=True)

# --- HELPER: DYNAMIC MODEL TRAINING ---
def train_dynamic_model(data, features, target):
    X = data[features].copy()
    y = data[target]
    
    # Auto-detect categorical columns for encoding
    cat_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(handle_unknown='ignore'), cat_cols)], 
        remainder='passthrough'
    )

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model_pipeline.fit(X, y)
    score = r2_score(y, model_pipeline.predict(X)) * 100
    return model_pipeline, score

# --- APP START ---
st.title("🚀 Universal Executive Intelligence Suite")
st.caption("Upload any sales dataset to generate instant AI insights and simulations.")

# 1. FILE UPLOADER
uploaded_file = st.sidebar.file_uploader("📥 Step 1: Upload Business Data (CSV)", type="csv")

if uploaded_file is not None:
    # Load and process data
    df_master = pd.read_csv(uploaded_file)
    
    # Auto-detect standard columns (Flexible for different naming conventions)
    target_col = next((c for c in df_master.columns if 'sales' in c.lower() or 'revenue' in c.lower()), None)
    date_col = next((c for c in df_master.columns if 'date' in c.lower()), None)
    cat_col = next((c for c in df_master.columns if 'product' in c.lower() or 'line' in c.lower()), None)
    geo_col = next((c for c in df_master.columns if 'country' in c.lower()), None)

    if not target_col:
        st.error("Error: Could not find a 'Sales' or 'Revenue' column in your file.")
    else:
        # Pre-process dates if they exist
        if date_col:
            df_master[date_col] = pd.to_datetime(df_master[date_col])
            df_master['YEAR'] = df_master[date_col].dt.year
            df_master['MONTH_NAME'] = df_master[date_col].dt.month_name()
            df_master['MONTH_ID'] = df_master[date_col].dt.month

        # --- SIDEBAR FILTERS ---
        st.sidebar.divider()
        st.sidebar.subheader("Strategic Filtering")
        
        years = sorted(df_master['YEAR'].unique()) if 'YEAR' in df_master.columns else [2024]
        st_year = st.sidebar.multiselect("Fiscal Year", options=years, default=years)
        
        countries = sorted(df_master[geo_col].unique()) if geo_col else ["Global"]
        st_country = st.sidebar.multiselect("Active Markets", options=countries, default=countries)

        # Filtered DataFrame
        df = df_master.copy()
        if 'YEAR' in df.columns: df = df[df['YEAR'].isin(st_year)]
        if geo_col: df = df[df[geo_col].isin(st_country)]

        # --- TABS ---
        tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Market Insights"])

        # TAB 1: EXECUTIVE DASHBOARD
        with tabs[0]:
            st.subheader("Performance KPIs")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Revenue", f"${df[target_col].sum()/1e6:.2f}M")
            k2.metric("Avg Order Value", f"${df[target_col].mean():,.2f}")
            k3.metric("Volume", f"{len(df):,}")
            k4.metric("Active Regions", f"{df[geo_col].nunique() if geo_col else 1}")

            st.markdown("---")
            c1, c2 = st.columns([2, 1])
            with c1:
                if date_col:
                    trend = df.groupby(['YEAR', 'MONTH_ID', 'MONTH_NAME'])[target_col].sum().reset_index().sort_values(['YEAR', 'MONTH_ID'])
                    fig_trend = px.line(trend, x='MONTH_NAME', y=target_col, color='YEAR', markers=True, template="plotly_white", title="Revenue Momentum")
                    st.plotly_chart(fig_trend, use_container_width=True)
            with c2:
                if cat_col:
                    fig_pie = px.pie(df, values=target_col, names=cat_col, hole=0.5, title="Portfolio Composition")
                    st.plotly_chart(fig_pie, use_container_width=True)

        # TAB 2: REVENUE SIMULATOR
        with tabs[1]:
            st.header("🔮 Strategic Scenario Simulator")
            # Automatically pick features that exist in the data
            sim_features = [f for f in ['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', cat_col, geo_col] if f in df_master.columns]
            
            # Train model only on these columns
            bi_pipe, ai_score = train_dynamic_model(df_master, sim_features, target_col)

            with st.container():
                cols = st.columns(len(sim_features))
                user_inputs = {}
                for i, feat in enumerate(sim_features):
                    if df_master[feat].dtype == 'object':
                        user_inputs[feat] = cols[i].selectbox(f"{feat}", df_master[feat].unique())
                    else:
                        user_inputs[feat] = cols[i].number_input(f"{feat}", value=int(df_master[feat].median()))

                if st.button("RUN AI SIMULATION", use_container_width=True, type="primary"):
                    input_df = pd.DataFrame([user_inputs])
                    prediction = bi_pipe.predict(input_df)[0]
                    st.markdown(f"""
                        <div style="background-color:#e3f2fd;padding:30px;border-radius:15px;text-align:center;">
                            <h3>Predicted Revenue Output</h3>
                            <h1>${prediction:,.2f}</h1>
                            <p><b>AI System Confidence:</b> {ai_score:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

        # TAB 3: MARKET INSIGHTS
        with tabs[2]:
            st.header("💡 Data-Driven Directives")
            
            # Logic-based Insights
            top_country = df.groupby(geo_col)[target_col].sum().idxmax() if geo_col else "N/A"
            top_prod = df.groupby(cat_col)[target_col].sum().idxmax() if cat_col else "N/A"
            
            i1, i2 = st.columns(2)
            i1.markdown(f"""<div class="card"><h4>🌍 Regional Strategy</h4>
                <p><b>Insight:</b> {top_country} is your primary revenue driver in the current selection.<br>
                <b>Action:</b> Focus marketing spend on {top_country} to maximize ROI.</p></div>""", unsafe_allow_html=True)
            
            i2.markdown(f"""<div class="card"><h4>📦 Inventory Optimization</h4>
                <p><b>Insight:</b> {top_prod} represents your strongest product line.<br>
                <b>Action:</b> Ensure safety stock for {top_prod} is increased by 15% for the upcoming quarter.</p></div>""", unsafe_allow_html=True)

            st.markdown("### Geographic Performance Heatmap")
            if geo_col:
                geo_df = df.groupby(geo_col)[target_col].sum().reset_index()
                fig_map = px.choropleth(geo_df, locations=geo_col, locationmode='country names', color=target_col, color_continuous_scale="Blues")
                st.plotly_chart(fig_map, use_container_width=True)

else:
    st.info("👋 Welcome to the BI Suite. Please upload a CSV file in the sidebar to begin.")
