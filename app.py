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

# --- ML PIPELINE FUNCTION ---
def train_dynamic_model(data, features, target):
    X = data[features]
    y = data[target]
    
    # Identify categorical vs numerical features
    cat_features = X.select_dtypes(include=['object']).columns.tolist()
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_features)
        ], remainder='passthrough')

    model_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('regressor', RandomForestRegressor(n_estimators=100, random_state=42))
    ])
    
    model_pipeline.fit(X, y)
    score = r2_score(y, model_pipeline.predict(X)) * 100
    return model_pipeline, score

# --- WELCOME & UPLOAD ---
st.title("🚀 Universal Executive Intelligence Suite")
st.markdown("Upload your business transaction data (CSV) to generate instant AI insights.")

uploaded_file = st.sidebar.file_uploader("Step 1: Upload Business Data", type="csv")

if uploaded_file is not None:
    # Load Data
    df_master = pd.read_csv(uploaded_file)
    
    # Clean Dates if present
    date_col = next((col for col in df_master.columns if 'date' in col.lower()), None)
    if date_col:
        df_master[date_col] = pd.to_datetime(df_master[date_col])
        df_master['YEAR'] = df_master[date_col].dt.year
        df_master['MONTH_NAME'] = df_master[date_col].dt.month_name()
    
    # Identify Core Columns
    target_col = next((col for col in df_master.columns if 'sales' in col.lower() or 'revenue' in col.lower()), None)
    cat_col = next((col for col in df_master.columns if 'product' in col.lower() or 'category' in col.lower()), None)
    geo_col = next((col for col in df_master.columns if 'country' in col.lower() or 'region' in col.lower()), None)

    if not target_col:
        st.error("Could not find a 'Sales' or 'Revenue' column. Please check your CSV.")
    else:
        # --- SIDEBAR FILTERS ---
        st.sidebar.divider()
        st.sidebar.subheader("Strategic Filters")
        
        # Filter by Year if possible
        if 'YEAR' in df_master.columns:
            st_year = st.sidebar.multiselect("Fiscal Year", options=sorted(df_master['YEAR'].unique()), default=df_master['YEAR'].unique())
            df = df_master[df_master['YEAR'].isin(st_year)]
        else:
            df = df_master

        # Filter by Country/Region if possible
        if geo_col:
            st_country = st.sidebar.multiselect("Active Markets", options=sorted(df[geo_col].unique()), default=df[geo_col].unique())
            df = df[df[geo_col].isin(st_country)]

        # --- MODEL TRAINING ---
        # Select features for the AI
        potential_features = ['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', cat_col, geo_col]
        available_features = [f for f in potential_features if f in df_master.columns]
        
        bi_pipe, ai_score = train_dynamic_model(df_master, available_features, target_col)

        # --- APP LAYOUT ---
        tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Market Insights"])

        with tabs[0]:
            st.subheader("Performance KPIs")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Revenue", f"${df[target_col].sum():,.2f}")
            k2.metric("Avg Transaction", f"${df[target_col].mean():,.2f}")
            k3.metric("Volume", f"{len(df):,}")
            if geo_col:
                k4.metric("Active Regions", f"{df[geo_col].nunique()}")

            st.markdown("---")
            c1, c2 = st.columns([2, 1])
            with c1:
                if 'MONTH_NAME' in df.columns:
                    trend = df.groupby(['MONTH_NAME'])[target_col].sum().reset_index()
                    fig_trend = px.line(trend, x='MONTH_NAME', y=target_col, title="Revenue Momentum", template="plotly_white")
                    st.plotly_chart(fig_trend, use_container_width=True)
            with c2:
                if cat_col:
                    fig_pie = px.pie(df, values=target_col, names=cat_col, hole=0.5, title="Portfolio Composition")
                    st.plotly_chart(fig_pie, use_container_width=True)

        with tabs[1]:
            st.header("🔮 AI Strategic Simulator")
            col1, col2 = st.columns(2)
            
            with col1:
                inputs = {}
                for feat in available_features:
                    if df_master[feat].dtype == 'object':
                        inputs[feat] = st.selectbox(f"Select {feat}", df_master[feat].unique())
                    else:
                        inputs[feat] = st.number_input(f"Enter {feat}", value=int(df_master[feat].median()))
            
            with col2:
                if st.button("RUN AI SIMULATION", use_container_width=True, type="primary"):
                    input_df = pd.DataFrame([inputs])
                    prediction = bi_pipe.predict(input_df)[0]
                    st.markdown(f"""
                        <div style="background-color:#e3f2fd;padding:30px;border-radius:15px;text-align:center;">
                            <h3>Predicted Outcome</h3>
                            <h1>${prediction:,.2f}</h1>
                            <p><b>Model Accuracy:</b> {ai_score:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)

        with tabs[2]:
            st.header("🌍 Geographic Performance")
            if geo_col:
                geo_df = df.groupby(geo_col)[target_col].sum().reset_index()
                fig_map = px.choropleth(geo_df, locations=geo_col, locationmode='country names', color=target_col, color_continuous_scale="Blues")
                st.plotly_chart(fig_map, use_container_width=True)
            else:
                st.write("No geographic data found in the file.")

else:
    st.info("👋 Welcome! Please upload a CSV file in the sidebar to begin the analysis.")
    # You could add a 'Download Sample CSV' button here too.
