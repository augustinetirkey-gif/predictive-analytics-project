import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import xgboost as xgb
import io

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="PredictiCorp BI Suite", layout="wide", initial_sidebar_state="expanded")

# --- EXECUTIVE THEMING ---
st.markdown("""
    <style>
    .stMetric { 
        padding: 20px; 
        border-radius: 12px; 
        box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        border-radius: 10px 10px 0 0; 
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 10px 20px; 
        font-weight: bold; 
    }
    .card { 
        padding: 25px; 
        border-radius: 15px; 
        box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        margin-bottom: 20px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .welcome-header {
        background: linear-gradient(90deg, rgba(31, 78, 121, 0.9) 0%, rgba(44, 62, 80, 0.9) 100%);
        color: white; 
        padding: 60px; 
        border-radius: 20px; 
        text-align: center; 
        margin-bottom: 40px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    [data-testid="stMetricValue"] div {
        font-weight: 700;
        font-size: 2rem;
    }
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

MODEL_FEATURES = ['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']

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
    
    # --- AI FORECAST ENGINE (FOR SYNTHETIC DATA) ---
    @st.cache_resource
    def train_forecasting_ai(data):
        X = data[MODEL_FEATURES]
        y = data['SALES']
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['PRODUCTLINE', 'COUNTRY']),
            ('num', StandardScaler(), ['MONTH_ID','QTR_ID','MSRP','QUANTITYORDERED'])
        ])
        model = Pipeline(steps=[('pre', preprocessor), ('model', LinearRegression())])
        model.fit(X, y)
        return model

    forecast_ai = train_forecasting_ai(df_master)

    def generate_forecast_data(year, ref_df):
        synthetic_rows = []
        countries = ref_df['COUNTRY'].unique()
        products = ref_df['PRODUCTLINE'].unique()
        # Fallback averages for smaller country stability
        avg_msrp = ref_df['MSRP'].mean()
        avg_qty = ref_df['QUANTITYORDERED'].mean()

        for month in range(1, 13):
            for country in countries:
                for product in products:
                    synthetic_rows.append({
                        'MONTH_ID': month, 'QTR_ID': (month-1)//3+1,
                        'MSRP': avg_msrp, 'QUANTITYORDERED': avg_qty,
                        'PRODUCTLINE': product, 'COUNTRY': country, 'YEAR': year,
                        'ORDERDATE': pd.to_datetime(f"{year}-{month}-01"),
                        'MONTH_NAME': pd.to_datetime(f"{year}-{month}-01").month_name(),
                        'CUSTOMERNAME': "AI Projected Segment", 'STATUS': 'Shipped', 'DEALSIZE': 'Medium', 'PHONE': 'N/A'
                    })
        synth_df = pd.DataFrame(synthetic_rows)
        synth_df['SALES'] = forecast_ai.predict(synth_df[MODEL_FEATURES]).clip(lower=0)
        return synth_df

    # --- SIDEBAR: PREDICTION CONTROLS ---
    st.sidebar.subheader("🔮 Predictive Intelligence")
    pred_years = [2006, 2007, 2008, 2009, 2010]
    all_years = sorted(list(df_master['YEAR'].unique()) + pred_years)
    
    # Use selectbox for clean Year Swapping
    selected_year = st.sidebar.selectbox("Target Fiscal Year", options=all_years, index=all_years.index(2005) if 2005 in all_years else 0)
    
    st_country = st.sidebar.multiselect("Active Markets", options=sorted(df_master['COUNTRY'].unique()), default=df_master['COUNTRY'].unique())
    st_product = st.sidebar.multiselect("Product Line", options=sorted(df_master['PRODUCTLINE'].unique()), default=df_master['PRODUCTLINE'].unique())

    # --- THE DATA SWAP LOGIC ---
    max_real_year = int(df_master['YEAR'].max())

    if selected_year < max_real_year:
        # PURE HISTORY
        df = df_master[df_master['YEAR'] == selected_year]
    elif selected_year == max_real_year:
        # HYBRID YEAR (e.g. 2005) - Real Data + Fill Remaining
        real_data = df_master[df_master['YEAR'] == selected_year]
        months_we_have = real_data['MONTH_ID'].unique()
        if len(months_we_have) < 12:
            gap_fill_df = generate_forecast_data(selected_year, df_master)
            missing_months = gap_fill_df[~gap_fill_df['MONTH_ID'].isin(months_we_have)]
            df = pd.concat([real_data, missing_months])
        else:
            df = real_data
    else:
        # PURE PREDICTION (2006-2010)
        df = generate_forecast_data(selected_year, df_master)

    # Re-apply global filters
    df = df[(df['COUNTRY'].isin(st_country)) & (df['PRODUCTLINE'].isin(st_product))]

    # Standard model training for Simulator
    @st.cache_resource
    def train_models(data):
        X = data[MODEL_FEATURES]
        y = data['SALES']
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['PRODUCTLINE', 'COUNTRY']),
            ('num', StandardScaler(), ['MONTH_ID','QTR_ID','MSRP','QUANTITYORDERED'])
        ])
        models = {"Linear Regression": LinearRegression(), "Random Forest": RandomForestRegressor(n_estimators=100)}
        results = {}
        for name, model in models.items():
            pipe = Pipeline(steps=[('pre', preprocessor), ('model', model)])
            pipe.fit(X, y)
            results[name] = (pipe, r2_score(y, pipe.predict(X)) * 100)
        return results

    trained_models = train_models(df_master)

    # --- LAYOUT REMAINS SAME ---
    tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Strategic Market Insights", "📅 Demand Forecast", "👥 Customer Analytics"])

    if df.empty:
        st.warning("⚠️ No data available for the current selection.")
    else:
        # --- TAB 1: EXECUTIVE DASHBOARD ---
        with tabs[0]:
            st.subheader(f"Performance Metrics: {selected_year}")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Revenue", f"${df['SALES'].sum()/1e6:.2f}M")
            k2.metric("Avg Order Value", f"${df['SALES'].mean():,.2f}")
            k3.metric("Transaction Volume", f"{len(df):,}")
            k4.metric("Active Regions", f"{df['COUNTRY'].nunique()}")
            st.markdown("---")
            
            c1, c2 = st.columns([2, 1])
            with c1:
                trend = df.groupby(['MONTH_ID', 'MONTH_NAME'])['SALES'].sum().reset_index().sort_values('MONTH_ID')
                st.plotly_chart(px.line(trend, x='MONTH_NAME', y='SALES', markers=True, title="Monthly Sales Momentum"), use_container_width=True)
            with c2:
                st.plotly_chart(px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, title="Revenue by Product Line"), use_container_width=True)
            
            st.plotly_chart(px.bar(df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False), x='COUNTRY', y='SALES', color='SALES', title="Market Rankings"), use_container_width=True)

        # --- TAB 2: REVENUE SIMULATOR ---
        with tabs[1]:
            st.header("🔮 Strategic Scenario Simulator")
            col1, col2, col3 = st.columns(3)
            in_country = col1.selectbox("Market", sorted(df_master['COUNTRY'].unique()))
            in_prod = col2.selectbox("Product", sorted(df_master['PRODUCTLINE'].unique()))
            in_qty = col1.number_input("Quantity", 1, 1000000, 50)
            in_msrp = col2.number_input("Unit Price ($)", 0.01, 1000000.0, 150.0)
            in_month = col3.slider("Month", 1, 12, 1)
            model_choice = st.selectbox("Choose AI Model", list(trained_models.keys()))
            
            if st.button("RUN SIMULATION", type="primary", use_container_width=True):
                inp = pd.DataFrame([{'MONTH_ID': in_month, 'QTR_ID': (in_month-1)//3+1, 'MSRP': in_msrp, 'QUANTITYORDERED': in_qty, 'PRODUCTLINE': in_prod, 'COUNTRY': in_country}])
                pred = trained_models[model_choice][0].predict(inp)[0]
                st.markdown(f"<div style='background:#e3f2fd;padding:20px;border-radius:10px;text-align:center'><h1>${pred:,.2f}</h1></div>", unsafe_allow_html=True)

        # --- TAB 3: MARKET INSIGHTS ---
        with tabs[2]:
            st.header("🌍 Strategic Market Distribution")
            geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
            st.plotly_chart(px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="SALES", title="Global Revenue Map"), use_container_width=True)
            
            heat_df = df.pivot_table(index='COUNTRY', columns='PRODUCTLINE', values='SALES', aggfunc='sum').fillna(0)
            st.plotly_chart(px.imshow(heat_df, text_auto='.2s', aspect="auto", color_continuous_scale="Spectral_r", title="Revenue Heatmap"), use_container_width=True)

        # --- TAB 4: DEMAND FORECAST ---
        with tabs[3]:
            st.header(f"📅 Demand Curve Projection: {selected_year}")
            
            f_df = df.groupby(['MONTH_ID'])['SALES'].sum().reset_index()
            fig_f = go.Figure()
            fig_f.add_trace(go.Scatter(x=f_df['MONTH_ID'], y=f_df['SALES'], name='Revenue Model', line=dict(color='#1f4e79', width=3)))
            st.plotly_chart(fig_f, use_container_width=True)

        # --- TAB 5: CUSTOMER ANALYTICS ---
        with tabs[4]:
            st.header("👥 Customer Insights")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.plotly_chart(px.pie(df, names='DEALSIZE', title="Deal Size Concentration"), use_container_width=True)
            with col_s2:
                st.plotly_chart(px.bar(df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(10), x='COUNTRY', y='SALES', title="Top Regional Performance"), use_container_width=True)
            
            st.subheader("🎯 Pareto Analysis (Market Concentration)")
            pareto_df = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).reset_index()
            pareto_df['Revenue_Share'] = (pareto_df['SALES'].cumsum() / pareto_df['SALES'].sum()) * 100
            st.plotly_chart(px.area(pareto_df, x='COUNTRY', y='Revenue_Share', title="Cumulative Revenue Share"), use_container_width=True)

else:
    st.markdown("""<div class="welcome-header"><h1>🚀 PredictiCorp Intelligence</h1><p>The Global Executive Suite for Data-Driven Market Strategy</p></div>""", unsafe_allow_html=True)
