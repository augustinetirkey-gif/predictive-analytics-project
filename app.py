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
    .feature-box {
        padding: 30px; 
        border-radius: 15px; 
        border: 1px solid rgba(128, 128, 128, 0.2);
        text-align: center; 
        transition: transform 0.3s ease;
    }
    .feature-box:hover { transform: translateY(-10px); }
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.05); 
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    [data-testid="stMetricLabel"] p {
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
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
    
    # Determine the break point for historical data
    max_hist_year = 2005

    # --- AI SYNTHETIC DATA GENERATOR ---
    @st.cache_resource
    def train_forecasting_model(data):
        X = data[MODEL_FEATURES]
        y = data['SALES']
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['PRODUCTLINE', 'COUNTRY']),
            ('num', StandardScaler(), ['MONTH_ID','QTR_ID','MSRP','QUANTITYORDERED'])
        ])
        model = Pipeline(steps=[('pre', preprocessor), ('model', LinearRegression())])
        model.fit(X, y)
        return model

    forecasting_ai = train_forecasting_model(df_master)

    def generate_synthetic_rows(target_year, ref_df):
        synthetic_data = []
        countries = ref_df['COUNTRY'].unique()
        products = ref_df['PRODUCTLINE'].unique()
        avg_msrp = ref_df['MSRP'].mean()
        avg_qty = ref_df['QUANTITYORDERED'].mean()

        for month in range(1, 13):
            for country in countries:
                for product in products:
                    synthetic_data.append({
                        'MONTH_ID': month, 'QTR_ID': (month-1)//3 + 1,
                        'MSRP': avg_msrp, 'QUANTITYORDERED': avg_qty,
                        'PRODUCTLINE': product, 'COUNTRY': country, 'YEAR': target_year,
                        'ORDERDATE': pd.to_datetime(f"{target_year}-{month}-01"),
                        'MONTH_NAME': pd.to_datetime(f"{target_year}-{month}-01").month_name(),
                        'CUSTOMERNAME': "AI Projected Segment", 'STATUS': 'Shipped', 'DEALSIZE': 'Medium', 'PHONE': 'N/A'
                    })
        synth_df = pd.DataFrame(synthetic_data)
        synth_df['SALES'] = forecasting_ai.predict(synth_df[MODEL_FEATURES]).clip(lower=0)
        return synth_df

    st.sidebar.subheader("🔍 Filter Strategy")
    # Extended year options to include prediction range
    available_years = sorted(list(df_master['YEAR'].unique()))
    prediction_years = [2006, 2007, 2008, 2009, 2010]
    full_year_range = sorted(list(set(available_years + prediction_years)))
    
    # We use a selectbox for the single-year "Time Machine" view you requested
    selected_year = st.sidebar.selectbox("Fiscal Year Focus", options=full_year_range, index=full_year_range.index(max_hist_year))
    st_country = st.sidebar.multiselect("Active Markets", options=sorted(df_master['COUNTRY'].unique()), default=df_master['COUNTRY'].unique())
    st_product = st.sidebar.multiselect("Product Line", options=sorted(df_master['PRODUCTLINE'].unique()), default=df_master['PRODUCTLINE'].unique())

    # --- YEAR-BASED DATA LOGIC (THE "SWITCH") ---
    if selected_year < max_hist_year:
        # PURE HISTORICAL (2003-2004)
        df = df_master[df_master['YEAR'] == selected_year]
    elif selected_year == max_hist_year:
        # HYBRID 2005 (Real Data + Prediction Gap Fill)
        real_2005 = df_master[df_master['YEAR'] == 2005]
        months_we_have = real_2005['MONTH_ID'].unique()
        if len(months_we_have) < 12:
            full_2005_synth = generate_synthetic_rows(2005, df_master)
            gap_fill = full_2005_synth[~full_2005_synth['MONTH_ID'].isin(months_we_have)]
            df = pd.concat([real_2005, gap_fill])
        else:
            df = real_2005
    else:
        # PURE PREDICTION (2006-2010)
        df = generate_synthetic_rows(selected_year, df_master)

    # Re-apply filters to the dynamic df
    df = df[(df['COUNTRY'].isin(st_country)) & (df['PRODUCTLINE'].isin(st_product))]

    @st.cache_resource
    def train_models(data):
        data = data[MODEL_FEATURES + ['SALES']].dropna()
        X = data[MODEL_FEATURES]
        y = data['SALES']
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['PRODUCTLINE', 'COUNTRY']),
            ('num', StandardScaler(), ['MONTH_ID','QTR_ID','MSRP','QUANTITYORDERED'])
        ])
        models = {
            "Linear Regression": LinearRegression(),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "XGBoost": xgb.XGBRegressor(objective='reg:squarederror')
        }
        trained_results = {}
        for name, model in models.items():
            pipe = Pipeline(steps=[('pre', preprocessor), ('model', model)])
            pipe.fit(X, y)
            score = r2_score(y, pipe.predict(X)) * 100
            trained_results[name] = (pipe, score)
        return trained_results

    trained_models = train_models(df_master)

    tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Strategic Market Insights", "📅 Demand Forecast", "👥 Customer Analytics"])

    if df.empty:
        st.warning("⚠️ No data available for the current selection. Please adjust your filters.")
    else:
        # --- TAB 1: EXECUTIVE DASHBOARD ---
        with tabs[0]:
            st.subheader(f"Performance KPIs ({selected_year})")
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Total Revenue", f"${df['SALES'].sum()/1e6:.2f}M")
            k2.metric("Avg Order Value", f"${df['SALES'].mean():,.2f}")
            k3.metric("Transaction Volume", f"{len(df):,}")
            k4.metric("Active Regions", f"{df['COUNTRY'].nunique()}")
            st.markdown("---")
            
            st.subheader("📊 Descriptive Statistics Summary")
            st.dataframe(df.describe(), use_container_width=True)

            st.subheader("🔗 Correlation Analysis")
            corr = df[['SALES','QUANTITYORDERED','MSRP','MONTH_ID']].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("#### Monthly Sales Trend")
                trend = df.groupby(['YEAR', 'MONTH_ID', 'MONTH_NAME'])['SALES'].sum().reset_index().sort_values(['YEAR', 'MONTH_ID'])
                fig_trend = px.line(trend, x='MONTH_NAME', y='SALES', markers=True, template="plotly")
                st.plotly_chart(fig_trend, use_container_width=True)
            with c2:
                st.markdown("#### Revenue by Product Line")
                fig_pie = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, template="plotly")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("#### Revenue Performance by Country (Ranked)")
            country_revenue = df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
            fig_bar = px.bar(country_revenue, x='COUNTRY', y='SALES', text_auto='.2s', color='SALES', template="plotly")
            st.plotly_chart(fig_bar, use_container_width=True)

        # --- TAB 2: REVENUE SIMULATOR ---
        with tabs[1]:
            st.header("🔮 Strategic Scenario Simulator")
            col1, col2, col3 = st.columns(3)
            in_country = col1.selectbox("Target Market (Country)", sorted(df_master['COUNTRY'].unique()))
            in_prod = col2.selectbox(f"Available Products", sorted(df_master['PRODUCTLINE'].unique()))
            in_qty = col1.number_input("Quantity to Sell", min_value=1, max_value=1000000, value=50)
            in_msrp = col2.number_input("Unit Price ($)", min_value=0.01, max_value=1000000.0, value=150.0)
            in_month = col3.slider("Order Month", 1, 12, 12)
            model_choice = st.selectbox("Choose Prediction Model", list(trained_models.keys()))
            selected_model, model_score = trained_models[model_choice]

            if st.button("RUN AI SIMULATION", use_container_width=True, type="primary"):
                inp = pd.DataFrame([{'MONTH_ID': in_month, 'QTR_ID': (in_month-1)//3+1, 'MSRP': in_msrp, 'QUANTITYORDERED': in_qty, 'PRODUCTLINE': in_prod, 'COUNTRY': in_country}])
                pred = selected_model.predict(inp)[0]
                st.markdown(f"<div style='background-color:#e3f2fd;padding:30px;border-radius:15px;text-align:center;border: 2px solid #1f4e79;'><p style='color:#1f4e79; font-weight:bold;'>PROJECTED REVENUE</p><h1 style='color:#1f4e79; font-size:48px;'>${pred:,.2f}</h1></div>", unsafe_allow_html=True)

        # --- TAB 3: STRATEGIC MARKET INSIGHTS ---
        with tabs[2]:
            st.header("🌍 Strategic Market Insights")
            geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
            fig_map = px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="SALES", hover_name="COUNTRY", template="plotly_white")
            st.plotly_chart(fig_map, use_container_width=True)

            heat_df = df.pivot_table(index='COUNTRY', columns='PRODUCTLINE', values='SALES', aggfunc='sum').fillna(0)
            fig_heat = px.imshow(heat_df, text_auto='.2s', aspect="auto", color_continuous_scale="Spectral_r", template="plotly_white")
            st.plotly_chart(fig_heat, use_container_width=True)

        # --- TAB 4: DEMAND FORECAST ---
        with tabs[3]:
            st.header(f"📅 Demand Forecasting: {selected_year}")
            forecast_df = df.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
            forecast_df['Target_Forecast'] = forecast_df['SALES'].rolling(window=3).mean().shift(-1)
            
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=forecast_df['MONTH_ID'], y=forecast_df['SALES'], name='Revenue Baseline', line=dict(color='#1f4e79', width=3)))
            fig_forecast.add_trace(go.Scatter(x=forecast_df['MONTH_ID'], y=forecast_df['Target_Forecast'], name='AI Trendline', line=dict(color='#ff7f0e', dash='dot')))
            st.plotly_chart(fig_forecast, use_container_width=True)

        # --- TAB 5: CUSTOMER ANALYTICS ---
        with tabs[4]:
            st.header("👥 Customer Intelligence")
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.plotly_chart(px.pie(df, names='DEALSIZE', title="Deal Size Distribution"), use_container_width=True)
            with col_s2:
                top_regions = df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(10)
                st.plotly_chart(px.bar(top_regions, x='COUNTRY', y='SALES', title="Top Regional Segments"), use_container_width=True)
            
            pareto_df = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).reset_index()
            pareto_df['Revenue_Share'] = (pareto_df['SALES'].cumsum() / pareto_df['SALES'].sum()) * 100
            fig_pareto = px.area(pareto_df, x='COUNTRY', y='Revenue_Share', title="The Pareto Curve (Revenue Concentration)")
            st.plotly_chart(fig_pareto, use_container_width=True)

else:
    st.markdown("""<div class="welcome-header"><h1>🚀 Welcome to PredictiCorp Intelligence</h1><p>Upload your CSV to activate Historical and AI-Forecasted Insights</p></div>""", unsafe_allow_html=True)
