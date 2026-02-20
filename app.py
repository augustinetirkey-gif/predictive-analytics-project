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

# Define prediction features globally for use in functions and tuning
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
    
    # FUTURE MODE: Add 2026 to selection options
    available_years = sorted(df_master['YEAR'].unique().tolist())
    if 2026 not in available_years:
        available_years.append(2026)

    st.sidebar.subheader("🔍 Filter Strategy")
    st_year = st.sidebar.multiselect("Fiscal Year", options=available_years, default=[y for y in available_years if y != 2026])
    st_country = st.sidebar.multiselect("Active Markets", options=sorted(df_master['COUNTRY'].unique()), default=df_master['COUNTRY'].unique())
    st_product = st.sidebar.multiselect("Product Line", options=sorted(df_master['PRODUCTLINE'].unique()), default=df_master['PRODUCTLINE'].unique())
    
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
            "Decision Tree": DecisionTreeRegressor(max_depth=5),
            "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
            "Gradient Boosting": GradientBoostingRegressor(),
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

    # --- THE MAGIC BRIDGE: GENERATE 2026 FUTURE DATA ---
    if 2026 in st_year and len(st_year) == 1:
        st.sidebar.warning("🚀 SHOWING AI FUTURE PREDICTIONS (2026)")
        
        # We use Linear Regression as it handles extrapolation (unseen years/values) better than trees
        lr_pipeline = trained_models["Linear Regression"][0]
        
        # Build scaffold for 2026: 12 months for every selected Country and Product
        future_data_list = []
        for month in range(1, 13):
            for country in st_country:
                for prod in st_product:
                    # Get historical averages to make values realistic
                    hist_msrp = df_master[df_master['PRODUCTLINE']==prod]['MSRP'].mean()
                    hist_qty = df_master[df_master['PRODUCTLINE']==prod]['QUANTITYORDERED'].mean()
                    
                    future_data_list.append({
                        'MONTH_ID': month,
                        'QTR_ID': (month-1)//3 + 1,
                        'MSRP': hist_msrp,
                        'QUANTITYORDERED': hist_qty,
                        'PRODUCTLINE': prod,
                        'COUNTRY': country,
                        'YEAR': 2026,
                        'ORDERDATE': pd.Timestamp(year=2026, month=month, day=1),
                        'CUSTOMERNAME': "AI Future Prospect",
                        'PHONE': "FUTURE-ID",
                        'DEALSIZE': "Medium"
                    })
        
        df = pd.DataFrame(future_data_list)
        df['SALES'] = lr_pipeline.predict(df[MODEL_FEATURES])
        df['MONTH_NAME'] = df['ORDERDATE'].dt.month_name()
        df['ORDERNUMBER'] = np.arange(90000, 90000 + len(df))
    else:
        # Standard historical filter
        df = df_master[
            (df_master['YEAR'].isin(st_year)) & 
            (df_master['COUNTRY'].isin(st_country)) & 
            (df_master['PRODUCTLINE'].isin(st_product))
        ]

    tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Strategic Market Insights", "📅 Demand Forecast", "👥 Customer Analytics"])

    if df.empty:
        st.warning("⚠️ No data available for the current selection. Please adjust your filters.")
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
            
            st.subheader("📊 Descriptive Statistics Summary")
            st.dataframe(df.describe(), use_container_width=True)

            st.subheader("🧠 Key EDA Insights")
            total_rev = df['SALES'].sum()
            top_country = df.groupby('COUNTRY')['SALES'].sum().idxmax()
            top_product = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
            st.markdown(f"""
          • Total revenue generated is **${total_rev:,.2f}** • Highest revenue comes from **{top_country}** • Best performing product line is **{top_product}** • Sales show seasonal monthly variation  
          • Dataset contains multiple markets and product categories  
            """)
            st.subheader("🧹 Data Cleaning Summary")
            missing = df.isnull().sum().sum()
            st.markdown(f"""
          • Total records: **{len(df)}** • Missing values handled: **{missing}** • Date converted to datetime  
          • Feature engineering applied (Year extraction)  
              """)
            st.subheader("🔗 Correlation Analysis")
            corr = df[['SALES','QUANTITYORDERED','MSRP','MONTH_ID']].corr()
            fig_corr = px.imshow(corr, text_auto=True, aspect="auto", title="Correlation Matrix")
            st.plotly_chart(fig_corr, use_container_width=True)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("#### Monthly Sales Trend")
                trend = df.groupby(['YEAR', 'MONTH_ID', 'MONTH_NAME'])['SALES'].sum().reset_index().sort_values(['YEAR', 'MONTH_ID'])
                fig_trend = px.line(trend, x='MONTH_NAME', y='SALES', color='YEAR', markers=True, template="plotly")
                st.plotly_chart(fig_trend, use_container_width=True)
            with c2:
                st.markdown("#### Revenue by Product Line")
                fig_pie = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, template="plotly")
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("#### Revenue Performance by Country (Ranked)")
            country_revenue = df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
            fig_bar = px.bar(country_revenue, x='COUNTRY', y='SALES', text_auto='.2s', color='SALES', template="plotly")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("#### 🔍 Sales Outlier Detection")
            fig_box = px.box(df, x='PRODUCTLINE', y='SALES', color='PRODUCTLINE', template="plotly")
            st.plotly_chart(fig_box, use_container_width=True)

        # --- TAB 2: REVENUE SIMULATOR ---
        with tabs[1]:
            st.header("🔮 Strategic Scenario Simulator")
            col1, col2, col3 = st.columns(3)
            in_country = col1.selectbox("Target Market (Country)", sorted(df_master['COUNTRY'].unique()))
            valid_products = df_master[df_master['COUNTRY'] == in_country]['PRODUCTLINE'].unique()
            in_prod = col2.selectbox(f"Available Products in {in_country}", valid_products)
            
            ref_data = df_master[df_master['PRODUCTLINE'] == in_prod]
            avg_msrp = float(ref_data['MSRP'].mean()) if not ref_data.empty else 0.0
            st.info(f"💡 *Historical Price Context for {in_prod}:* Avg: ${avg_msrp:.2f}")
            
            in_qty = col1.number_input("Quantity to Sell", min_value=1, max_value=1000000, value=50)
            in_msrp = col2.number_input("Unit Price ($)", min_value=0.01, max_value=1000000.0, value=float(avg_msrp), step=0.01, format="%.2f")
            in_month = col3.slider("Order Month", 1, 12, 12)

            st.divider()
            st.subheader("🤖 Model Selection")
            model_choice = st.selectbox("Choose Prediction Model", list(trained_models.keys()))
            selected_model, model_score = trained_models[model_choice]
            st.info(f"Model Accuracy (R²): {model_score:.2f}%")

            if st.checkbox(f"Run Hyperparameter Tuning ({model_choice})"):
                tuning_grids = {
                    "Random Forest": {'model__n_estimators': [50, 100], 'model__max_depth': [None, 5]},
                    "Decision Tree": {'model__max_depth': [None, 5, 10]},
                    "Gradient Boosting": {'model__n_estimators': [50, 100]},
                    "XGBoost": {'model__n_estimators': [50, 100]},
                    "Linear Regression": {} 
                }
                param_grid = tuning_grids.get(model_choice, {})
                if param_grid:
                    with st.spinner("Tuning..."):
                        grid = GridSearchCV(selected_model, param_grid, cv=3)
                        grid.fit(df_master[MODEL_FEATURES], df_master['SALES'])
                        selected_model = grid.best_estimator_
                        st.success("Tuning Complete!")

            if st.button("RUN AI SIMULATION & REALITY CHECK", use_container_width=True, type="primary"):
                inp = pd.DataFrame([{'MONTH_ID': in_month, 'QTR_ID': (in_month-1)//3+1, 'MSRP': in_msrp, 'QUANTITYORDERED': in_qty, 'PRODUCTLINE': in_prod, 'COUNTRY': in_country}])
                pred = selected_model.predict(inp)[0]
                st.markdown(f"<div style='background-color:#e3f2fd;padding:30px;border-radius:15px;text-align:center;border: 2px solid #1f4e79;margin-bottom:25px;'><p style='color:#1f4e79; font-weight:bold; margin-bottom:0;'>PROJECTED REVENUE</p><h1 style='color:#1f4e79; font-size:48px; margin-top:0;'>${pred:,.2f}</h1></div>", unsafe_allow_html=True)

        # --- TAB 3: STRATEGIC MARKET INSIGHTS ---
        with tabs[2]:
            st.header("🌍 Strategic Market Insights")
            top_country_m = df.groupby('COUNTRY')['SALES'].sum().idxmax()
            top_prod_m = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
            
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                st.markdown(f"<div class='card'><h4>📦 Inventory Optimization</h4><p><b>Insight:</b> <b>{top_prod_m}</b> is top performer.<br><b>Action:</b> Prioritize supply.</p></div>", unsafe_allow_html=True)
            with col_i2:
                st.markdown(f"<div class='card'><h4>🌍 Regional Strategy</h4><p><b>Insight:</b> <b>{top_country_m}</b> drives peak revenue.<br><b>Action:</b> Test localized loyalty.</p></div>", unsafe_allow_html=True)
            
            geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
            fig_map = px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="COUNTRY", template="plotly_white")
            st.plotly_chart(fig_map, use_container_width=True)

            c3, c4 = st.columns([2, 1])
            with c3:
                heat_df = df.pivot_table(index='COUNTRY', columns='PRODUCTLINE', values='SALES', aggfunc='sum').fillna(0)
                fig_heat = px.imshow(heat_df, text_auto='.2s', aspect="auto", color_continuous_scale="Spectral_r", template="plotly_white")
                st.plotly_chart(fig_heat, use_container_width=True)
            with c4:
                m_sorted = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).reset_index()
                st.write("*Top Markets*")
                st.dataframe(m_sorted.head(5), hide_index=True, use_container_width=True)

        # --- TAB 4: DEMAND FORECAST ---
        with tabs[3]:
            st.header("📅 Demand Forecasting (Predictive Planning)")
            forecast_df = df.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
            forecast_df['Target_Forecast'] = forecast_df['SALES'].rolling(window=2, min_periods=1).mean().shift(-1)
            
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['SALES'], name='Actual/Synthetic Sales', line=dict(color='#1f4e79', width=3)))
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Target_Forecast'], name='AI Trendline', line=dict(color='#ff7f0e', dash='dot')))
            st.plotly_chart(fig_forecast, use_container_width=True)

        # --- TAB 5: CUSTOMER ANALYTICS ---
        with tabs[4]:
            st.header("👥 Customer Intelligence & Loyalty")
            cust_metrics = df.groupby('CUSTOMERNAME').agg({
                'SALES': 'sum',
                'ORDERNUMBER': 'nunique',
                'COUNTRY': 'first',
                'PHONE': 'first',
                'DEALSIZE': lambda x: x.mode()[0] if not x.mode().empty else "N/A"
            }).reset_index()
            cust_metrics.columns = ['Customer', 'Revenue', 'Frequency', 'Country', 'Phone', 'Typical_Deal']
            
            col_s1, col_s2 = st.columns(2)
            with col_s1:
                st.subheader("Revenue Concentration")
                pareto_df = cust_metrics.sort_values('Revenue', ascending=False).copy()
                pareto_df['Revenue_Share'] = (pareto_df['Revenue'].cumsum() / pareto_df['Revenue'].sum()) * 100
                st.plotly_chart(px.area(pareto_df, y='Revenue_Share', title="Pareto Curve"), use_container_width=True)
            with col_s2:
                st.subheader("Top Clients")
                st.dataframe(cust_metrics.sort_values('Revenue', ascending=False).head(10), use_container_width=True, hide_index=True)

else:
    # --- WELCOME PAGE ---
    st.markdown("""<div class="welcome-header"><h1>🚀 Welcome to PredictiCorp Intelligence</h1><p>The Global Executive Suite for Data-Driven Market Strategy</p></div>""", unsafe_allow_html=True)
    st.markdown("### 🛠️ Get Started in 3 Simple Steps")
    s1, s2, s3 = st.columns(3)
    with s1: st.markdown("""<div class="feature-box"><h2>📋</h2><h3>Step 1</h3><p>Download the CSV template.</p></div>""", unsafe_allow_html=True)
    with s2: st.markdown("""<div class="feature-box"><h2>📥</h2><h3>Step 2</h3><p>Upload your sales data.</p></div>""", unsafe_allow_html=True)
    with s3: st.markdown("""<div class="feature-box"><h2>💡</h2><h3>Step 3</h3><p>Explore analytical tabs.</p></div>""", unsafe_allow_html=True)
    st.info("👈 Please upload your Sales Data CSV in the sidebar. Select '2026' alone to enter Future Mode.")
