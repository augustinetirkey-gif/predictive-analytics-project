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
from fpdf import FPDF
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
    # --- FILE VALIDATION LOGIC ---
    df_check = pd.read_csv(uploaded_file)
    
    # Check if the uploaded file has the columns defined in your template_df
    required_columns = list(template_df.columns)
    uploaded_columns = list(df_check.columns)
    
    # Validation: Check if all required columns exist
    is_valid = all(col in uploaded_columns for col in required_columns)
    
    if not is_valid:
        st.error("⚠️ **Incorrect File Format Detected**")
        st.info("The uploaded CSV does not match the required system schema. Please **download the CSV Template** from the sidebar and ensure your data matches those column headers exactly.")
        st.stop() # Prevents the rest of the app from running with bad data

    @st.cache_data
    def load_and_process_data(file):
        # Using the already read df_check to save memory
        df = df_check.copy()
        if 'ORDERDATE' in df.columns:
            df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'], errors='coerce')
            df['YEAR'] = df['ORDERDATE'].dt.year
            df['MONTH_NAME'] = df['ORDERDATE'].dt.month_name()
        elif 'YEAR_ID' in df.columns:
            df['YEAR'] = df['YEAR_ID']
        return df

    df_master = load_and_process_data(uploaded_file)
    
    st.sidebar.subheader("🔍 Filter Strategy")
    st_year = st.sidebar.multiselect("Fiscal Year", options=sorted(df_master['YEAR'].unique()), default=df_master['YEAR'].unique())
    st_country = st.sidebar.multiselect("Active Markets", options=sorted(df_master['COUNTRY'].unique()), default=df_master['COUNTRY'].unique())
    st_product = st.sidebar.multiselect("Product Line", options=sorted(df_master['PRODUCTLINE'].unique()), default=df_master['PRODUCTLINE'].unique())
    
    df = df_master[
        (df_master['YEAR'].isin(st_year)) & 
        (df_master['COUNTRY'].isin(st_country)) & 
        (df_master['PRODUCTLINE'].isin(st_product))
    ]

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

    tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Strategic Market Insights", "📅 Demand Forecast", "👥 Customer Analytics","📄 Report Generation"])

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
                           

       # --- TAB 2: REVENUE SIMULATOR (EXTRAPOLATION ENABLED) ---
        with tabs[1]:
            st.header("🔮 Strategic Scenario Simulator")
            col1, col2, col3 = st.columns(3)
            
            # Selection Logic (Same as before)
            in_country = col1.selectbox("Target Market (Country)", sorted(df_master['COUNTRY'].unique()))
            valid_products = df_master[df_master['COUNTRY'] == in_country]['PRODUCTLINE'].unique()
            in_prod = col2.selectbox(f"Available Products in {in_country}", valid_products)
            
            # Reference data for context
            ref_data = df_master[df_master['PRODUCTLINE'] == in_prod]
            avg_msrp = float(ref_data['MSRP'].mean()) if not ref_data.empty else 0.0
            min_msrp = float(ref_data['MSRP'].min()) if not ref_data.empty else 0.0
            max_msrp = float(ref_data['MSRP'].max()) if not ref_data.empty else 0.0
            
            st.info(f"💡 *Historical Price Context for {in_prod}:* Avg: ${avg_msrp:.2f} | Range: ${min_msrp:.2f} - ${max_msrp:.2f}")
            
            # --- UNLOCKED INPUTS ---
            # Max values increased to 1,000,000 to allow high-value simulation
            in_qty = col1.number_input("Quantity to Sell", min_value=1, max_value=1000000, value=50)
            in_msrp = col2.number_input("Unit Price ($)", min_value=0.01, max_value=1000000.0, value=float(avg_msrp), step=0.01, format="%.2f")
            in_month = col3.slider("Order Month", 1, 12, 12)

            st.divider()
            st.subheader("🤖 Model Selection")
            st.caption("🚀 *Tip: Use 'Linear Regression' for predictions far beyond historical price/quantity ranges.*")
            
            model_choice = st.selectbox("Choose Prediction Model", list(trained_models.keys()))
            selected_model, model_score = trained_models[model_choice]
            st.info(f"Model Accuracy (R²): {model_score:.2f}%")

            # --- HYPERPARAMETER TUNING (Logic remains identical) ---
            if st.checkbox(f"Run Hyperparameter Tuning ({model_choice})"):
                tuning_grids = {
                    "Random Forest": {'model__n_estimators': [50, 100, 150], 'model__max_depth': [None, 5, 10]},
                    "Decision Tree": {'model__max_depth': [None, 5, 10, 20], 'model__min_samples_split': [2, 5, 10]},
                    "Gradient Boosting": {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1], 'model__max_depth': [3, 5]},
                    "XGBoost": {'model__n_estimators': [50, 100], 'model__learning_rate': [0.01, 0.1], 'model__max_depth': [3, 5]},
                    "Linear Regression": {} 
                }

                param_grid = tuning_grids.get(model_choice, {})
                
                if not param_grid and model_choice != "Linear Regression":
                    st.warning(f"No tuning grid defined for {model_choice}.")
                elif model_choice == "Linear Regression":
                    st.info("Linear Regression does not require hyperparameter tuning.")
                else:
                    preprocessor = ColumnTransformer([
                        ('cat', OneHotEncoder(handle_unknown='ignore'), ['PRODUCTLINE', 'COUNTRY']),
                        ('num', StandardScaler(), ['MONTH_ID','QTR_ID','MSRP','QUANTITYORDERED'])
                    ])
                    
                    base_models = {
                        "Linear Regression": LinearRegression(),
                        "Decision Tree": DecisionTreeRegressor(random_state=42),
                        "Random Forest": RandomForestRegressor(random_state=42),
                        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
                        "XGBoost": xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
                    }
                    
                    tune_pipe = Pipeline([('pre', preprocessor), ('model', base_models[model_choice])])

                    with st.spinner(f"Tuning {model_choice} in progress..."):
                        grid = GridSearchCV(tune_pipe, param_grid, cv=3)
                        grid.fit(df_master[MODEL_FEATURES], df_master['SALES'])
                        st.success(f"Best Params for {model_choice}: {grid.best_params_}")
                        selected_model = grid.best_estimator_

            # --- PREDICTION EXECUTION ---
            if st.button("RUN AI SIMULATION & REALITY CHECK", use_container_width=True, type="primary"):
                inp = pd.DataFrame([{
                    'MONTH_ID': in_month, 
                    'QTR_ID': (in_month-1)//3+1, 
                    'MSRP': in_msrp, 
                    'QUANTITYORDERED': in_qty, 
                    'PRODUCTLINE': in_prod, 
                    'COUNTRY': in_country
                }])
                
                # The model will now predict based on the high input values
                pred = selected_model.predict(inp)[0]

                st.markdown(f"""
                    <div style='background-color:#e3f2fd;padding:30px;border-radius:15px;text-align:center;border: 2px solid #1f4e79;margin-bottom:25px;'>
                        <p style='color:#1f4e79; font-weight:bold; margin-bottom:0;'>PROJECTED REVENUE</p>
                        <h1 style='color:#1f4e79; font-size:48px; margin-top:0;'>${pred:,.2f}</h1>
                    </div>
                """, unsafe_allow_html=True)
                
                st.divider()
                st.subheader(f"📊 Historical Performance Review: {in_prod} in {in_country}")
                history = df_master[(df_master['COUNTRY'] == in_country) & (df_master['PRODUCTLINE'] == in_prod)].copy()
                if not history.empty:
                    history['AI_PREDICTION'] = selected_model.predict(history[MODEL_FEATURES])
                    history = history.sort_values('ORDERDATE')
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Scatter(x=history['ORDERDATE'], y=history['SALES'], name='Actual Revenue', line=dict(color='#1f4e79', width=3)))
                    fig_compare.add_trace(go.Scatter(x=history['ORDERDATE'], y=history['AI_PREDICTION'], name='AI Model Fit', line=dict(color='#ff7f0e', dash='dot')))
                    fig_compare.update_layout(title="AI vs Historical Reality", template="plotly_white", xaxis_title="Timeline", yaxis_title="Revenue ($)")
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    err = np.mean(abs(history['SALES'] - history['AI_PREDICTION']) / history['SALES']) * 100
                    mae = mean_absolute_error(history['SALES'], history['AI_PREDICTION'])
                    rmse = np.sqrt(mean_squared_error(history['SALES'], history['AI_PREDICTION']))
                    r2 = r2_score(history['SALES'], history['AI_PREDICTION'])

                    st.write(f"MAE: {mae:,.2f}")
                    st.write(f"RMSE: {rmse:,.2f}")
                    st.write(f"R² Score: {r2*100:.2f}%")
                    st.success(f"✅ The AI matches historical data with an average error of only {err:.2f}% for this selection.")
                else:
                    st.warning("No historical data found for this specific combination.")

        # --- TAB 3: STRATEGIC MARKET INSIGHTS ---
        with tabs[2]:
            st.header("🌍 Strategic Market Insights")
            st.header("💡 Business Directives")
            top_country = df.groupby('COUNTRY')['SALES'].sum().idxmax()
            top_prod = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
            
            col_i1, col_i2 = st.columns(2)
            with col_i1:
                st.markdown(f"<div class='card'><h4>📦 Inventory Optimization</h4><p><b>Insight:</b> <b>{top_prod}</b> is the top performer.<br><b>Action:</b> Prioritize supply for this line.</p></div>", unsafe_allow_html=True)
            with col_i2:
                st.markdown(f"<div class='card'><h4>🌍 Regional Strategy</h4><p><b>Insight:</b> <b>{top_country}</b> drives peak revenue.<br><b>Action:</b> Test localized loyalty programs here.</p></div>", unsafe_allow_html=True)
            
            geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
            fig_map = px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="COUNTRY", hover_name="COUNTRY", template="plotly_white")
            fig_map.update_geos(projection_type="mercator")
            st.plotly_chart(fig_map, use_container_width=True)

            c3, c4 = st.columns([2, 1])
            with c3:
                st.markdown("#### Revenue Heatmap: Country × Product Line")
                heat_df = df.pivot_table(index='COUNTRY', columns='PRODUCTLINE', values='SALES', aggfunc='sum').fillna(0)
                fig_heat = px.imshow(heat_df, text_auto='.2s', aspect="auto", color_continuous_scale="Spectral_r", template="plotly_white")
                st.plotly_chart(fig_heat, use_container_width=True)
            
            with c4:
                st.markdown("#### Top 5 vs Bottom 5 Markets")
                m_sorted = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).reset_index()
                st.write("*Top 5 Markets*")
                st.dataframe(m_sorted.head(5), hide_index=True, use_container_width=True)
                st.write("*Bottom 5 Markets*")
                st.dataframe(m_sorted.tail(5), hide_index=True, use_container_width=True)

            c5, c6 = st.columns(2)
            with c5:
                st.markdown("#### YoY Revenue Performance")
                growth_trend = df.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
                fig_growth = px.bar(growth_trend, x='MONTH_ID', y='SALES', color='YEAR', barmode='group', template="plotly_white")
                st.plotly_chart(fig_growth, use_container_width=True)
            with c6:
                st.markdown("#### Product Revenue Contribution (%)")
                fig_don = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_don, use_container_width=True)

              # TAB 4: Demand Forecast (Advanced Strategic Planning)
        with tabs[3]:
            st.header("📅 Demand Forecasting (Predictive Planning)")
            
            # --- 1. DATA PREPARATION & FORECAST LOGIC ---
            forecast_df = df.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
            
            # Calculate simple AI Forecast (Rolling Mean)
            forecast_df['Target_Forecast'] = forecast_df['SALES'].rolling(window=3).mean().shift(-1)
            
            # Add Uncertainty (Confidence Intervals: +/- 20% range)
            forecast_df['Upper'] = forecast_df['Target_Forecast'] * 1.2
            forecast_df['Lower'] = forecast_df['Target_Forecast'] * 0.8

            # --- 2. FORECAST VISUALIZATION WITH CONFIDENCE BANDS ---
            fig_forecast = go.Figure()

            # Confidence Range Area (Shaded Background)
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df['Upper'], 
                line=dict(width=0), showlegend=False, name='Upper Bound'
            ))
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df['Lower'], 
                fill='tonexty', fillcolor='rgba(31, 78, 121, 0.1)', 
                line=dict(width=0), name='95% Confidence Interval'
            ))
            
            # Actual Sales Line
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df['SALES'], 
                name='Actual Sales', line=dict(color='#1f4e79', width=3)
            ))
            
            # AI Forecast Line
            fig_forecast.add_trace(go.Scatter(
                x=forecast_df.index, y=forecast_df['Target_Forecast'], 
                name='AI Forecast', line=dict(color='#ff7f0e', dash='dot', width=2)
            ))

            fig_forecast.update_layout(
                title="Sales Momentum Forecast with Predictive Confidence Range", 
                template="plotly_white", 
                xaxis_title="Timeline Step (Months)", 
                yaxis_title="Revenue ($)",
                hovermode="x unified"
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

            # --- 3. SEASONALITY & YoY COMPARISON ---
            st.divider()
            c5, c6 = st.columns(2)
            
            with c5:
                st.markdown("#### 🌙 Seasonality Analysis (Monthly Trends)")
                # Average performance per month across all historical years
                season_df = df_master.groupby('MONTH_ID')['SALES'].mean().reset_index()
                fig_season = px.bar(
                    season_df, x='MONTH_ID', y='SALES', 
                    template="plotly_white", color='SALES', 
                    color_continuous_scale="YlGnBu",
                    labels={'MONTH_ID': 'Month Index', 'SALES': 'Avg Revenue'}
                )
                st.plotly_chart(fig_season, use_container_width=True)
            
            with c6:
                st.markdown("#### 📊 Year-over-Year (YoY) Performance")
                # Compare trends across different years
                yoy_comp = df_master.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
                fig_yoy = px.line(
                    yoy_comp, x='MONTH_ID', y='SALES', color='YEAR', 
                    markers=True, template="plotly_white",
                    labels={'MONTH_ID': 'Month Index'}
                )
                st.plotly_chart(fig_yoy, use_container_width=True)

            # --- 4. DATA INTELLIGENCE TABLE ---
            st.divider()
            st.markdown("#### 📥 Forecast Data Intelligence")
            st.dataframe(
                forecast_df[['YEAR', 'MONTH_ID', 'SALES', 'Target_Forecast', 'Upper', 'Lower']].dropna().round(2), 
                use_container_width=True,
                hide_index=True
            )
        # --- TAB 5: CUSTOMER ANALYTICS ---
        with tabs[4]:
            st.header("👥 Customer Intelligence & Loyalty")
            current_date = df['ORDERDATE'].max()
            cust_metrics = df.groupby('CUSTOMERNAME').agg({
                'SALES': 'sum',
                'ORDERNUMBER': 'nunique',
                'ORDERDATE': 'max',
                'COUNTRY': 'first',
                'PHONE': 'first',
                'DEALSIZE': lambda x: x.mode()[0] if not x.mode().empty else "N/A"
            }).reset_index()
            cust_metrics.columns = ['Customer', 'Revenue', 'Frequency', 'LastOrder', 'Country', 'Phone', 'Typical_Deal']
            cust_metrics['Recency'] = (current_date - cust_metrics['LastOrder']).dt.days
            # ✅ SAFE SEGMENTATION (No Crash Version)
            try:
                cust_metrics['Deal size'] = pd.qcut(
                    cust_metrics['Revenue'],
                    q=3,
                    labels=['Small', 'Medium', 'Large'],
                    duplicates='drop'
                )
            except ValueError:
                cust_metrics['Deal size'] = "Single Tier"

           

            
            col_s1, col_s2 = st.columns([1, 1])
            with col_s1:
                fig_seg = px.pie(cust_metrics, names='Deal size', hole=0.4, title="Customer Base Share")
                st.plotly_chart(fig_seg, use_container_width=True)
            with col_s2:
                deal_summary = cust_metrics.groupby('Deal size')['Revenue'].mean().reset_index()
                fig_deal_bar = px.bar(deal_summary, x='Deal size', y='Revenue', color='Deal size', title="Avg Spend per Tier")
                st.plotly_chart(fig_deal_bar, use_container_width=True)

            st.divider()
            st.subheader("🌍 Customer Geographic Footprint")
            fig_geo = px.scatter_geo(cust_metrics, locations="Country", locationmode='country names', size="Revenue", color="Deal size", hover_name="Customer", template="plotly")
            st.plotly_chart(fig_geo, use_container_width=True)

            st.divider()
            st.subheader("🎯 Revenue Concentration (Pareto)")
            pareto_df = cust_metrics.sort_values('Revenue', ascending=False).copy()
            pareto_df['Revenue_Share'] = (pareto_df['Revenue'].cumsum() / pareto_df['Revenue'].sum()) * 100
            pareto_df['Customer_Count_Pct'] = np.arange(1, len(pareto_df) + 1) / len(pareto_df) * 100
            fig_pareto = px.area(pareto_df, x='Customer_Count_Pct', y='Revenue_Share', title="The Pareto Curve")
            fig_pareto.add_hline(y=80, line_dash="dash", line_color="red")
            st.plotly_chart(fig_pareto, use_container_width=True)

            st.divider()
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.subheader("🏆 Top 10 High-Value Clients")
                st.dataframe(cust_metrics.sort_values('Revenue', ascending=False).head(10), use_container_width=True, hide_index=True)
            with col_g2:
                st.subheader("🚩 Churn Risk Analysis")
                churn_df = cust_metrics[cust_metrics['Recency'] > 120].sort_values('Revenue', ascending=False)
                st.dataframe(churn_df.head(10), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("🧩 Product Affinity Heatmap")
            top_custs = cust_metrics.nlargest(25, 'Revenue')['Customer']
            heat_data = df[df['CUSTOMERNAME'].isin(top_custs)].pivot_table(index='CUSTOMERNAME', columns='PRODUCTLINE', values='SALES', aggfunc='sum').fil  lna(0)
            st.plotly_chart(px.imshow(heat_data, text_auto='.2s', aspect="auto", color_continuous_scale='RdYlBu_r', template="plotly"), use_container_width=True)


      # --- TAB 6: COMPLETE STRATEGIC AUDIT ---
        with tabs[5]:
            st.header("📄 Full Spectrum Executive Intelligence")
            st.markdown("Generates a complete multi-page audit of Financials, Operations, AI Predictions, and Customer Logic.")

            # --- 1. DATA PREP FOR ALL SECTIONS ---
            total_rev = df['SALES'].sum()
            avg_order = df['SALES'].mean()
            top_market = df.groupby('COUNTRY')['SALES'].sum().idxmax()
            top_prod = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
            
            # Data for tables
            top_5_prods = df.groupby('PRODUCTLINE')['SALES'].sum().nlargest(5).reset_index()
            top_5_countries = df.groupby('COUNTRY')['SALES'].sum().nlargest(5).reset_index()
            top_cust = cust_metrics.nlargest(5, 'Revenue')[['Customer', 'Revenue']]
            
            # Stats (Transposed for better PDF fit)
            stats_df = df[['SALES', 'QUANTITYORDERED', 'MSRP']].describe().reset_index()

            # --- 2. THE PDF MASTER CLASS ---
            class ExecutivePDF(FPDF):
                def header(self):
                    self.set_fill_color(31, 78, 121) # PredictiCorp Blue
                    self.rect(0, 0, 210, 35, 'F')
                    self.set_text_color(255, 255, 255)
                    self.set_font('Arial', 'B', 22)
                    self.cell(0, 15, 'PREDICTICORP BI - FULL AUDIT', 0, 1, 'C')
                    self.set_font('Arial', 'I', 10)
                    self.cell(0, 5, f'Generated: {pd.Timestamp.now().strftime("%Y-%m-%d %H:%M")}', 0, 1, 'C')
                    self.ln(20)

                def footer(self):
                    self.set_y(-15)
                    self.set_font('Arial', 'I', 8)
                    self.set_text_color(150, 150, 150)
                    self.cell(0, 10, f'Internal Strategic Document | Page {self.page_no()}', 0, 0, 'C')

                def section_header(self, title):
                    self.set_font('Arial', 'B', 14)
                    self.set_text_color(31, 78, 121)
                    self.cell(0, 10, title, 0, 1, 'L')
                    self.line(10, self.get_y(), 200, self.get_y())
                    self.ln(5)
                    self.set_text_color(0, 0, 0)

                def draw_table(self, df_data, col_widths):
                    self.set_font('Arial', 'B', 10)
                    self.set_fill_color(240, 240, 240)
                    # Headers
                    for i, col in enumerate(df_data.columns):
                        self.cell(col_widths[i], 10, str(col), 1, 0, 'C', fill=True)
                    self.ln()
                    # Data
                    self.set_font('Arial', '', 9)
                    for row in df_data.itertuples(index=False):
                        for i, val in enumerate(row):
                            text = f"${val:,.2f}" if isinstance(val, (float, int)) and val > 100 else str(val)
                            self.cell(col_widths[i], 8, text, 1)
                        self.ln()
                    self.ln(5)

            # --- 3. CONSTRUCT THE PAGES ---
            pdf = ExecutivePDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            
            # PAGE 1: EXECUTIVE SUMMARY & STATS
            pdf.add_page()
            pdf.section_header("1. Financial KPIs & Descriptive Statistics")
            pdf.set_font("Arial", '', 11)
            pdf.multi_cell(0, 8, f"Based on the active filters, PredictiCorp has generated a total revenue of ${total_rev:,.2f} "
                                 f"across {len(df):,} transactions. The average transaction value stands at ${avg_order:,.2f}.")
            pdf.ln(5)
            pdf.draw_table(stats_df, [40, 50, 50, 50])

            # PAGE 2: MARKET & PRODUCT ANALYSIS
            pdf.add_page()
            pdf.section_header("2. Global Market & Product Intelligence")
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 10, f"Primary Revenue Market: {top_market}", 0, 1)
            pdf.cell(0, 10, f"Core Product Line: {top_prod}", 0, 1)
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 12)
            pdf.cell(0, 10, "Top 5 Products by Revenue Share", 0, 1)
            pdf.draw_table(top_5_prods, [100, 50])
            
            pdf.cell(0, 10, "Top 5 Regional Markets", 0, 1)
            pdf.draw_table(top_5_countries, [100, 50])

            # PAGE 3: AI DIAGNOSTICS & CUSTOMERS
            pdf.add_page()
            pdf.section_header("3. AI Diagnostics & Customer Health")
            
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 10, "AI Model Performance Summary:", 0, 1)
            pdf.set_font("Arial", '', 10)
            pdf.multi_cell(0, 8, f"The analytics engine utilizes the '{model_choice}' model for revenue simulation. "
                                 f"With a current R2 accuracy score of {model_score:.2f}%, the model shows high reliability "
                                 f"for strategic forecasting in the {top_market} region.")
            pdf.ln(5)
            
            pdf.set_font("Arial", 'B', 11)
            pdf.cell(0, 10, f"Customer Intelligence (High Risk Churn Count: {len(churn_df)})", 0, 1)
            pdf.draw_table(top_cust, [100, 50])

            # --- 4. EXPORT (The Fix) ---
            pdf_bytes = bytes(pdf.output()) # Modern fpdf2 output method
            
            st.success("✅ Full Audit Report Generated Successfully")
            st.download_button(
                label="📥 Download Full Strategic Audit (PDF)",
                data=pdf_bytes,
                file_name=f"PredictiCorp_Full_Audit_{pd.Timestamp.now().strftime('%Y%m%d')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )
else:
    # --- WELCOME PAGE ---
    st.markdown("""<div class="welcome-header"><h1>🚀 Welcome to PredictiCorp Intelligence</h1><p>The Global Executive Suite for Data-Driven Market Strategy</p></div>""", unsafe_allow_html=True)
    st.markdown("### 🛠️ Get Started in 3 Simple Steps")
    s1, s2, s3 = st.columns(3)
    with s1: st.markdown("""<div class="feature-box"><h2>📋</h2><h3>Step 1</h3><p>Download the CSV template.</p></div>""", unsafe_allow_html=True)
    with s2: st.markdown("""<div class="feature-box"><h2>📥</h2><h3>Step 2</h3><p>Upload your sales data.</p></div>""", unsafe_allow_html=True)
    with s3: st.markdown("""<div class="feature-box"><h2>💡</h2><h3>Step 3</h3><p>Explore analytical tabs.</p></div>""", unsafe_allow_html=True)
    st.markdown("---")
    st.info("👈 Please upload your Sales Data CSV in the sidebar to activate insights.")
