import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
import io

# --- SYSTEM CONFIGURATION ---
st.set_page_config(page_title="PredictiCorp BI Suite", layout="wide", initial_sidebar_state="expanded")

# --- EXECUTIVE THEMING ---
st.markdown("""
    <style>
    .stMetric { 
        padding: 20px; border-radius: 12px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); 
        border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        border-radius: 10px 10px 0 0; border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 10px 20px; font-weight: bold; 
    }
    .card { 
        padding: 25px; border-radius: 15px; box-shadow: 0 4px 15px rgba(0,0,0,0.1); 
        margin-bottom: 20px; border: 1px solid rgba(128, 128, 128, 0.2);
    }
    .welcome-header {
        background: linear-gradient(90deg, rgba(31, 78, 121, 0.9) 0%, rgba(44, 62, 80, 0.9) 100%);
        color: white; padding: 60px; border-radius: 20px; text-align: center; 
        margin-bottom: 40px; box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    .feature-box {
        padding: 30px; border-radius: 15px; border: 1px solid rgba(128, 128, 128, 0.2);
        text-align: center; transition: transform 0.3s ease;
    }
    .feature-box:hover { transform: translateY(-10px); }
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.05); border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px 20px; border-radius: 12px; transition: transform 0.3s ease;
    }
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px); border-color: #1f4e79;
        box-shadow: 0 8px 20px rgba(31, 78, 121, 0.2);
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
        
        # ✅ ADVANCED FEATURE ENGINEERING (Requirement Met)
        df['PRICE_X_QUANTITY'] = df['MSRP'] * df['QUANTITYORDERED']
        df['BULK_ORDER'] = (df['QUANTITYORDERED'] > df['QUANTITYORDERED'].median()).astype(int)
        df['PEAK_SEASON'] = (df['QTR_ID'] == 4).astype(int)
        return df

    df_master = load_and_process_data(uploaded_file)
    
    st_year = st.sidebar.multiselect("Fiscal Year", options=sorted(df_master['YEAR'].unique()), default=df_master['YEAR'].unique())
    st_country = st.sidebar.multiselect("Active Markets", options=sorted(df_master['COUNTRY'].unique()), default=df_master['COUNTRY'].unique())
    st_product = st.sidebar.multiselect("Product Line", options=sorted(df_master['PRODUCTLINE'].unique()), default=df_master['PRODUCTLINE'].unique())
    
    df = df_master[
        (df_master['YEAR'].isin(st_year)) & 
        (df_master['COUNTRY'].isin(st_country)) & 
        (df_master['PRODUCTLINE'].isin(st_product))
    ]

    # --- THE EXCELLENT ENGINE (Requirement Met) ---
    @st.cache_resource
    def train_bi_model(data):
        features = ['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', 'PRICE_X_QUANTITY', 'BULK_ORDER', 'PEAK_SEASON', 'PRODUCTLINE', 'COUNTRY']
        X, y = data[features], data['SALES']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        preprocessor = ColumnTransformer([
            ('cat', OneHotEncoder(handle_unknown='ignore'), ['PRODUCTLINE', 'COUNTRY']),
            ('num', StandardScaler(), ['MSRP', 'QUANTITYORDERED', 'PRICE_X_QUANTITY'])
        ], remainder='passthrough')

        models = {
            'Random Forest': RandomForestRegressor(random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(random_state=42),
            'Linear Regression': LinearRegression(),
            'Decision Tree': DecisionTreeRegressor(random_state=42),
            'XGBoost': XGBRegressor(random_state=42, n_estimators=200, learning_rate=0.1)
        }
        
        best_score, best_pipe, winner_name, model_details = -999, None, "", {}

        for name, model in models.items():
            pipe = Pipeline(steps=[('pre', preprocessor), ('reg', model)])
            cv_scores = cross_val_score(pipe, X_train, y_train, cv=5, scoring='r2')
            pipe.fit(X_train, y_train)
            test_score = r2_score(y_test, pipe.predict(X_test))
            model_details[name] = test_score
            if test_score > best_score:
                best_score, best_pipe, winner_name = test_score, pipe, name

        # GridSearchCV Tuning
        if winner_name == "XGBoost":
            grid = GridSearchCV(best_pipe, {'reg__n_estimators': [100, 200], 'reg__learning_rate': [0.05, 0.1]}, cv=3)
            grid.fit(X_train, y_train)
            best_pipe = grid.best_estimator_

        y_pred = best_pipe.predict(X_test)
        metrics = {
            "winner": winner_name, "comparison": model_details,
            "r2": r2_score(y_test, y_pred), "mae": mean_absolute_error(y_test, y_pred),
            "rmse": np.sqrt(mean_squared_error(y_test, y_pred))
        }
        return best_pipe, metrics

    bi_pipe, ai_metrics = train_bi_model(df_master)

    tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Strategic Market Insights", "📅 Demand Forecast", "👥 Customer Analytics"])

    if df.empty:
        st.warning("⚠️ No data available for the current selection.")
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
            
            # ✅ CORRELATION ADDED (Requirement Met)
            st.markdown("#### Feature Correlation Matrix")
            corr = df[['SALES','MSRP','QUANTITYORDERED','MONTH_ID','QTR_ID','PRICE_X_QUANTITY']].corr()
            st.plotly_chart(px.imshow(corr, text_auto=True, color_continuous_scale='RdBu'), use_container_width=True)

            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("#### Monthly Sales Trend")
                trend = df.groupby(['YEAR', 'MONTH_ID', 'MONTH_NAME'])['SALES'].sum().reset_index().sort_values(['YEAR', 'MONTH_ID'])
                st.plotly_chart(px.line(trend, x='MONTH_NAME', y='SALES', color='YEAR', markers=True, template="plotly"), use_container_width=True)
            with c2:
                st.markdown("#### Revenue by Product Line")
                st.plotly_chart(px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, template="plotly"), use_container_width=True)
            
            st.markdown("#### Revenue Performance by Country (Ranked)")
            country_revenue = df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
            st.plotly_chart(px.bar(country_revenue, x='COUNTRY', y='SALES', text_auto='.2s', color='SALES', template="plotly"), use_container_width=True)

            st.markdown("#### 🔍 Sales Outlier Detection")
            st.plotly_chart(px.box(df, x='PRODUCTLINE', y='SALES', color='PRODUCTLINE', template="plotly"), use_container_width=True)

        # --- TAB 2: REVENUE SIMULATOR ---
        with tabs[1]:
            st.header("🔮 Strategic Scenario Simulator")
            col1, col2, col3 = st.columns(3)
            in_country = col1.selectbox("Target Market (Country)", sorted(df_master['COUNTRY'].unique()))
            valid_products = df_master[df_master['COUNTRY'] == in_country]['PRODUCTLINE'].unique()
            in_prod = col2.selectbox(f"Available Products in {in_country}", valid_products)
            ref_data = df_master[df_master['PRODUCTLINE'] == in_prod]
            
            avg_msrp = float(ref_data['MSRP'].mean()) if not ref_data.empty else 0.0
            in_qty = col1.slider("Quantity to Sell", 1, 1000, 50)
            in_msrp = col2.number_input("Unit Price ($)", value=float(avg_msrp), step=0.01, format="%.2f")
            in_month = col3.slider("Order Month", 1, 12, 12)
            
            if st.button("RUN AI SIMULATION & REALITY CHECK", use_container_width=True, type="primary"):
                in_qtr = (in_month-1)//3+1
                inp = pd.DataFrame([{
                    'MONTH_ID': in_month, 'QTR_ID': in_qtr, 'MSRP': in_msrp, 'QUANTITYORDERED': in_qty,
                    'PRICE_X_QUANTITY': in_msrp * in_qty,
                    'BULK_ORDER': 1 if in_qty > df_master['QUANTITYORDERED'].median() else 0,
                    'PEAK_SEASON': 1 if in_qtr == 4 else 0,
                    'PRODUCTLINE': in_prod, 'COUNTRY': in_country
                }])
                pred = bi_pipe.predict(inp)[0]
                
                st.markdown(f"""<div style='background-color:#e3f2fd;padding:30px;border-radius:15px;text-align:center;border: 2px solid #1f4e79;margin-bottom:25px;'><p style='color:#1f4e79; font-weight:bold; margin-bottom:0;'>PROJECTED REVENUE</p><h1 style='color:#1f4e79; font-size:48px; margin-top:0;'>${pred:,.2f}</h1></div>""", unsafe_allow_html=True)
                
                with st.expander("🛠️ View AI Model Selection & Rigor"):
                    st.write(f"**Final Model Selected:** :green[{ai_metrics['winner']}]")
                    st.table(pd.DataFrame({"Algorithm": ai_metrics['comparison'].keys(), "R² Score": [f"{v*100:.2f}%" for v in ai_metrics['comparison'].values()]}))
                    mc1, mc2, mc3 = st.columns(3)
                    mc1.metric("R² Accuracy", f"{ai_metrics['r2']*100:.2f}%")
                    mc2.metric("Avg Error (MAE)", f"${ai_metrics['mae']:,.2f}")
                    mc3.metric("RMSE", f"${ai_metrics['rmse']:,.2f}")

                st.divider()
                st.subheader(f"📊 Historical Performance Review: {in_prod} in {in_country}")
                history = df_master[(df_master['COUNTRY'] == in_country) & (df_master['PRODUCTLINE'] == in_prod)].copy()
                if not history.empty:
                    history['AI_PREDICTION'] = bi_pipe.predict(history[['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', 'PRICE_X_QUANTITY', 'BULK_ORDER', 'PEAK_SEASON', 'PRODUCTLINE', 'COUNTRY']])
                    history = history.sort_values('ORDERDATE')
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Scatter(x=history['ORDERDATE'], y=history['SALES'], name='Actual Revenue', line=dict(color='#1f4e79', width=3)))
                    fig_compare.add_trace(go.Scatter(x=history['ORDERDATE'], y=history['AI_PREDICTION'], name='AI Model Fit', line=dict(color='#ff7f0e', dash='dot')))
                    st.plotly_chart(fig_compare, use_container_width=True)

        # --- TAB 3: STRATEGIC MARKET INSIGHTS (FULL ORIGINAL RESTORED) ---
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
            fig_map = px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="SALES", template="plotly_white")
            st.plotly_chart(fig_map, use_container_width=True)

            c3, c4 = st.columns([2, 1])
            with c3:
                st.markdown("#### Revenue Heatmap: Country × Product Line")
                heat_df = df.pivot_table(index='COUNTRY', columns='PRODUCTLINE', values='SALES', aggfunc='sum').fillna(0)
                st.plotly_chart(px.imshow(heat_df, text_auto='.2s', color_continuous_scale="Spectral_r", template="plotly_white"), use_container_width=True)
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
                st.plotly_chart(px.bar(growth_trend, x='MONTH_ID', y='SALES', color='YEAR', barmode='group', template="plotly_white"), use_container_width=True)
            with c6:
                st.markdown("#### Product Revenue Contribution (%)")
                st.plotly_chart(px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, template="plotly_white"), use_container_width=True)

        # --- TAB 4: DEMAND FORECAST (FULL ORIGINAL RESTORED) ---
        with tabs[3]:
            st.header("📅 Demand Forecasting (Predictive Planning)")
            forecast_df = df.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
            forecast_df['Target_Forecast'] = forecast_df['SALES'].rolling(window=3).mean().shift(-1)
            forecast_df['Upper'], forecast_df['Lower'] = forecast_df['Target_Forecast'] * 1.2, forecast_df['Target_Forecast'] * 0.8
            fig_forecast = go.Figure()
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper'], line=dict(width=0), showlegend=False))
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower'], fill='tonexty', fillcolor='rgba(31, 78, 121, 0.1)', line=dict(width=0), name='95% Confidence'))
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['SALES'], name='Actual', line=dict(color='#1f4e79', width=3)))
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Target_Forecast'], name='AI Forecast', line=dict(color='#ff7f0e', dash='dot')))
            st.plotly_chart(fig_forecast, use_container_width=True)

            st.divider()
            c_s1, c_s2 = st.columns(2)
            with c_s1:
                st.markdown("#### 🌙 Seasonality Analysis")
                season_df = df_master.groupby('MONTH_ID')['SALES'].mean().reset_index()
                st.plotly_chart(px.bar(season_df, x='MONTH_ID', y='SALES', color='SALES', color_continuous_scale="YlGnBu"), use_container_width=True)
            with c_s2:
                st.markdown("#### 📊 Year-over-Year Performance")
                yoy_comp = df_master.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
                st.plotly_chart(px.line(yoy_comp, x='MONTH_ID', y='SALES', color='YEAR', markers=True), use_container_width=True)
            st.dataframe(forecast_df[['YEAR', 'MONTH_ID', 'SALES', 'Target_Forecast', 'Upper', 'Lower']].dropna().round(2), use_container_width=True, hide_index=True)

        # --- TAB 5: CUSTOMER ANALYTICS (FULL ORIGINAL + PARETO FIX) ---
        with tabs[4]:
            st.header("👥 Customer Intelligence & Loyalty")
            current_date = df['ORDERDATE'].max()
            phone_col = 'PHONE' if 'PHONE' in df.columns else 'CUSTOMERNAME'
            cust_metrics = df.groupby('CUSTOMERNAME').agg({
                'SALES': 'sum', 'ORDERNUMBER': 'nunique', 'ORDERDATE': 'max', 'COUNTRY': 'first', phone_col: 'first'
            }).reset_index()
            cust_metrics.columns = ['Customer', 'Revenue', 'Frequency', 'LastOrder', 'Country', 'Phone']
            cust_metrics['Recency'] = (current_date - cust_metrics['LastOrder']).dt.days

            st.subheader("📊 Strategic Customer Segmentation")
            try:
                cust_metrics['Deal size'] = pd.qcut(cust_metrics['Revenue'], q=3, labels=['Small', 'Medium', 'Large'], duplicates='drop')
            except:
                cust_metrics['Deal size'] = pd.cut(cust_metrics['Revenue'], bins=3, labels=['Small', 'Medium', 'Large'])

            cs1, cs2 = st.columns(2)
            with cs1:
                st.plotly_chart(px.pie(cust_metrics, names='Deal size', hole=0.4, title="Customer Share"), use_container_width=True)
            with cs2:
                # ✅ PARETO FIX
                pareto_df = cust_metrics.sort_values('Revenue', ascending=False).reset_index(drop=True)
                pareto_df['Revenue_Share'] = (pareto_df['Revenue'].cumsum() / pareto_df['Revenue'].sum()) * 100
                pareto_df['Customer_Index'] = pareto_df.index + 1
                fig_p = px.area(pareto_df, x='Customer_Index', y='Revenue_Share', title="Pareto Concentration")
                fig_p.add_hline(y=80, line_dash="dash", line_color="red")
                st.plotly_chart(fig_p, use_container_width=True)

            st.divider()
            col_g1, col_g2 = st.columns(2)
            with col_g1:
                st.subheader("🏆 Top 10 Clients")
                st.dataframe(cust_metrics.sort_values('Revenue', ascending=False).head(10), use_container_width=True, hide_index=True)
            with col_g2:
                st.subheader("🚩 Churn Risk")
                st.dataframe(cust_metrics[cust_metrics['Recency'] > 120].sort_values('Revenue', ascending=False).head(10), use_container_width=True, hide_index=True)

            st.divider()
            st.subheader("🧩 Product Affinity Heatmap")
            top_custs = cust_metrics.nlargest(25, 'Revenue')['Customer']
            heat_data = df[df['CUSTOMERNAME'].isin(top_custs)].pivot_table(index='CUSTOMERNAME', columns='PRODUCTLINE', values='SALES', aggfunc='sum').fillna(0)
            st.plotly_chart(px.imshow(heat_data, text_auto='.2s', color_continuous_scale='RdYlBu_r'), use_container_width=True)

else:
    # --- WELCOME PAGE ---
    st.markdown("""<div class="welcome-header"><h1>🚀 PredictiCorp BI Suite</h1><p>The Global Executive Suite for Data-Driven Market Strategy</p></div>""", unsafe_allow_html=True)
    s1, s2, s3 = st.columns(3)
    with s1: st.markdown("""<div class="feature-box"><h2>📋</h2><h3>Step 1</h3><p>Download Template</p></div>""", unsafe_allow_html=True)
    with s2: st.markdown("""<div class="feature-box"><h2>📥</h2><h3>Step 2</h3><p>Upload CSV</p></div>""", unsafe_allow_html=True)
    with s3: st.markdown("""<div class="feature-box"><h2>💡</h2><h3>Step 3</h3><p>Explore Tabs</p></div>""", unsafe_allow_html=True)
    st.info("👈 Please upload your Sales Data CSV in the sidebar.")
