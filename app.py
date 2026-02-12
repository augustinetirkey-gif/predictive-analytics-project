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
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
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
    
    # --- SIDEBAR FILTERS ---
    st.sidebar.subheader("🔍 Filter Strategy")
    st_year = st.sidebar.multiselect("Fiscal Year", options=sorted(df_master['YEAR'].unique()), default=df_master['YEAR'].unique())
    st_country = st.sidebar.multiselect("Active Markets", options=sorted(df_master['COUNTRY'].unique()), default=df_master['COUNTRY'].unique())
    st_product = st.sidebar.multiselect("Product Line", options=sorted(df_master['PRODUCTLINE'].unique()), default=df_master['PRODUCTLINE'].unique())
    
    # Global Data Filtering
    df = df_master[
        (df_master['YEAR'].isin(st_year)) & 
        (df_master['COUNTRY'].isin(st_country)) & 
        (df_master['PRODUCTLINE'].isin(st_product))
    ]

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
            
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("#### Monthly Sales Trend")
                trend = df.groupby(['YEAR', 'MONTH_ID', 'MONTH_NAME'])['SALES'].sum().reset_index().sort_values(['YEAR', 'MONTH_ID'])
                fig_trend = px.line(trend, x='MONTH_NAME', y='SALES', color='YEAR', markers=True, template="plotly_white")
                st.plotly_chart(fig_trend, use_container_width=True)
            with c2:
                # --- PIE CHART OF REVENUE ---
                st.markdown("#### Revenue by Product Line")
                fig_pie = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            # --- RANKED REVENUE BY COUNTRY BAR CHART ---
            st.markdown("#### Revenue Performance by Country (Ranked)")
            country_revenue = df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
            fig_bar = px.bar(country_revenue, x='COUNTRY', y='SALES', text_auto='.2s', color='SALES', color_continuous_scale='Blues', template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)

            # OUTLIER DETECTION
            st.markdown("#### 🔍 Sales Outlier Detection")
            fig_box = px.box(df, x='PRODUCTLINE', y='SALES', color='PRODUCTLINE', template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)
# --- TAB 2: UNIFIED AI COMMAND CENTER (MOVABLE 360° CALCULATOR) ---
        with tabs[1]:
            st.header("🔮 Strategic AI Simulator")
            
            # --- 1. SELECTION CONTROLS (THE "MOVEABLE" CONTEXT) ---
            col_sel1, col_sel2, col_sel3, col_sel4 = st.columns(4)
            active_country = col_sel1.selectbox("Region Context (Country):", sorted(df_master['COUNTRY'].unique()))
            
            valid_prods = df_master[df_master['COUNTRY'] == active_country]['PRODUCTLINE'].unique()
            active_prod = col_sel2.selectbox(f"Product Line", valid_prods)
            
            valid_years = sorted(df_master[(df_master['COUNTRY'] == active_country) & (df_master['PRODUCTLINE'] == active_prod)]['YEAR'].unique())
            active_year = col_sel3.selectbox("Historical Year Context", ["All Years"] + list(valid_years))
            
            active_month = col_sel4.slider("Simulation Month", 1, 12, 12)
            
            # Historical Filtering Logic for Context
            ref_filter = (df_master['PRODUCTLINE'] == active_prod) & (df_master['COUNTRY'] == active_country)
            if active_year != "All Years":
                ref_filter &= (df_master['YEAR'] == active_year)
            ref_data = df_master[ref_filter]
            
            # Calculate Averages for the Info Box
            avg_msrp = float(ref_data['MSRP'].mean()) if not ref_data.empty else 0.0
            min_msrp = float(ref_data['MSRP'].min()) if not ref_data.empty else 0.0
            max_msrp = float(ref_data['MSRP'].max()) if not ref_data.empty else 0.0
            
            context_label = f"Year {active_year}" if active_year != "All Years" else "All Historical Years"
            st.info(f"💡 **Historical Context ({context_label}):** Avg Unit Price: ${avg_msrp:.2f} | Range: ${min_msrp:.2f} - ${max_msrp:.2f}")

            # --- 2. MULTI-INPUT PARAMETERS ---
            st.markdown("### 🛠️ Simulation Parameters")
            p_col1, p_col2, p_col3 = st.columns(3)
            in_qty = p_col1.slider("Quantity to Sell", 1, 1000, 50)
            in_price = p_col2.number_input("Simulation Unit Price ($)", value=float(avg_msrp), step=0.01, format="%.2f")
            target_rev_goal = p_col3.number_input("Strategic Revenue Target ($)", value=5000.0, step=500.0)

            # --- 3. THE 360° AI CALCULATION (ONE CLICK, MULTIPLE ACTIONS) ---
            if st.button("🚀 EXECUTE 360° AI SCENARIO ANALYSIS", use_container_width=True, type="primary"):
                # ACTION A: Forward Prediction (Inputs -> Revenue)
                inp = pd.DataFrame([{
                    'MONTH_ID': active_month, 
                    'QTR_ID': (active_month-1)//3+1, 
                    'MSRP': in_price, 
                    'QUANTITYORDERED': in_qty, 
                    'PRODUCTLINE': active_prod, 
                    'COUNTRY': active_country
                }])
                pred_revenue = bi_pipe.predict(inp)[0]
                
                # ACTION B: Reverse Prediction (Revenue Target -> Price)
                required_unit_price = target_rev_goal / in_qty
                
                # ACTION C: Accuracy Reality Check
                # Calculate Error specifically for the selected historical context
                history = ref_data.copy()
                
                # Display Dual Results
                res1, res2 = st.columns(2)
                with res1:
                    st.markdown(f"""
                        <div style='background-color:#e3f2fd;padding:25px;border-radius:15px;text-align:center;border:2px solid #1f4e79;'>
                            <p style='color:#1f4e79; font-weight:bold; margin-bottom:0;'>PROJECTED REVENUE</p>
                            <h2 style='color:#1f4e79; margin-top:0;'>${pred_revenue:,.2f}</h2>
                            <p style='font-size:12px; color:#555;'>Based on Price: ${in_price:,.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)

                with res2:
                    st.markdown(f"""
                        <div style='background-color:#f0f7da;padding:25px;border-radius:15px;text-align:center;border:2px solid #2e7d32;'>
                            <p style='color:#1b5e20; font-weight:bold; margin-bottom:0;'>REQUIRED PRICE FOR TARGET</p>
                            <h2 style='color:#1b5e20; margin-top:0;'>${required_unit_price:,.2f}</h2>
                            <p style='font-size:12px; color:#333;'>To hit Target: ${target_rev_goal:,.2f}</p>
                        </div>
                    """, unsafe_allow_html=True)
                
                st.divider()
                
                # --- 4. DATA VISUALIZATION & MODEL REALITY CHECK ---
                st.subheader(f"📊 Model Reality Check ({context_label})")
                
                if not history.empty:
                    # Run AI over history to show "Fit"
                    hist_features = history[['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']]
                    history['AI_PREDICTION'] = bi_pipe.predict(hist_features)
                    history = history.sort_values('ORDERDATE')
                    
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Scatter(x=history['ORDERDATE'], y=history['SALES'], name='Actual Revenue', line=dict(color='#1f4e79', width=3)))
                    fig_compare.add_trace(go.Scatter(x=history['ORDERDATE'], y=history['AI_PREDICTION'], name='AI Model Fit', line=dict(color='#ff7f0e', dash='dot')))
                    fig_compare.update_layout(title=f"AI Intelligence vs Historical Reality", template="plotly_white", xaxis_title="Timeline", yaxis_title="Revenue ($)")
                    st.plotly_chart(fig_compare, use_container_width=True)
                    
                    err = np.mean(abs(history['SALES'] - history['AI_PREDICTION']) / history['SALES']) * 100
                    st.success(f"✅ **AI Intelligence Check:** The model is {100-err:.2f}% accurate for this market segment.")
                else:
                    st.warning("Insufficient historical data for a reality check in this specific segment.")
        

        # TAB 3: Market Insights
        with tabs[2]:
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

        # TAB 4: Demand Forecast
        with tabs[3]:
            st.header("📅 Demand Forecasting (Predictive Planning)")
            forecast_df = df.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
            forecast_df['Target_Forecast'] = forecast_df['SALES'].rolling(window=3).mean().shift(-1)
            fig_forecast = px.line(forecast_df, x='MONTH_ID', y=['SALES', 'Target_Forecast'], markers=True, template="plotly_white", title="3-Month Sales Momentum Forecast")
            st.plotly_chart(fig_forecast, use_container_width=True)

        # TAB 5: Customer Analytics
        with tabs[4]:
            st.header("👥 Customer Lifetime Value & Loyalty")
            cust_val = df.groupby('CUSTOMERNAME')['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(10)
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                st.subheader("Top 10 High-Value Customers")
                st.plotly_chart(px.bar(cust_val, x='SALES', y='CUSTOMERNAME', orientation='h', template="plotly_white"), use_container_width=True)
            with col_c2:
                st.subheader("Deal Size Analysis")
                st.plotly_chart(px.histogram(df, x='DEALSIZE', color='DEALSIZE', template="plotly_white"), use_container_width=True)

else:
    # --- WELCOME PAGE ---
    st.markdown("""<div class="welcome-header"><h1>🚀 Welcome to PredictiCorp Intelligence</h1><p>The Global Executive Suite for Data-Driven Market Strategy</p></div>""", unsafe_allow_html=True)
    st.info("👈 Please upload your Sales Data CSV in the sidebar to activate insights.")
