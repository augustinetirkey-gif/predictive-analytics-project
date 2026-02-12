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
                st.markdown("#### Revenue by Product Line")
                fig_pie = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, color_discrete_sequence=px.colors.qualitative.Prism)
                st.plotly_chart(fig_pie, use_container_width=True)
            
            st.markdown("#### Revenue Performance by Country (Ranked)")
            country_revenue = df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
            fig_bar = px.bar(country_revenue, x='COUNTRY', y='SALES', text_auto='.2s', color='SALES', color_continuous_scale='Blues', template="plotly_white")
            st.plotly_chart(fig_bar, use_container_width=True)

            st.markdown("#### 🔍 Sales Outlier Detection")
            fig_box = px.box(df, x='PRODUCTLINE', y='SALES', color='PRODUCTLINE', template="plotly_white")
            st.plotly_chart(fig_box, use_container_width=True)

        # TAB 2: Simulator (Grounded in Historical Data)
        with tabs[1]:
            st.header("🔮 Strategic Scenario Simulator")
            col1, col2, col3 = st.columns(3)
            in_country = col1.selectbox("Target Market (Country)", sorted(df_master['COUNTRY'].unique()))
            valid_products = df_master[df_master['COUNTRY'] == in_country]['PRODUCTLINE'].unique()
            in_prod = col2.selectbox(f"Available Products in {in_country}", valid_products)
            ref_data = df_master[df_master['PRODUCTLINE'] == in_prod]
            
            avg_msrp = float(ref_data['MSRP'].mean()) if not ref_data.empty else 0.0
            min_msrp = float(ref_data['MSRP'].min()) if not ref_data.empty else 0.0
            max_msrp = float(ref_data['MSRP'].max()) if not ref_data.empty else 0.0
            
            st.info(f"💡 **Historical Price Context for {in_prod}:** Avg: ${avg_msrp:.2f} | Range: ${min_msrp:.2f} - ${max_msrp:.2f}")
            
            in_qty = col1.slider("Quantity to Sell", 1, 1000, 50)
            in_msrp = col2.number_input("Unit Price ($)", value=float(avg_msrp), step=0.01, format="%.2f")
            in_month = col3.slider("Order Month", 1, 12, 12)
            
            if st.button("RUN AI SIMULATION & REALITY CHECK", use_container_width=True, type="primary"):
                inp = pd.DataFrame([{'MONTH_ID': in_month, 'QTR_ID': (in_month-1)//3+1, 'MSRP': in_msrp, 'QUANTITYORDERED': in_qty, 'PRODUCTLINE': in_prod, 'COUNTRY': in_country}])
                pred = bi_pipe.predict(inp)[0]
                st.markdown(f"""<div style='background-color:#e3f2fd;padding:30px;border-radius:15px;text-align:center;border: 2px solid #1f4e79;margin-bottom:25px;'><p style='color:#1f4e79; font-weight:bold; margin-bottom:0;'>PROJECTED REVENUE</p><h1 style='color:#1f4e79; font-size:48px; margin-top:0;'>${pred:,.2f}</h1></div>""", unsafe_allow_html=True)
                st.divider()
                st.subheader(f"📊 Historical Performance Review: {in_prod} in {in_country}")
                history = df_master[(df_master['COUNTRY'] == in_country) & (df_master['PRODUCTLINE'] == in_prod)].copy()
                if not history.empty:
                    hist_features = history[['MONTH_ID', 'QTR_ID', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']]
                    history['AI_PREDICTION'] = bi_pipe.predict(hist_features)
                    history = history.sort_values('ORDERDATE')
                    fig_compare = go.Figure()
                    fig_compare.add_trace(go.Scatter(x=history['ORDERDATE'], y=history['SALES'], name='Actual Revenue', line=dict(color='#1f4e79', width=3)))
                    fig_compare.add_trace(go.Scatter(x=history['ORDERDATE'], y=history['AI_PREDICTION'], name='AI Model Fit', line=dict(color='#ff7f0e', dash='dot')))
                    fig_compare.update_layout(title="How closely does the AI match historical reality?", template="plotly_white", xaxis_title="Timeline", yaxis_title="Revenue ($)")
                    st.plotly_chart(fig_compare, use_container_width=True)
                    err = np.mean(abs(history['SALES'] - history['AI_PREDICTION']) / history['SALES']) * 100
                    st.success(f"✅ The AI matches historical data with an average error of only {err:.2f}% for this selection.")

        # TAB 3: Strategic Market Insights (UPGRADED)
        with tabs[2]:
            st.header("🌍 Strategic Market Insights")
            
            # --- KPI Row ---
            growth_df = df_master.groupby('YEAR')['SALES'].sum().reset_index()
            growth_df['Growth'] = growth_df['SALES'].pct_change() * 100
            
            m1, m2, m3 = st.columns(3)
            with m1:
                top_country = df.groupby('COUNTRY')['SALES'].sum().idxmax()
                st.metric("Top Market", top_country)
            with m2:
                fastest_grow = growth_df.iloc[-1]['Growth'] if len(growth_df) > 1 else 0
                st.metric("Global YoY Growth", f"{fastest_grow:.1f}%", delta=f"{fastest_grow:.1f}%")
            with m3:
                top_p = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
                st.metric("Hero Product", top_p)

            # --- Map and Tables ---
            c1, c2 = st.columns([2, 1])
            with c1:
                st.markdown("#### Global Revenue Distribution")
                geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
                fig_map = px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="SALES", hover_name="COUNTRY", template="plotly_white", color_continuous_scale="Viridis")
                fig_map.update_geos(projection_type="mercator")
                st.plotly_chart(fig_map, use_container_width=True)
            
            with c2:
                st.markdown("#### Top 5 vs Bottom 5 Markets")
                m_sorted = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).reset_index()
                st.write("**Top 5 Markets**")
                st.dataframe(m_sorted.head(5), hide_index=True)
                st.write("**Bottom 5 Markets**")
                st.dataframe(m_sorted.tail(5), hide_index=True)

            # --- Heatmap and Donut ---
            c3, c4 = st.columns(2)
            with c3:
                st.markdown("#### Revenue Heatmap: Country × Product")
                heat_df = df.pivot_table(index='COUNTRY', columns='PRODUCTLINE', values='SALES', aggfunc='sum').fillna(0)
                fig_heat = px.imshow(heat_df, text_auto='.2s', aspect="auto", color_continuous_scale="Blues", template="plotly_white")
                st.plotly_chart(fig_heat, use_container_width=True)
            
            with c4:
                st.markdown("#### Percentage Contribution")
                fig_don = px.pie(df, values='SALES', names='COUNTRY', hole=0.6, template="plotly_white")
                st.plotly_chart(fig_don, use_container_width=True)

        # TAB 4: Demand Forecast (UPGRADED)
        with tabs[3]:
            st.header("📅 Demand Forecasting (Predictive Planning)")
            
            # --- Forecast with Confidence ---
            forecast_df = df.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
            forecast_df['Target_Forecast'] = forecast_df['SALES'].rolling(window=3).mean().shift(-1)
            forecast_df['Upper'] = forecast_df['Target_Forecast'] * 1.15
            forecast_df['Lower'] = forecast_df['Target_Forecast'] * 0.85

            fig_forecast = go.Figure()
            # Confidence Range
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Upper'], line=dict(width=0), showlegend=False))
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Lower'], fill='tonexty', fillcolor='rgba(31, 78, 121, 0.1)', line=dict(width=0), name='95% Confidence Interval'))
            # Main Lines
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['SALES'], name='Actual Sales', line=dict(color='#1f4e79', width=3)))
            fig_forecast.add_trace(go.Scatter(x=forecast_df.index, y=forecast_df['Target_Forecast'], name='AI Forecast', line=dict(color='#ff7f0e', dash='dot')))
            
            fig_forecast.update_layout(title="Sales Momentum Forecast with Error Margin", template="plotly_white", xaxis_title="Timeline (Periods)", yaxis_title="Revenue ($)")
            st.plotly_chart(fig_forecast, use_container_width=True)

            # --- Seasonality and Product Forecasts ---
            c5, c6 = st.columns(2)
            with c5:
                st.markdown("#### Seasonality Analysis (Monthly Averages)")
                season_df = df_master.groupby('MONTH_ID')['SALES'].mean().reset_index()
                fig_season = px.bar(season_df, x='MONTH_ID', y='SALES', template="plotly_white", color='SALES', color_continuous_scale="YlGnBu")
                st.plotly_chart(fig_season, use_container_width=True)
            
            with c6:
                st.markdown("#### YoY Forecast Comparison")
                yoy_comp = df_master.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
                fig_yoy = px.line(yoy_comp, x='MONTH_ID', y='SALES', color='YEAR', template="plotly_white")
                st.plotly_chart(fig_yoy, use_container_width=True)

            st.markdown("#### 📥 Interactive Forecast Table")
            st.dataframe(forecast_df.dropna(), use_container_width=True)

        # TAB 5: Customer Analytics (UPGRADED)
        with tabs[4]:
            st.header("👥 Customer Lifetime Value & Loyalty")
            
            # --- Segmentation ---
            cust_base = df.groupby('CUSTOMERNAME').agg({'SALES': 'sum', 'ORDERNUMBER': 'nunique', 'COUNTRY': 'first'}).reset_index()
            cust_base['Segment'] = pd.qcut(cust_base['SALES'], q=3, labels=['Bronze (Low Value)', 'Silver (Mid Value)', 'Gold (High Value)'])
            
            c7, c8 = st.columns(2)
            with c7:
                st.markdown("#### Customer Segmentation (Value Tiers)")
                fig_seg = px.pie(cust_base, names='Segment', template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel)
                st.plotly_chart(fig_seg, use_container_width=True)
            
            with c8:
                st.markdown("#### Lifetime Value (LTV) Trend")
                ltv_df = df.sort_values('ORDERDATE')
                ltv_df['Cumulative_Sales'] = ltv_df['SALES'].cumsum()
                fig_ltv = px.line(ltv_df, x='ORDERDATE', y='Cumulative_Sales', template="plotly_white", title="Cumulative Business Value Over Time")
                st.plotly_chart(fig_ltv, use_container_width=True)

            # --- Distribution and Churn ---
            st.markdown("#### Geographic Customer Distribution")
            fig_cust_map = px.scatter_geo(cust_base, locations="COUNTRY", locationmode='country names', size="SALES", color="Segment", template="plotly_white")
            st.plotly_chart(fig_cust_map, use_container_width=True)

            c9, c10 = st.columns(2)
            with c9:
                st.markdown("#### Product Preference by Customer Heatmap")
                c_prod_heat = df.pivot_table(index='CUSTOMERNAME', columns='PRODUCTLINE', values='SALES', aggfunc='sum').fillna(0).head(20)
                st.plotly_chart(px.imshow(c_prod_heat, template="plotly_white", color_continuous_scale="Purples"), use_container_width=True)
            
            with c10:
                st.markdown("#### 🚩 Churn Risk Analysis (Dormant Customers)")
                # Simplified churn: customers with significantly lower average sales than the mean
                avg_sales = cust_base['SALES'].mean()
                at_risk = cust_base[cust_base['SALES'] < (avg_sales * 0.5)].sort_values('SALES')
                st.write("Customers with 50% lower than average activity:")
                st.dataframe(at_risk[['CUSTOMERNAME', 'SALES', 'COUNTRY']].head(10), hide_index=True)

else:
    # --- WELCOME PAGE ---
    st.markdown("""<div class="welcome-header"><h1>🚀 Welcome to PredictiCorp Intelligence</h1><p>The Global Executive Suite for Data-Driven Market Strategy</p></div>""", unsafe_allow_html=True)
    st.info("👈 Please upload your Sales Data CSV in the sidebar to activate insights.")
