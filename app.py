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

    tabs = st.tabs(["📈 Executive Dashboard", "🔮 Revenue Simulator", "🌍 Strategic Market Insights", "📅 AI Demand Forecast", "👥 AI Customer Intelligence"])

    if df.empty:
        st.warning("⚠️ No data available for the current selection.")
    else:
        # TAB 1: Dashboard
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
                st.plotly_chart(px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5), use_container_width=True)
            
            st.markdown("#### Top Revenue Generating Countries")
            country_revenue = df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False)
            st.plotly_chart(px.bar(country_revenue, x='COUNTRY', y='SALES', color='SALES', template="plotly_white"), use_container_width=True)

        # TAB 2: Simulator
        with tabs[1]:
            st.header("🔮 Strategic Scenario Simulator")
            col1, col2, col3 = st.columns(3)
            in_prod = col1.selectbox("Product Line", df_master['PRODUCTLINE'].unique())
            in_qty = col1.slider("Quantity", 10, 500, 50)
            in_country = col2.selectbox("Country", sorted(df_master['COUNTRY'].unique()))
            in_msrp = col2.number_input("Unit Price ($)", value=100)
            in_month = col3.slider("Order Month", 1, 12, 6)
            if st.button("RUN AI SIMULATION"):
                inp = pd.DataFrame([{'MONTH_ID': in_month, 'QTR_ID': (in_month-1)//3+1, 'MSRP': in_msrp, 'QUANTITYORDERED': in_qty, 'PRODUCTLINE': in_prod, 'COUNTRY': in_country}])
                pred = bi_pipe.predict(inp)[0]
                st.markdown(f"<div style='background-color:#e3f2fd;padding:30px;border-radius:15px;text-align:center;'><h1>Predicted Revenue: ${pred:,.2f}</h1><p>Model Confidence: {ai_score:.1f}%</p></div>", unsafe_allow_html=True)

        # TAB 3: Market Insights
        with tabs[2]:
            st.header("💡 Business Directives")
            top_country = df.groupby('COUNTRY')['SALES'].sum().idxmax()
            top_prod = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
            col_i1, col_i2 = st.columns(2)
            with col_i1: st.markdown(f"<div class='card'><h4>📦 Inventory</h4><p>Focus on <b>{top_prod}</b>.</p></div>", unsafe_allow_html=True)
            with col_i2: st.markdown(f"<div class='card'><h4>🌍 Markets</h4><p>Invest in <b>{top_country}</b>.</p></div>", unsafe_allow_html=True)
            geo_df = df.groupby('COUNTRY')['SALES'].sum().reset_index()
            st.plotly_chart(px.choropleth(geo_df, locations="COUNTRY", locationmode='country names', color="SALES", template="plotly_white"), use_container_width=True)

        # --- UPDATED TAB 4: AI DEMAND FORECAST ---
        with tabs[3]:
            st.header("📅 AI Demand Forecast")
            st.markdown("---")
            
            # AI Logic: Predict the next 3 months based on current business context
            last_month = df['MONTH_ID'].max()
            future_months = [((last_month + i - 1) % 12) + 1 for i in range(1, 4)]
            
            # Prepare average inputs for the AI model to "estimate" future path
            projections = []
            for m in future_months:
                qtr = (m-1)//3 + 1
                input_df = pd.DataFrame([{
                    'MONTH_ID': m, 'QTR_ID': qtr, 
                    'MSRP': df['MSRP'].mean(), 
                    'QUANTITYORDERED': df['QUANTITYORDERED'].mean(), 
                    'PRODUCTLINE': df['PRODUCTLINE'].mode()[0], 
                    'COUNTRY': df['COUNTRY'].mode()[0]
                }])
                pred = bi_pipe.predict(input_df)[0]
                projections.append({'MONTH_ID': m, 'SALES': pred, 'Type': 'AI Projection'})
            
            proj_df = pd.DataFrame(projections)
            hist_df = df.groupby('MONTH_ID')['SALES'].mean().reset_index()
            hist_df['Type'] = 'Historical Average'
            
            combined = pd.concat([hist_df, proj_df])
            
            f_col1, f_col2 = st.columns([2, 1])
            with f_col1:
                st.subheader("Projected Sales Path")
                fig_forecast = px.line(combined, x='MONTH_ID', y='SALES', color='Type', markers=True, 
                                       line_dash='Type', template="plotly_white", color_discrete_sequence=['#1f4e79', '#ff7f0e'])
                st.plotly_chart(fig_forecast, use_container_width=True)
            
            with f_col2:
                st.markdown(f"""
                <div class='card'>
                    <h4>🧠 AI Prediction Insight</h4>
                    <p>The model predicts a revenue baseline of <b>${proj_df['SALES'].mean():,.2f}</b> for the next quarter.</p>
                    <p><b>Business Action:</b> If the orange line is higher than the blue, increase inventory levels immediately.</p>
                </div>
                """, unsafe_allow_html=True)

        # --- UPDATED TAB 5: AI CUSTOMER INTELLIGENCE ---
        with tabs[4]:
            st.header("👥 AI Customer Intelligence")
            st.markdown("---")
            
            # Smart Logic: Segment customers by Value and Order Frequency
            cust_stats = df.groupby('CUSTOMERNAME').agg({'SALES': 'sum', 'ORDERNUMBER': 'nunique'}).reset_index()
            cust_stats.columns = ['Customer', 'Total_Spend', 'Order_Count']
            
            # AI-Style Segmentation
            top_val = cust_stats['Total_Spend'].quantile(0.75)
            def get_segment(row):
                if row['Total_Spend'] >= top_val: return '💎 VIP Platinum'
                if row['Order_Count'] >= 3: return '⭐ Loyal Partner'
                return '🌱 Emerging Client'
            
            cust_stats['Segment'] = cust_stats.apply(get_segment, axis=1)
            
            cust_col1, cust_col2 = st.columns([1, 1])
            with cust_col1:
                st.subheader("Customer Segment Distribution")
                st.plotly_chart(px.pie(cust_stats, names='Segment', hole=0.4, color_discrete_sequence=px.colors.qualitative.Safe), use_container_width=True)
            
            with cust_col2:
                st.subheader("Customer Value vs. Loyalty")
                st.plotly_chart(px.scatter(cust_stats, x='Order_Count', y='Total_Spend', color='Segment', size='Total_Spend', hover_name='Customer', template="plotly_white"), use_container_width=True)
            
            st.markdown("""
            <div class='card'>
                <h4>🎯 Strategy Recommendations</h4>
                <ul>
                    <li><b>VIP Platinum:</b> Offer exclusive bulk-pricing and dedicated support.</li>
                    <li><b>Loyal Partners:</b> Run "Referral Bonus" programs to grow their network.</li>
                    <li><b>Emerging Clients:</b> Send targeted "Welcome Back" discounts to increase order frequency.</li>
                </ul>
            </div>
            """, unsafe_allow_html=True)

else:
    # --- WELCOME PAGE ---
    st.markdown("""<div class="welcome-header"><h1>🚀 PredictiCorp Intelligence</h1><p>Executive AI Suite</p></div>""", unsafe_allow_html=True)
    st.info("👈 Please upload your Sales Data CSV in the sidebar.")
