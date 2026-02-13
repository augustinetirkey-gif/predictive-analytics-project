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

# --- EXECUTIVE THEMING (Refactored for Theme Support) ---
st.markdown("""
    <style>
    /* Use transparent borders and adaptive shadows for theme compatibility */
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
    /* Use dynamic variables for the KPI Box */
    [data-testid="stMetric"] {
        background-color: rgba(128, 128, 128, 0.05); /* Light translucent grey */
        border: 1px solid rgba(128, 128, 128, 0.2);
        padding: 15px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    /* Add a nice hover effect to make it 'Attractive' */
    [data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        border-color: #1f4e79;
        box-shadow: 0 8px 20px rgba(31, 78, 121, 0.2);
    }

    /* Fix the Label (Small text) visibility */
    [data-testid="stMetricLabel"] p {
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    /* Fix the Value (Big number) visibility */
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
            
            st.info(f"💡 *Historical Price Context for {in_prod}:* Avg: ${avg_msrp:.2f} | Range: ${min_msrp:.2f} - ${max_msrp:.2f}")
            
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
                else:
                    st.warning("No historical data found for this specific combination to show a comparison.")      
        # TAB 3: Strategic Market Insights (UPGRADED VERSION)
        with tabs[2]:
            st.header("🌍 Strategic Market Insights")
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
            
           

            # --- 1. Heatmap and Top/Bottom Tables ---
            c3, c4 = st.columns([2, 1])
            with c3:
                st.markdown("#### Revenue Heatmap: Country × Product Line")
                heat_df = df.pivot_table(index='COUNTRY', columns='PRODUCTLINE', values='SALES', aggfunc='sum').fillna(0)
                # Spectral_r uses a distinct multi-color spectrum for high visibility
                fig_heat = px.imshow(heat_df, text_auto='.2s', aspect="auto", 
                                     color_continuous_scale="Spectral_r", 
                                     template="plotly_white")
                st.plotly_chart(fig_heat, use_container_width=True)
            
            with c4:
                st.markdown("#### Top 5 vs Bottom 5 Markets")
                m_sorted = df.groupby('COUNTRY')['SALES'].sum().sort_values(ascending=False).reset_index()
                st.write("*Top 5 Markets*")
                st.dataframe(m_sorted.head(5), hide_index=True, use_container_width=True)
                st.write("*Bottom 5 Markets*")
                st.dataframe(m_sorted.tail(5), hide_index=True, use_container_width=True)

            # --- 2. Growth Trend and Contribution Donut ---
            c5, c6 = st.columns(2)
            with c5:
                st.markdown("#### YoY Revenue Performance")
                growth_trend = df.groupby(['YEAR', 'MONTH_ID'])['SALES'].sum().reset_index()
                fig_growth = px.bar(growth_trend, x='MONTH_ID', y='SALES', color='YEAR', 
                                    barmode='group', template="plotly_white")
                st.plotly_chart(fig_growth, use_container_width=True)
            
            with c6:
                st.markdown("#### Product Revenue Contribution (%)")
                fig_don = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, 
                                 template="plotly_white", color_discrete_sequence=px.colors.qualitative.Pastel)
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

    # TAB 5: Customer Analytics (Advanced Intelligence Suite)
        with tabs[4]:
            st.header("👥 Customer Intelligence & Loyalty")
            
            # --- 1. DATA PREPARATION (RFM Logic) ---
            current_date = df['ORDERDATE'].max()
            cust_metrics = df.groupby('CUSTOMERNAME').agg({
                'SALES': 'sum',
                'ORDERNUMBER': 'nunique',
                'ORDERDATE': 'max',
                'COUNTRY': 'first',
                'PHONE': 'first',
                'DEALSIZE': lambda x: x.mode()[0] # Capture most frequent deal size
            }).reset_index()
            
            cust_metrics.columns = ['Customer', 'Revenue', 'Frequency', 'LastOrder', 'Country', 'Phone', 'Typical_Deal']
            cust_metrics['Recency'] = (current_date - cust_metrics['LastOrder']).dt.days
            
            # --- 2. CUSTOMER SEGMENTATION ---
            st.subheader("📊 Strategic Customer Segmentation")
            # Segmenting by Revenue Tiers
            cust_metrics['Deal size'] = pd.qcut(cust_metrics['Revenue'], q=3, labels=['Small', 'Medium', 'Large (VIP)'])
            
            col_s1,  = st.columns([1])
            with col_s1:
                fig_seg = px.pie(cust_metrics, names='Deal size', hole=0.4, 
                                 color_discrete_sequence=px.colors.qualitative.Pastel,
                                 title="Customer Base by Value Tier")
                st.plotly_chart(fig_seg, use_container_width=True)
            
            
            
            # --- 3. GEOGRAPHIC DISTRIBUTION ---
            st.divider()
            st.subheader("🌍 Customer Geographic Footprint")
            fig_geo = px.scatter_geo(cust_metrics, locations="Country", locationmode='country names',
                                     size="Revenue", color="deal size", hover_name="Customer",
                                     template="plotly", projection="natural earth")
            st.plotly_chart(fig_geo, use_container_width=True)
            
               # --- 4. REVENUE CONCENTRATION (80/20 Rule) ---
            st.divider()
            st.subheader("🎯 Revenue Concentration Analysis")

            # Sort customers by revenue and calculate percentage of total revenue
            pareto_df = cust_metrics.sort_values('Revenue', ascending=False).copy()
            pareto_df['Revenue_Share'] = (pareto_df['Revenue'].cumsum() / pareto_df['Revenue'].sum()) * 100
            pareto_df['Customer_Count_Pct'] = np.arange(1, len(pareto_df) + 1) / len(pareto_df) * 100

            fig_pareto = px.area(pareto_df, x='Customer_Count_Pct', y='Revenue_Share',
                                 title="The Pareto Curve: % of Customers vs. % of Total Revenue",
                                 labels={'Customer_Count_Pct': '% of Total Customers', 'Revenue_Share': '% of Total Revenue'},
                                 template="plotly")

            # Add a reference line for the 80/20 rule
            fig_pareto.add_hline(y=80, line_dash="dash", line_color="red", annotation_text="80% Revenue Mark")
            fig_pareto.update_traces(line_color='#1f4e79', fillcolor='rgba(31, 78, 121, 0.2)')

            st.plotly_chart(fig_pareto, use_container_width=True)
            st.info("💡 **Insight:** If the curve rises very steeply, your business is heavily dependent on a few top customers.")


            # --- 5. TOP 10 HIGH-VALUE CLIENT DOSSIER & CHURN RISK ---
            st.divider()
            col_g1, col_g2 = st.columns(2)
            
            with col_g1:
                st.subheader("🏆 Top 10 High-Value Client Dossier")
                top_10_df = cust_metrics.sort_values('Revenue', ascending=False).head(10)
                st.dataframe(
                    top_10_df[['Customer', 'Revenue', 'Recency', 'Country', 'Phone', 'Typical_Deal']],
                    column_config={
                        "Revenue": st.column_config.NumberColumn("Total Spend", format="$%.2f"),
                        "Recency": st.column_config.NumberColumn("Days Since Last Order", format="%d d"),
                    },
                    use_container_width=True, 
                    hide_index=True
                )

            with col_g2:
                st.subheader("🚩 Churn Risk Analysis")
                # Flagging customers who haven't ordered in a long time (Recency > 120 days)
                churn_df = cust_metrics[cust_metrics['Recency'] > 120].sort_values('Revenue', ascending=False)
                st.write(f"*Found {len(churn_df)} customers at risk (No orders in 120+ days)*")
                st.dataframe(
                    churn_df[['Customer', 'Revenue', 'Recency', 'Country', 'Phone']].head(10), 
                    column_config={"Revenue": st.column_config.NumberColumn(format="$%.2f")},
                    use_container_width=True, 
                    hide_index=True
                )

            # --- 6. PRODUCT PREFERENCE HEATMAP (Multi-Color) ---
            st.divider()
            st.subheader("🧩 Product Affinity Heatmap")
            st.write("Correlation between top customers and product line preferences.")
            
            # Pivot data for heatmap: Top 25 customers by revenue
            top_cust_names = cust_metrics.nlargest(25, 'Revenue')['Customer']
            heat_data = df[df['CUSTOMERNAME'].isin(top_cust_names)].pivot_table(
                index='CUSTOMERNAME', columns='PRODUCTLINE', values='SALES', aggfunc='sum'
            ).fillna(0)
            
            # Using the 'RdYlBu_r' color scale for high distinction
            fig_heat = px.imshow(heat_data, text_auto='.2s', aspect="auto",
                                 color_continuous_scale='RdYlBu_r', 
                                 template="plotly")
            st.plotly_chart(fig_heat, use_container_width=True)
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
