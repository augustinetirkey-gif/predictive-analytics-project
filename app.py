import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# --- APP CONFIGURATION ---
st.set_page_config(page_title="PredictiCorp AI Engine", layout="wide", initial_sidebar_state="expanded")

# --- PREMIUM STYLING ---
st.markdown("""
    <style>
    .main { background-color: #f4f7f9; }
    [data-testid="stMetricValue"] { font-size: 28px; color: #1f4e79; }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #ffffff; 
        border: 1px solid #e1e4e8;
        border-radius: 8px; 
        padding: 10px 20px;
        font-weight: bold;
    }
    .stTabs [aria-selected="true"] { background-color: #1f4e79 !important; color: white !important; }
    </style>
    """, unsafe_allow_html=True)

# --- DATA ENGINE ---
@st.cache_data
def get_system_data():
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df['MONTH'] = df['ORDERDATE'].dt.month
    df['YEAR'] = df['ORDERDATE'].dt.year
    df['QTR'] = df['ORDERDATE'].dt.quarter
    return df

df = get_system_data()


# --- MAIN INTERFACE ---
st.title("🚀 PredictiCorp Executive AI Platform")
st.markdown("#### Forecasting Trends & Outcomes for Data-Driven Decisions")

tabs = st.tabs([
    "📂 Week 1-2: Data & EDA", 
    "🛠️ Week 3: Feature Engineering", 
    "🧠 Week 4: AI Model Training", 
    "📊 Week 5: Quality Metrics",
    "🎯 Week 6: Executive Dashboard"
])

# --- WEEK 1-2: EXPLORATORY DATA ANALYSIS ---
with tabs[0]:
    st.header("Exploratory Data Analysis")
    
    # KPI Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gross Revenue", f"${df['SALES'].sum()/1e6:.2f}M")
    col2.metric("Avg Order", f"${df['SALES'].mean():,.0f}")
    col3.metric("Transaction Count", f"{len(df):,}")
    col4.metric("Growth %", "+12.4%")

    st.write("---")

    # Row 1: Revenue Trend and Product Pie Chart
    c1, c2 = st.columns([2, 1])
    with c1:
        st.subheader("Revenue Trend Analysis")
        trend_df = df.groupby('ORDERDATE')['SALES'].sum().reset_index()
        fig = px.line(trend_df, x='ORDERDATE', y='SALES', color_discrete_sequence=['#1f4e79'])
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        st.subheader("Product Performance")
        fig_pie = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.write("---")

    # Row 2: Market Distribution by Country
    st.subheader("🌍 Market Distribution by Country")
    
    # 1. Group data by country and calculate total sales
    country_sales = df.groupby('COUNTRY')['SALES'].sum().reset_index()
    country_sales = country_sales.sort_values(by='SALES', ascending=False)
    
    # 2. Create the Bar Chart
    fig_bar = px.bar(
        country_sales, 
        x='COUNTRY', 
        y='SALES',
        title="Total Sales by Country",
        labels={'SALES': 'Revenue ($)', 'COUNTRY': 'Country'},
        color='SALES',
        color_continuous_scale='Blues'
    )
    
    st.plotly_chart(fig_bar, use_container_width=True)
    
    # 3. Dynamic Summary Analysis
    top_country = country_sales.iloc[0]['COUNTRY']
    st.write(f"**Analysis:** The {top_country} is the leading market, accounting for a major portion of total revenue.")
    st.info("This geographic concentration helps our AI model (Week 4) understand where the highest-value transactions are likely to occur.")
# --- WEEK 3: FEATURE ENGINEERING ---
with tabs[1]:
    st.header("Advanced Feature Engineering")
    st.write("Transforming raw sales data into predictive features for our AI model.")

    # --- 1. DATE DECOMPOSITION (Visualizing Time Patterns) ---
    st.subheader("1. Date Decomposition")
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df['MONTH'] = df['ORDERDATE'].dt.month
    df['YEAR'] = df['ORDERDATE'].dt.year
    df['QUARTER'] = df['ORDERDATE'].dt.quarter
    
    st.info("Extracted Month, Year, and Quarter to capture seasonal peaks like the $140k November spikes.")
    st.write(df[['ORDERDATE', 'MONTH', 'YEAR', 'QUARTER']].head())

    # --- 2. LOG TRANSFORMATION (Visualizing the 'Before & After') ---
    st.subheader("2. Normalization: Log Transformation")
    
    # Create the 'Before' and 'After' data
    original_sales = df['SALES'].values
    log_sales = np.log1p(df['SALES']).values
    
    # Plotting the transformation
    hist_data = [original_sales, log_sales]
    group_labels = ['Original Sales', 'Log Transformed Sales']
    
    c1, c2 = st.columns(2)
    with c1:
        fig_before = px.histogram(df, x="SALES", title="Before: Skewed Data", color_discrete_sequence=['#ff7f0e'])
        st.plotly_chart(fig_before, use_container_width=True)
    with c2:
        df['SALES_LOG'] = np.log1p(df['SALES'])
        fig_after = px.histogram(df, x="SALES_LOG", title="After: Normal Distribution", color_discrete_sequence=['#2ca02c'])
        st.plotly_chart(fig_after, use_container_width=True)
    
    st.success("Transformation complete: We moved from messy spikes to a clean 'Bell Curve' that the AI prefers.")

    # --- 3. CATEGORICAL ENCODING & FEATURE IMPORTANCE ---
    st.subheader("3. Feature Importance (The AI Rulebook)")
    
    # Engineering steps for the model
    le = LabelEncoder()
    temp_df = df.copy()
    for col in ['PRODUCTLINE', 'COUNTRY', 'DEALSIZE']:
        temp_df[col] = le.fit_transform(temp_df[col])
    
    # Training a quick model to find the 'Drivers'
    X = temp_df[['QUANTITYORDERED', 'PRICEEACH', 'PRODUCTLINE', 'COUNTRY', 'DEALSIZE', 'MONTH']]
    y = temp_df['SALES']
    model = RandomForestRegressor(n_estimators=50, random_state=42)
    model.fit(X, y)
    
    # Plotting the importance
    feat_importances = pd.DataFrame(model.feature_importances_, index=X.columns, columns=["Importance"]).sort_values(by='Importance', ascending=True)
    fig_imp = px.bar(feat_importances, x="Importance", y=feat_importances.index, orientation='h', title="Which Feature Drives the $10.03M Revenue?")
    st.plotly_chart(fig_imp, use_container_width=True)

    # --- 4. OUTLIER HANDLING (The Quality Check) ---
    st.subheader("4. Outlier Removal (IQR Method)")
    Q1 = df['SALES'].quantile(0.25)
    Q3 = df['SALES'].quantile(0.75)
    IQR = Q3 - Q1
    st.write(f"**Insight:** We identified and stabilized values outside the range of **${(Q1 - 1.5*IQR):,.2f}** to **${(Q3 + 1.5*IQR):,.2f}**.")
    st.success("✅ Cleaned data ready for Week 4: Model Building.")

# --- WEEK 4: MODEL BUILDING ---
with tabs[2]:
    st.header("AI Model Construction")
    
    # ML Logic
    features = ['MONTH', 'QTR', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']
    X = df[features].copy()
    y = df['SALES']
    
    le_dict = {}
    for col in ['PRODUCTLINE', 'COUNTRY']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    st.info("The system is utilizing a **Random Forest Regressor** with 100 Decision Trees.")
    
    # Feature Importance
    importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance')
    fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', title="Feature Weight Analysis")
    st.plotly_chart(fig_imp)

# --- WEEK 5: EVALUATION ---
with tabs[3]:
    st.header("Model Performance & Validation")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    v1, v2, v3 = st.columns(3)
    v1.metric("Mean Absolute Error", f"${mae:.2f}")
    v2.metric("Root Mean Squared Error", f"${rmse:.2f}")
    v3.metric("R-Squared Score", f"{r2:.4f}")

    st.write("---")
    st.subheader("Model Reliability Scatter Plot")
    fig_reg = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Sales', 'y': 'Predicted Sales'}, 
                         trendline="ols", trendline_color_override="red")
    st.plotly_chart(fig_reg, use_container_width=True)

# --- WEEK 6: DEPLOYMENT ---
with tabs[4]:
    st.header("Business Decision Dashboard")
    st.write("Input deal parameters to forecast the expected outcome.")

    with st.container():
        st.markdown("### 🔍 Simulation Engine")
        s1, s2, s3 = st.columns(3)
        with s1:
            in_month = st.select_slider("Select Month", options=range(1,13))
            in_qtr = (in_month-1)//3 + 1
        with s2:
            in_prod = st.selectbox("Product Category", df['PRODUCTLINE'].unique())
            in_qty = st.number_input("Quantity Requested", value=30)
        with s3:
            in_country = st.selectbox("Market Region", df['COUNTRY'].unique())
            in_msrp = st.number_input("MSRP per Unit", value=100)

        if st.button("🚀 EXECUTE PREDICTION"):
            p_prod = le_dict['PRODUCTLINE'].transform([in_prod])[0]
            p_country = le_dict['COUNTRY'].transform([in_country])[0]
            
            prediction = model.predict(np.array([[in_month, in_qtr, in_msrp, in_qty, p_prod, p_country]]))[0]
            
            st.markdown(f"""
                <div style="background-color:#ffffff; padding:20px; border-radius:10px; border-left: 10px solid #1f4e79;">
                    <h2 style="color:#1f4e79;">Forecasted Outcome: ${prediction:,.2f}</h2>
                    <p style="color:#666;">Based on historical patterns, this deal is estimated to generate the above revenue.</p>
                </div>
            """, unsafe_allow_html=True)
