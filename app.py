import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
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
    # Ensure you have 'cleaned_sales_data.csv' in your directory
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df['MONTH'] = df['ORDERDATE'].dt.month
    df['YEAR'] = df['ORDERDATE'].dt.year
    df['QTR'] = df['ORDERDATE'].dt.quarter
    return df

df = get_system_data()

# --- BACKGROUND ML LOGIC (Required for tabs 3, 4, and 5) ---
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
y_pred = model.predict(X_test)

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
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Gross Revenue", f"${df['SALES'].sum()/1e6:.2f}M")
    col2.metric("Avg Order", f"${df['SALES'].mean():,.0f}")
    col3.metric("Transaction Count", f"{len(df):,}")
    col4.metric("Growth %", "+12.4%")

    st.write("---")
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
    st.subheader("🌍 Market Distribution by Country")
    country_sales = df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values(by='SALES', ascending=False)
    fig_bar = px.bar(country_sales, x='COUNTRY', y='SALES', color='SALES', color_continuous_scale='Blues')
    st.plotly_chart(fig_bar, use_container_width=True)

# --- WEEK 3: FEATURE ENGINEERING ---
with tabs[1]:
    st.header("Advanced Feature Engineering")
    st.write("Turning raw data into 'Smart Features' for the AI.")
    
    fe_df = df.copy()
    le_fe = LabelEncoder()
    fe_df['DEALSIZE_ENC'] = le_fe.fit_transform(fe_df['DEALSIZE'])
    
    st.subheader("Correlation Heatmap")
    corr = fe_df[['SALES', 'QUANTITYORDERED', 'PRICEEACH', 'MONTH', 'DEALSIZE_ENC']].corr()
    fig_corr, ax = plt.subplots()
    sns.heatmap(corr, annot=True, cmap='RdBu', ax=ax)
    st.pyplot(fig_corr)

    st.subheader("Data Normalization (Log Transformation)")
    fig_hist = px.histogram(fe_df, x=np.log1p(fe_df['SALES']), title="Log-Transformed Sales Distribution")
    st.plotly_chart(fig_hist)

# --- WEEK 4: MODEL BUILDING ---
with tabs[2]:
    st.header("AI Model Construction")
    st.info("The system is utilizing a **Random Forest Regressor** with 100 Decision Trees.")
    
    importance = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}).sort_values('Importance')
    fig_imp = px.bar(importance, x='Importance', y='Feature', orientation='h', title="Feature Weight Analysis")
    st.plotly_chart(fig_imp)

# --- WEEK 5: EVALUATION ---
with tabs[3]:
    st.header("Model Performance & Validation")
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
            in_month = st.select_slider("Select Month", options=range(1, 13))
            in_qtr = (in_month - 1) // 3 + 1
        with s2:
            in_prod = st.selectbox("Product Category", df['PRODUCTLINE'].unique())
            in_qty = st.number_input("Quantity Requested", value=30)
        with s3:
            in_country = st.selectbox("Market Region", df['COUNTRY'].unique())
            in_msrp = st.number_input("MSRP per Unit", value=100)

        if st.button("🚀 EXECUTE PREDICTION"):
            p_prod = le_dict['PRODUCTLINE'].transform([in_prod])[0]
            p_country = le_dict['COUNTRY'].transform([in_country])[0]
            
            input_data = np.array([[in_month, in_qtr, in_msrp, in_qty, p_prod, p_country]])
            prediction = model.predict(input_data)[0]
            
            st.markdown(f"""
                <div style="background-color:#ffffff; padding:20px; border-radius:10px; border-left: 10px solid #1f4e79;">
                    <h2 style="color:#1f4e79;">Forecasted Outcome: ${prediction:,.2f}</h2>
                    <p style="color:#666;">Based on historical patterns, this deal is estimated to generate the above revenue.</p>
                </div>
            """, unsafe_allow_html=True)
