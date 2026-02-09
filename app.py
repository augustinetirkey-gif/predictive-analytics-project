import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# --- 1. PREMIUM CONFIGURATION ---
st.set_page_config(page_title="Pro AI Analytics", layout="wide", initial_sidebar_state="expanded")

# Custom CSS for a professional "Dark Mode" feel or Clean Corporate look
st.markdown("""
    <style>
    .reportview-container { background: #f0f2f6; }
    .stMetric { background-color: #ffffff; border-radius: 10px; padding: 15px; border: 1px solid #e0e0e0; box-shadow: 2px 2px 5px rgba(0,0,0,0.05); }
    </style>
    """, unsafe_allow_html=True)

# --- 2. DATA ENGINE (WEEK 1 & 3) ---
@st.cache_data
def load_and_prep():
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    df['MONTH'] = df['ORDERDATE'].dt.month
    df['QUARTER'] = df['ORDERDATE'].dt.quarter
    df['YEAR'] = df['ORDERDATE'].dt.year
    return df

try:
    df = load_and_prep()
except:
    st.error("Please upload 'cleaned_sales_data.csv'")
    st.stop()

# --- 3. SIDEBAR CONTROLS (THE "PRO" TOUCH) ---
st.sidebar.title("🛠️ Control Panel")
st.sidebar.info("Adjust parameters to see real-time impact on AI predictions.")

# Global Currency Toggle (Kept in USD as requested)
currency_symbol = "$"

# --- 4. THE AUTO-ML ENGINE (WEEK 4 & 5) ---
# We automate the selection of the best model
features = ['MONTH', 'QUARTER', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']
X = df[features].copy()
y = df['SALES']

le_dict = {}
for col in ['PRODUCTLINE', 'COUNTRY']:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col])
    le_dict[col] = le

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

@st.cache_resource
def train_best_model():
    models = {
        "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "Gradient Boosting": GradientBoostingRegressor(random_state=42),
        "Linear Regression": LinearRegression()
    }
    
    best_m = None
    best_score = -float('inf')
    results = {}

    for name, m in models.items():
        m.fit(X_train, y_train)
        score = m.score(X_test, y_test)
        results[name] = score
        if score > best_score:
            best_score = score
            best_m = m
    return best_m, results

best_model, all_model_results = train_best_model()

# --- 5. MAIN INTERFACE ---
st.title("🚀 Advanced AI Predictive Analytics")
st.markdown(f"**Current Champion Model:** `{type(best_model).__name__}` (Accuracy: {max(all_model_results.values()):.2%})")

tabs = st.tabs(["📈 Executive Dashboard", "🧠 AI Training Lab", "🔮 Future Forecaster"])

# --- TAB 1: EXECUTIVE DASHBOARD (WEEK 2) ---
with tabs[0]:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Revenue", f"{currency_symbol}{df['SALES'].sum():,.0f}")
    col2.metric("Avg Order", f"{currency_symbol}{df['SALES'].mean():,.2f}")
    col3.metric("Total Orders", len(df))
    col4.metric("Best Market", df.groupby('COUNTRY')['SALES'].sum().idxmax())

    c1, c2 = st.columns([6, 4])
    with c1:
        # Time Series Trend
        trend_data = df.groupby('ORDERDATE')['SALES'].sum().reset_index()
        fig_trend = px.line(trend_data, x='ORDERDATE', y='SALES', title="Historical Sales Trend", color_discrete_sequence=['#00CC96'])
        st.plotly_chart(fig_trend, use_container_width=True)
    with c2:
        fig_pie = px.pie(df, values='SALES', names='PRODUCTLINE', hole=0.5, title="Revenue Mix")
        st.plotly_chart(fig_pie, use_container_width=True)

# --- TAB 2: AI TRAINING LAB (WEEK 4 & 5) ---
with tabs[1]:
    st.header("🧪 Model Performance Comparison")
    res_df = pd.DataFrame(list(all_model_results.items()), columns=['Model', 'R² Score'])
    fig_res = px.bar(res_df, x='Model', y='R² Score', color='R² Score', text_auto='.2f', title="AutoML: Finding the Best Fit")
    st.plotly_chart(fig_res, use_container_width=True)
    
    st.markdown("""
    > **What is R²?** It represents how much of the sales variance is explained by the AI. 
    > A score of **1.00** would be a perfect prediction.
    """)

# --- TAB 3: FUTURE FORECASTER (WEEK 6) ---
with tabs[2]:
    st.header("🔮 Strategic Scenario Simulator")
    st.write("Simulate any business scenario to see the predicted outcome.")
    
    with st.expander("Configure Scenario Details", expanded=True):
        f_col1, f_col2, f_col3 = st.columns(3)
        with f_col1:
            in_month = st.slider("Month of Year", 1, 12, 6)
            in_qtr = (in_month-1)//3 + 1
        with f_col2:
            in_prod = st.selectbox("Select Product Line", df['PRODUCTLINE'].unique())
            in_qty = st.number_input("Quantity to be Ordered", min_value=1, value=30)
        with f_col3:
            in_country = st.selectbox("Destination Country", df['COUNTRY'].unique())
            in_msrp = st.number_input("Unit MSRP ($)", value=100)

    if st.button("🚀 Run AI Prediction"):
        # Process Input
        p_prod = le_dict['PRODUCTLINE'].transform([in_prod])[0]
        p_country = le_dict['COUNTRY'].transform([in_country])[0]
        input_data = np.array([[in_month, in_qtr, in_msrp, in_qty, p_prod, p_country]])
        
        # Predict
        pred = best_model.predict(input_data)[0]
        
        # UI Results
        st.divider()
        res_col1, res_col2 = st.columns(2)
        with res_col1:
            st.metric("Predicted Transaction Value", f"{currency_symbol}{pred:,.2f}", delta=f"{((pred/df['SALES'].mean())-1)*100:.1f}% vs Avg")
        with res_col2:
            if pred > df['SALES'].mean():
                st.success("✅ **High Value Target**: This scenario is predicted to exceed your historical average.")
            else:
                st.warning("⚠️ **Low Margin Alert**: This scenario may result in below-average revenue.")

        # Interactive Forecast Chart
        fig_gauge = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = pred,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "Prediction Confidence Meter"},
            gauge = {'axis': {'range': [None, df['SALES'].max()]},
                     'bar': {'color': "#00CC96"},
                     'steps' : [
                         {'range': [0, df['SALES'].mean()], 'color': "lightgray"}]}))
        st.plotly_chart(fig_gauge)
