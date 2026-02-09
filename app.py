import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, rmse_score, r2_score
from sklearn.preprocessing import LabelEncoder
# Change this:
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# And later in the code where you calculate RMSE, use this:
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

# --- APP CONFIGURATION ---
st.set_page_config(page_title="AI Predictive Platform", layout="wide")

# Custom CSS for Professional Branding
st.markdown("""
    <style>
    .main { background-color: #f0f2f6; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #ffffff; border-radius: 5px; padding: 10px; }
    .stMetric { border: 1px solid #d1d5db; padding: 15px; border-radius: 10px; background: white; }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_and_clean():
    # Week 1: Data Understanding & Collection
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    # Week 3: Feature Engineering
    df['MONTH'] = df['ORDERDATE'].dt.month
    df['YEAR'] = df['ORDERDATE'].dt.year
    df['QUARTER'] = df['ORDERDATE'].dt.quarter
    return df

df = load_and_clean()

# --- HEADER SECTION ---
st.title("🎯 AI-Based Predictive Analytics Platform")
st.caption("Internship Project: Forecasting Trends & Business Outcomes")
st.write("---")

# --- 6-WEEK PROJECT WORKFLOW TABS ---
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📍 Week 1-2: EDA & Insights", 
    "📍 Week 3: Feature Engineering", 
    "📍 Week 4: Model Building", 
    "📍 Week 5: Evaluation",
    "📍 Week 6: Deployment Dashboard"
])

# --- WEEK 1 & 2: EXPLORATORY DATA ANALYSIS ---
with tab1:
    st.header("📊 Exploratory Data Analysis (EDA)")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Total Revenue", f"${df['SALES'].sum():,.2f}")
    with c2:
        st.metric("Top Category", df.groupby('PRODUCTLINE')['SALES'].sum().idxmax())
    with c3:
        st.metric("Global Reach", f"{df['COUNTRY'].nunique()} Countries")

    st.subheader("Revenue Distribution by Geography")
    fig_map = px.choropleth(df.groupby('COUNTRY')['SALES'].sum().reset_index(), 
                            locations="COUNTRY", locationmode='country names', 
                            color="SALES", hover_name="COUNTRY", color_continuous_scale="Viridis")
    st.plotly_chart(fig_map, use_container_width=True)

# --- WEEK 3: FEATURE ENGINEERING ---
with tab2:
    st.header("⚙️ Feature Transformation")
    st.write("In this phase, we converted raw dates and categories into mathematical features.")
    
    col_f1, col_f2 = st.columns(2)
    with col_f1:
        st.write("**Before Transformation:**")
        st.dataframe(df[['ORDERDATE', 'PRODUCTLINE', 'COUNTRY']].head(3))
    with col_f2:
        st.write("**After Encoding:**")
        # Visual representation of the logic
        encoded_sample = df[['MONTH', 'QUARTER', 'MSRP']].head(3)
        st.dataframe(encoded_sample)
    
    st.info("💡 Insight: Added 'Month' and 'Quarter' features to capture seasonality spikes found during EDA.")

# --- WEEK 4: MODEL BUILDING ---
with tab3:
    st.header("🤖 Machine Learning Model (Random Forest)")
    
    # Model Preparation
    features = ['MONTH', 'QUARTER', 'MSRP', 'QUANTITYORDERED', 'PRODUCTLINE', 'COUNTRY']
    X = df[features].copy()
    y = df['SALES']
    
    # Label Encoding for categorical columns
    le_dict = {}
    for col in ['PRODUCTLINE', 'COUNTRY']:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        le_dict[col] = le

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    st.success("Model trained successfully using Random Forest Regressor.")
    
    st.subheader("Feature Importance (How the AI Decides)")
    importance = pd.DataFrame({'Feature': features, 'Weight': model.feature_importances_}).sort_values('Weight', ascending=False)
    fig_imp = px.bar(importance, x='Weight', y='Feature', orientation='h', color='Weight')
    st.plotly_chart(fig_imp)

# --- WEEK 5: EVALUATION & OPTIMIZATION ---
with tab4:
    st.header("📉 Model Performance Metrics")
    y_pred = model.predict(X_test)
    
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_absolute_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE (Mean Absolute Error)", f"${mae:.2f}")
    m2.metric("RMSE", f"${rmse:.2f}")
    m3.metric("R² Score", f"{r2:.4f}")
    
    st.subheader("Actual vs. Predicted Sales")
    fig_res = px.scatter(x=y_test, y=y_pred, labels={'x': 'Actual Sales', 'y': 'Predicted Sales'}, 
                         trendline="ols", title="Model Reliability Check")
    st.plotly_chart(fig_res, use_container_width=True)

# --- WEEK 6: DEPLOYMENT & REPORTING ---
with tab5:
    st.header("🚀 Predictive Dashboard for Decision Makers")
    
    with st.expander("📝 Generate Business Prediction"):
        col_in1, col_in2, col_in3 = st.columns(3)
        with col_in1:
            in_month = st.slider("Select Forecast Month", 1, 12, 6)
            in_qtr = (in_month-1)//3 + 1
        with col_in2:
            in_prod = st.selectbox("Product Line", df['PRODUCTLINE'].unique())
            in_qty = st.number_input("Quantity Ordered", value=35)
        with col_in3:
            in_country = st.selectbox("Market Country", df['COUNTRY'].unique())
            in_msrp = st.number_input("MSRP", value=100)

        if st.button("Run Business Prediction"):
            # Encode Input
            p_prod = le_dict['PRODUCTLINE'].transform([in_prod])[0]
            p_country = le_dict['COUNTRY'].transform([in_country])[0]
            
            input_array = np.array([[in_month, in_qtr, in_msrp, in_qty, p_prod, p_country]])
            prediction = model.predict(input_array)[0]
            
            st.markdown(f"### 🔮 Predicted Order Value: **${prediction:,.2f}**")
            
            # Scenario Analysis
            if prediction > df['SALES'].mean():
                st.warning("High Value Transaction: Recommend priority fulfillment and manager approval.")
            else:
                st.info("Standard Transaction: Automated processing recommended.")

    st.write("---")
    st.subheader("Final Project Summary")
    st.write("""
    - **Goal:** Shift from experience-based to data-driven decision making.
    - **Outcome:** System can forecast revenue with an R² accuracy of **{:.2f}**.
    - **Impact:** Identified MSRP and Quantity as the primary drivers of revenue variance.
    """.format(r2))
