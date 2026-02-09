import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

# --- PAGE CONFIG ---
st.set_page_config(page_title="AI Sales Predictive System", layout="wide")

# --- WEEK 1: DATA COLLECTION & CLEANING ---
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    # Week 3: Simple Feature Engineering (Extracting date parts)
    df['MONTH'] = df['ORDERDATE'].dt.month
    df['YEAR'] = df['ORDERDATE'].dt.year
    df['DAY_OF_WEEK'] = df['ORDERDATE'].dt.dayofweek
    return df

df = load_data()

# --- SIDEBAR NAVIGATION ---
st.sidebar.title("📊 Project Phases")
page = st.sidebar.radio("Go to", ["Dashboard & EDA", "Sales Prediction Model", "Business Insights"])

# --- PAGE 1: DASHBOARD & EDA (WEEK 2) ---
if page == "Dashboard & EDA":
    st.title("📈 Business Exploratory Data Analysis")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${df['SALES'].sum():,.2f}")
    col2.metric("Total Orders", len(df))
    col3.metric("Avg Order Value", f"${df['SALES'].mean():,.2f}")

    st.subheader("Sales Trends Over Time")
    time_series = df.groupby('ORDERDATE')['SALES'].sum().reset_index()
    fig_line = px.line(time_series, x='ORDERDATE', y='SALES', title="Daily Revenue Trend")
    st.plotly_chart(fig_line, use_container_width=True)

    col_left, col_right = st.columns(2)
    
    with col_left:
        st.subheader("Sales by Product Line")
        prod_sales = df.groupby('PRODUCTLINE')['SALES'].sum().sort_values(ascending=False).reset_index()
        fig_bar = px.bar(prod_sales, x='PRODUCTLINE', y='SALES', color='SALES', color_continuous_scale='Blues')
        st.plotly_chart(fig_bar)

    with col_right:
        st.subheader("Geographic Sales Distribution")
        geo_sales = df.groupby('COUNTRY')['SALES'].sum().reset_index()
        fig_pie = px.pie(geo_sales, values='SALES', names='COUNTRY', hole=0.4)
        st.plotly_chart(fig_pie)

# --- PAGE 2: SALES PREDICTION MODEL (WEEK 4 & 5) ---
elif page == "Sales Prediction Model":
    st.title("🤖 AI Prediction Engine")
    st.write("This model predicts the **Sales Value** of a potential order based on product and timing.")

    # --- WEEK 3: FEATURE ENGINEERING ---
    # Prepare data for ML
    features = ['MONTH', 'YEAR', 'PRODUCTLINE', 'MSRP', 'COUNTRY', 'DEALSIZE']
    X = df[features].copy()
    y = df['SALES']

    # Encoding Categorical Data
    le = LabelEncoder()
    for col in ['PRODUCTLINE', 'COUNTRY', 'DEALSIZE']:
        X[col] = le.fit_transform(X[col])

    # --- WEEK 4: MODEL BUILDING ---
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # --- WEEK 5: EVALUATION ---
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    st.sidebar.subheader("Model Performance")
    st.sidebar.write(f"**MAE:** ${mae:.2f}")
    st.sidebar.write(f"**RMSE:** ${rmse:.2f}")
    st.sidebar.write(f"**R² Score:** {r2:.2f}")

    # --- USER PREDICTION INPUT ---
    st.subheader("Live Prediction Tool")
    c1, c2, c3 = st.columns(3)
    
    with c1:
        in_prod = st.selectbox("Product Line", df['PRODUCTLINE'].unique())
        in_country = st.selectbox("Country", df['COUNTRY'].unique())
    with c2:
        in_msrp = st.number_input("MSRP of Product", min_value=30, max_value=250, value=100)
        in_deal = st.selectbox("Deal Size", df['DEALSIZE'].unique())
    with c3:
        in_month = st.slider("Month", 1, 12, 6)
        in_year = st.selectbox("Year", [2005, 2006])

    if st.button("Predict Order Value"):
        # Process input
        input_data = pd.DataFrame([[in_month, in_year, in_prod, in_msrp, in_country, in_deal]], 
                                  columns=features)
        
        # We need to use the same label encoding as training
        # Simplified for demo: map categorical back to codes
        for col in ['PRODUCTLINE', 'COUNTRY', 'DEALSIZE']:
            temp_le = LabelEncoder()
            temp_le.fit(df[col])
            input_data[col] = temp_le.transform(input_data[col])
        
        prediction = model.predict(input_data)
        st.success(f"### Predicted Sales Value: ${prediction[0]:,.2f}")
        st.info("The model suggests this order size based on historical patterns for this product and region.")

# --- PAGE 3: BUSINESS INSIGHTS (WEEK 6) ---
elif page == "Business Insights":
    st.title("💡 Strategic Recommendations")
    
    # Simple logic-based insights
    top_country = df.groupby('COUNTRY')['SALES'].sum().idxmax()
    top_prod = df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
    peak_month = df.groupby('MONTH_ID')['SALES'].sum().idxmax()
    
    st.markdown(f"""
    ### Key Takeaways for Management:
    1. **Primary Market:** Your strongest market is **{top_country}**. Consider increasing marketing budget here.
    2. **Star Product:** **{top_prod}** is your highest revenue generator. Ensure supply chain priority for this line.
    3. **Seasonality:** Historical data shows peak demand occurs in **Month {peak_month}**. Prepare inventory 2 months in advance.
    4. **Deal Strategy:** Medium-sized deals contribute the most to consistent cash flow compared to rare Large deals.
    """)
    
    # Feature Importance visualization
    st.subheader("What drives Sales?")
    st.write("Based on the Machine Learning model, these factors impact revenue the most:")
    # (Note: Feature importance needs the model from Page 2)
    # Re-running logic briefly for chart
    features = ['MONTH', 'YEAR', 'PRODUCTLINE', 'MSRP', 'COUNTRY', 'DEALSIZE']
    X_sample = df[features].copy()
    for col in ['PRODUCTLINE', 'COUNTRY', 'DEALSIZE']:
        X_sample[col] = LabelEncoder().fit_transform(X_sample[col])
    m = RandomForestRegressor().fit(X_sample, df['SALES'])
    
    imp_df = pd.DataFrame({'Feature': features, 'Importance': m.feature_importances_}).sort_values(by='Importance', ascending=False)
    fig_imp = px.bar(imp_df, x='Importance', y='Feature', orientation='h')
    st.plotly_chart(fig_imp)
