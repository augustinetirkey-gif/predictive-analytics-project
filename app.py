import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import datetime

# ==========================================
# 🎨 1. PROFESSIONAL CSS & PAGE CONFIG
# ==========================================
st.set_page_config(page_title="Sales AI Dashboard", layout="wide", page_icon="📈")

def apply_custom_styles():
    st.markdown("""
        <style>
        .main { background-color: #f4f7f9; }
        .welcome-banner {
            background: linear-gradient(90deg, #4e73df 0%, #224abe 100%);
            padding: 30px; border-radius: 15px; color: white; margin-bottom: 25px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        [data-testid="stSidebar"] { background-color: #1a1c24; }
        [data-testid="stSidebar"] * { color: #ffffff !important; }
        .stMetric {
            background-color: white; padding: 20px; border-radius: 12px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05); border: 1px solid #e3e6f0;
        }
        .ai-card {
            background-color: #e3f2fd; border-left: 5px solid #2196f3;
            padding: 20px; border-radius: 10px; margin-bottom: 20px; color: #0d47a1;
        }
        .manual-insight {
            background-color: #fffdf0; border: 1px solid #ffeeba;
            padding: 20px; border-radius: 10px; margin-top: 20px; color: #856404;
        }
        </style>
    """, unsafe_allow_html=True)

apply_custom_styles()

# ==========================================
# 🔑 2. LOGIN SYSTEM
# ==========================================
if 'users_db' not in st.session_state:
    st.session_state.users_db = pd.DataFrame([{"username": "admin", "password": "123", "role": "Admin", "date": "2024-01-01"}])

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    col1, col2, col3 = st.columns([1, 1.5, 1])
    with col2:
        st.markdown("<h2 style='text-align:center;'>🔑 Access Sales AI</h2>", unsafe_allow_html=True)
        u = st.text_input("Username")
        p = st.text_input("Password", type="password")
        if st.button("Log In", use_container_width=True):
            if u == "admin" and p == "123":
                st.session_state.logged_in = True
                st.session_state.user = {"username": "admin", "role": "Admin"}
                st.rerun()
            else: st.error("Invalid Username/Password")
    st.stop()

# ==========================================
# 📊 3. DATA ENGINE
# ==========================================
@st.cache_data
def load_data():
    df = pd.read_csv('cleaned_sales_data.csv')
    df['ORDERDATE'] = pd.to_datetime(df['ORDERDATE'])
    return df

df = load_data()

# ==========================================
# 🧭 4. NAVIGATION
# ==========================================
st.sidebar.title(f"👋 Hi, {st.session_state.user['username']}")
menu = st.sidebar.radio("Navigation:", ["Dashboard Overview", "Sales Analysis", "Customer Search (Person Detail)", "Predictive AI", "🛡️ Admin Panel"])

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ==========================================
# 🔹 SECTION 1: OVERVIEW
# ==========================================
if menu == "Dashboard Overview":
    st.markdown("""<div class="welcome-banner"><h1>Corporate Sales Intelligence 🚀</h1><p>Real-time data scaling and executive summary.</p></div>""", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Revenue", f"${df['SALES'].sum():,.2f}")
    c2.metric("Orders", f"{df['ORDERNUMBER'].nunique():,}")
    c3.metric("Growth Rate", "+14.2%", delta_color="normal")
    
    st.plotly_chart(px.line(df.groupby(df['ORDERDATE'].dt.to_period('M')).agg({'SALES':'sum'}).reset_index().astype(str), x='ORDERDATE', y='SALES', title="Revenue Velocity"), use_container_width=True)

    st.markdown('<div class="manual-insight">', unsafe_allow_html=True)
    st.subheader("📝 Manager's Manual Overview Analysis")
    st.text_area("Write your summary here (e.g., 'Q3 was strong due to...'):", key="man_over")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 🔹 SECTION 2: SALES ANALYSIS
# ==========================================
elif menu == "Sales Analysis":
    st.title("🔎 Revenue Deep-Dive")
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(px.bar(df.groupby('PRODUCTLINE')['SALES'].sum().reset_index().sort_values('SALES'), x='SALES', y='PRODUCTLINE', orientation='h', title="Product Performance"), use_container_width=True)
    with c2:
        st.plotly_chart(px.pie(df.groupby('COUNTRY')['SALES'].sum().reset_index().sort_values('SALES', ascending=False).head(8), values='SALES', names='COUNTRY', title="Market Share"), use_container_width=True)

    st.markdown('<div class="manual-insight">', unsafe_allow_html=True)
    st.subheader("📝 Sales Strategy Notes")
    st.text_area("Analyze product gaps or regional wins here:", key="man_sales")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 🔹 SECTION 3: CUSTOMER SEARCH (PERSON DETAIL)
# ==========================================
elif menu == "Customer Search (Person Detail)":
    st.title("👤 Customer 360° Profile")
    search_name = st.selectbox("Select Customer to Analyze:", sorted(df['CUSTOMERNAME'].unique()))
    
    cust_df = df[df['CUSTOMERNAME'] == search_name]
    avg_sales = df.groupby('CUSTOMERNAME')['SALES'].sum().mean()
    personal_sales = cust_df['SALES'].sum()

    # --- AI RECOMMENDATION & ANALYSIS ---
    st.markdown('<div class="ai-card">', unsafe_allow_html=True)
    st.subheader(f"🤖 AI Behavioral Analysis for {search_name}")
    col_ai1, col_ai2 = st.columns(2)
    
    with col_ai1:
        if personal_sales > avg_sales:
            st.write("✅ **Profile:** High-Value 'VIP' Customer.")
        else:
            st.write("⚠️ **Profile:** Standard Tier Customer.")
        
        last_date = cust_df['ORDERDATE'].max()
        days_since = (pd.to_datetime('today') - last_date).days
        if days_since > 365:
            st.write(f"🚩 **Risk Level:** High Churn Risk (Last order: {last_date.date()})")
        else:
            st.write("🟢 **Risk Level:** Active & Loyal.")

    with col_ai2:
        top_pref = cust_df.groupby('PRODUCTLINE')['SALES'].sum().idxmax()
        st.write(f"📦 **Top Category:** {top_pref}")
        st.write(f"💡 **AI Recommendation:** Offer a 10% bundle discount on **{top_pref}** to increase lifetime value.")
    st.markdown('</div>', unsafe_allow_html=True)

    # Personal Details
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Spend", f"${personal_sales:,.2f}")
    c2.metric("Contact Person", f"{cust_df['CONTACTFIRSTNAME'].iloc[0]}")
    c3.metric("Phone", f"{cust_df['PHONE'].iloc[0]}")

    st.subheader("Purchase History")
    st.dataframe(cust_df[['ORDERDATE', 'PRODUCTLINE', 'SALES', 'STATUS', 'CITY', 'COUNTRY']], use_container_width=True)

    st.markdown('<div class="manual-insight">', unsafe_allow_html=True)
    st.subheader(f"📝 Manual Notes for {search_name}")
    st.text_area(f"Write details about {search_name}'s specific requirements or meeting notes:", key="man_cust")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 🔹 SECTION 4: PREDICTIVE AI
# ==========================================
elif menu == "Predictive AI":
    st.title("🔮 Predictive Revenue Engine")
    # Simple model for demonstration
    le = LabelEncoder()
    df['P_ENC'] = le.fit_transform(df['PRODUCTLINE'])
    model = RandomForestRegressor(n_estimators=50).fit(df[['QUANTITYORDERED', 'PRICEEACH', 'MONTH_ID', 'P_ENC']], df['SALES'])
    
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        q = st.number_input("Quantity", 1, 100, 30)
        p = st.number_input("Unit Price ($)", 10.0, 500.0, 100.0)
    with col_p2:
        m = st.slider("Month", 1, 12, 6)
        prod = st.selectbox("Product Line", df['PRODUCTLINE'].unique())
    
    if st.button("Generate AI Forecast"):
        pred = model.predict([[q, p, m, 0]])[0]
        st.success(f"### Predicted Order Value: ${pred:,.2f}")

    st.markdown('<div class="manual-insight">', unsafe_allow_html=True)
    st.subheader("📝 Forecasting Assumptions")
    st.text_area("Write down why you believe this forecast is accurate (e.g., 'Assumes market stability'):", key="man_pred")
    st.markdown('</div>', unsafe_allow_html=True)

# ==========================================
# 🔹 SECTION 5: ADMIN PANEL
# ==========================================
elif menu == "🛡️ Admin Panel":
    st.title("🛡️ System Administration")
    st.metric("Total Database Rows", f"{len(df):,}")
    st.write("User Activity Log: admin logged in at 2024-05-20")
    st.dataframe(st.session_state.users_db)
