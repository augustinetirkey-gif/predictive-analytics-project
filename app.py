import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ---------------- PAGE CONFIG ----------------
st.set_page_config(page_title="Data Analysis & Forecast App", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("Upload Dataset")

option = st.sidebar.radio(
    "Choose data input method:",
    ("Browse File", "Paste CSV Link")
)

df = None

if option == "Browse File":
    file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if file:
        df = pd.read_csv(file)
else:
    link = st.sidebar.text_input("Enter CSV URL")
    if link:
        try:
            df = pd.read_csv(link)
        except:
            st.sidebar.error("Invalid CSV link")

st.sidebar.markdown("---")
st.sidebar.info("Simple & Student Friendly App 😊")

# ---------------- TITLE ----------------
st.title("📊 Data Analysis and Prediction Tool")

# ---------------- TABS ----------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Overview",
    "Data Cleaning",
    "Visualization",
    "Model Building",
    "Future Forecasting"
])

# ---------------- TAB 1 ----------------
with tab1:
    st.subheader("Dataset Overview")
    if df is None:
        st.warning("Please upload a dataset")
    else:
        st.write("Rows & Columns:", df.shape)
        st.dataframe(df.head())
        st.write("Column Types")
        st.write(df.dtypes)
        st.write("Missing Values")
        st.write(df.isnull().sum())

# ---------------- TAB 2 ----------------
with tab2:
    st.subheader("Simple Data Cleaning")
    if df is None:
        st.warning("Upload dataset first")
    else:
        if st.checkbox("Remove duplicates"):
            df = df.drop_duplicates()
            st.success("Duplicates removed")

        if st.checkbox("Fill missing numeric values with mean"):
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            st.success("Missing values filled")

        st.write("Cleaned Data")
        st.dataframe(df.head())

# ---------------- TAB 3 ----------------
with tab3:
    st.subheader("Data Visualization")
    if df is None:
        st.warning("Upload dataset first")
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if num_cols:
            col = st.selectbox("Select numeric column", num_cols)

            fig = plt.figure()
            plt.hist(df[col], bins=20)
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.title(f"Distribution of {col}")
            st.pyplot(fig)
        else:
            st.info("No numeric columns found")

# ---------------- TAB 4 ----------------
with tab4:
    st.subheader("Prediction Model (Linear Regression)")
    if df is None:
        st.warning("Upload dataset first")
    else:
        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns")
        else:
            target = st.selectbox("Target column", num_cols)
            features = st.multiselect(
                "Feature columns",
                [c for c in num_cols if c != target]
            )

            if features:
                X = df[features]
                y = df[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                st.success(f"Model Accuracy (R² Score): {r2_score(y_test, y_pred):.2f}")

                st.markdown("### 🔢 Try Your Own Values")
                input_data = []
                for col in features:
                    input_data.append(st.number_input(f"{col}"))

                if st.button("Predict Value"):
                    result = model.predict([input_data])
                    st.success(f"Predicted Result: {result[0]:.2f}")

# ---------------- TAB 5 ----------------
with tab5:
    st.subheader("📈 Future Forecasting")

    if df is None:
        st.warning("Upload dataset first")
    else:
        st.markdown("""
        **What is happening here?**
        - We use past data  
        - Create a trend using Linear Regression  
        - Predict future values  
        """)

        num_cols = df.select_dtypes(include=np.number).columns.tolist()

        if num_cols:
            target = st.selectbox("Select column to forecast", num_cols)
            steps = st.slider("Future time steps", 1, 30, 5)

            y = df[target].values
            X = np.arange(len(y)).reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)

            future_X = np.arange(len(y), len(y) + steps).reshape(-1, 1)
            future_y = model.predict(future_X)

            forecast_df = pd.DataFrame({
                "Future Step": range(1, steps + 1),
                "Predicted Value": future_y
            })

            st.success("Forecast Generated Successfully")
            st.dataframe(forecast_df)

            # -------- LINE CHART --------
            fig = plt.figure()
            plt.plot(y, label="Past Data")
            plt.plot(range(len(y), len(y) + steps), future_y, label="Future Prediction")
            plt.xlabel("Time Index")
            plt.ylabel(target)
            plt.title("Past vs Future Forecast")
            plt.legend()
            st.pyplot(fig)

        else:
            st.info("No numeric data available")
