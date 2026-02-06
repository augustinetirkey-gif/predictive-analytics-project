import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Data Analysis and Prediction Tool", layout="wide")

# ------------------ SIDEBAR ------------------
st.sidebar.title("Import Data")
option = st.sidebar.radio(
    "Select any one method to fetch data:",
    ("Browse Files", "Link/Name")
)

df = None

if option == "Browse Files":
    file = st.sidebar.file_uploader("Upload CSV file", type=["csv"])
    if file is not None:
        df = pd.read_csv(file)
else:
    link = st.sidebar.text_input("Enter dataset URL or filename")
    if link:
        try:
            df = pd.read_csv(https://github.com/augustinetirkey-gif/predictive-analytics-project/blob/main/cleaned_sales_data.csv)
        except:
            st.sidebar.error("Unable to load dataset")

st.sidebar.markdown("---")
st.sidebar.button("Follow Me 👍😎")

# ------------------ MAIN TITLE ------------------
st.title("Data Analysis and Prediction Tool")

# ------------------ TABS ------------------
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Data Overview",
    "Data Cleaning",
    "Data Visualization",
    "Model Building",
    "Future Forecasting"
])

# ------------------ DATA OVERVIEW ------------------
with tab1:
    st.subheader("Data Profile")
    if df is None:
        st.error("Please upload dataset!")
    else:
        st.write("Dataset Shape:", df.shape)
        st.dataframe(df.head())
        st.write("Column Types")
        st.write(df.dtypes)
        st.write("Missing Values")
        st.write(df.isnull().sum())

# ------------------ DATA CLEANING ------------------
with tab2:
    st.subheader("Data Cleaning")
    if df is None:
        st.warning("Upload dataset first")
    else:
        if st.checkbox("Remove duplicate rows"):
            df = df.drop_duplicates()
            st.success("Duplicate rows removed")

        if st.checkbox("Fill missing numeric values with mean"):
            num_cols = df.select_dtypes(include=np.number).columns
            df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            st.success("Missing values filled")

        st.write("Cleaned Data Preview")
        st.dataframe(df.head())

# ------------------ DATA VISUALIZATION ------------------
with tab3:
    st.subheader("Data Visualization")
    if df is None:
        st.warning("Upload dataset first")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) > 0:
            col = st.selectbox("Select numeric column", numeric_cols)

            fig = plt.figure()
            plt.hist(df[col], bins=20)
            plt.xlabel(col)
            plt.ylabel("Frequency")
            plt.title(f"Histogram of {col}")
            st.pyplot(fig)
        else:
            st.info("No numeric columns found")

# ------------------ MODEL BUILDING ------------------
with tab4:
    st.subheader("Prediction Model")
    if df is None:
        st.warning("Upload dataset first")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 2:
            st.info("Need at least 2 numeric columns")
        else:
            target = st.selectbox("Select Target Column", numeric_cols)
            features = st.multiselect(
                "Select Feature Columns",
                [c for c in numeric_cols if c != target]
            )

            if len(features) > 0:
                X = df[features]
                y = df[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                model = LinearRegression()
                model.fit(X_train, y_train)

                y_pred = model.predict(X_test)
                st.write("R² Score:", r2_score(y_test, y_pred))

                st.subheader("Make Single Prediction")
                input_data = []

                for col in features:
                    val = st.number_input(f"Enter {col}")
                    input_data.append(val)

                if st.button("Predict"):
                    result = model.predict([input_data])
                    st.success(f"Predicted Value: {result[0]}")

# ------------------ FUTURE FORECASTING ------------------
with tab5:
    st.subheader("Future Forecasting")

    if df is None:
        st.warning("Upload dataset first")
    else:
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()

        if len(numeric_cols) < 1:
            st.info("No numeric column available")
        else:
            target = st.selectbox(
                "Select Target Column to Forecast",
                numeric_cols
            )

            steps = st.number_input(
                "How many future values to predict?",
                min_value=1,
                max_value=50,
                value=5
            )

            y = df[target].values
            X = np.arange(len(y)).reshape(-1, 1)

            model = LinearRegression()
            model.fit(X, y)

            future_X = np.arange(len(y), len(y) + steps).reshape(-1, 1)
            future_predictions = model.predict(future_X)

            forecast_df = pd.DataFrame({
                "Future Step": range(1, steps + 1),
                "Predicted Value": future_predictions
            })

            st.success("Future Forecast Generated ✅")
            st.dataframe(forecast_df)
