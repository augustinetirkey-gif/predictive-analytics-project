import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

st.set_page_config("Universal CSV Analyzer", layout="wide")

# ---------------- SIDEBAR ----------------
st.sidebar.title("📂 Upload CSV")
file = st.sidebar.file_uploader("Upload any CSV file", type="csv")

df = None
if file:
    df = pd.read_csv(file)

st.title("📊 Universal Data Analytics System")

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📘 Dataset Knowledge",
    "🧹 Data Cleaning",
    "📈 Visualization",
    "🤖 Prediction",
    "🔮 Insights & Forecast"
])

# ======================================================
# TAB 1 — COMPLETE DATASET KNOWLEDGE
# ======================================================
with tab1:
    if df is None:
        st.warning("Upload a CSV file")
    else:
        st.header("Dataset Overview")

        st.write("Rows:", df.shape[0])
        st.write("Columns:", df.shape[1])

        st.subheader("Sample Data")
        st.dataframe(df.head())

        st.subheader("Column Information")
        info = pd.DataFrame({
            "Column": df.columns,
            "Data Type": df.dtypes.astype(str),
            "Missing Values": df.isnull().sum(),
            "Unique Values": df.nunique()
        })
        st.dataframe(info)

        st.subheader("Numeric Columns")
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        st.write(num_cols)

        st.subheader("Categorical Columns")
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
        st.write(cat_cols)

        st.subheader("Duplicates")
        st.write("Duplicate Rows:", df.duplicated().sum())

        st.subheader("Memory Usage")
        st.write(f"{df.memory_usage().sum() / 1024:.2f} KB")

        st.subheader("Categorical Value Counts")
        for col in cat_cols:
            st.write(col)
            st.write(df[col].value_counts())

# ======================================================
# TAB 2 — DATA CLEANING
# ======================================================
with tab2:
    if df is None:
        st.warning("Upload a CSV file")
    else:
        st.header("Data Cleaning Options")

        if st.checkbox("Remove duplicate rows"):
            df = df.drop_duplicates()
            st.success("Duplicates removed")

        fill_method = st.selectbox(
            "Fill missing numeric values using",
            ["None", "Mean", "Median"]
        )

        if fill_method != "None":
            for col in num_cols:
                if fill_method == "Mean":
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].median(), inplace=True)
            st.success("Missing values filled")

        st.subheader("Cleaned Dataset")
        st.dataframe(df.head())

# ======================================================
# TAB 3 — VISUALIZATION
# ======================================================
with tab3:
    if df is None:
        st.warning("Upload a CSV file")
    else:
        st.header("Automatic Visualization")

        if num_cols:
            col = st.selectbox("Select numeric column", num_cols)

            fig = plt.figure()
            plt.hist(df[col], bins=20)
            plt.title("Histogram")
            st.pyplot(fig)

            fig = plt.figure()
            plt.boxplot(df[col])
            plt.title("Boxplot")
            st.pyplot(fig)

            fig = plt.figure()
            plt.plot(df[col])
            plt.title("Line Trend")
            st.pyplot(fig)

        if cat_cols:
            col2 = st.selectbox("Select categorical column", cat_cols)
            fig = plt.figure()
            df[col2].value_counts().plot(kind="bar")
            plt.title("Category Count")
            st.pyplot(fig)

        if len(num_cols) > 1:
            st.subheader("Correlation Matrix")
            st.dataframe(df[num_cols].corr())

# ======================================================
# TAB 4 — FEATURE ENGINEERING + PREDICTION
# ======================================================
with tab4:
    if df is None:
        st.warning("Upload a CSV file")
    else:
        st.header("Prediction Model")

        if len(num_cols) < 2:
            st.info("Need at least 2 numeric columns")
        else:
            target = st.selectbox("Target Column", num_cols)
            features = st.multiselect(
                "Feature Columns",
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

                score = r2_score(y_test, model.predict(X_test))
                st.success(f"Model R² Score: {score:.2f}")

                st.subheader("Manual Prediction")
                inputs = []
                for f in features:
                    inputs.append(st.number_input(f))

                if st.button("Predict Value"):
                    result = model.predict([inputs])
                    st.success(f"Predicted Value: {result[0]:.2f}")

# ======================================================
# TAB 5 — INSIGHTS & FORECASTING
# ======================================================
with tab5:
    if df is None:
        st.warning("Upload a CSV file")
    else:
        st.header("Insights & Future Forecast")

        if num_cols:
            col = st.selectbox("Select column for trend analysis", num_cols)

            st.write("Mean:", df[col].mean())
            st.write("Max:", df[col].max())
            st.write("Min:", df[col].min())

            X = np.arange(len(df[col])).reshape(-1, 1)
            y = df[col].values

            model = LinearRegression()
            model.fit(X, y)

            future = st.slider("Future days", 1, 30, 7)
            future_X = np.arange(len(y), len(y) + future).reshape(-1, 1)
            future_y = model.predict(future_X)

            fig = plt.figure()
            plt.plot(y, label="Past")
            plt.plot(range(len(y), len(y) + future), future_y, label="Future")
            plt.legend()
            st.pyplot(fig)

            st.info("Forecast based on historical trend (Linear Regression)")
