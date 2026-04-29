import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

st.set_page_config(
    page_title="SmartSalary Predictor",
    page_icon="💰",
    layout="centered"
)

st.markdown("""
<style>
[data-testid="stSidebar"] {
    display: none;
}

.block-container {
    max-width: 950px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

.app-title {
    text-align: center;
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.2rem;
}

.app-subtitle {
    text-align: center;
    opacity: 0.75;
    margin-bottom: 1.5rem;
}

[data-testid="stMetric"] {
    background-color: #151922;
    padding: 12px;
    border-radius: 12px;
    border: 1px solid rgba(255,255,255,0.06);
}

.stTabs [data-baseweb="tab-list"] {
    justify-content: center;
    gap: 18px;
}

.stTabs [data-baseweb="tab"] {
    font-weight: 700;
    padding: 10px 18px;
}
</style>
""", unsafe_allow_html=True)

BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_model_files():
    model = joblib.load(BASE_DIR / "model.pkl")
    scaler = joblib.load(BASE_DIR / "scaler.pkl")
    encoders = joblib.load(BASE_DIR / "encoders.pkl")
    return model, scaler, encoders


@st.cache_data
def load_data():
    columns = [
        "age", "workclass", "fnlwgt", "education", "education_num",
        "marital_status", "occupation", "relationship", "race",
        "sex", "capital_gain", "capital_loss", "hours_per_week",
        "native_country", "income"
    ]

    return pd.read_csv(
        BASE_DIR / "adult.csv",
        names=columns,
        skipinitialspace=True
    )


model, scaler, encoders = load_model_files()
df = load_data()

def clean_data(data):
    data = data.copy()

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip()

    data = data.replace("?", pd.NA)
    data = data.dropna()

    if "income" in data.columns:
        data["income"] = data["income"].str.replace(".", "", regex=False)

    return data


df_clean = clean_data(df)

def encode_data(data):
    data = data.copy()

    for col in data.select_dtypes(include="object").columns:
        if col in encoders:
            data[col] = encoders[col].transform(data[col])

    return data


df_encoded = encode_data(df_clean)

st.markdown('<div class="app-title">💰 SmartSalary Predictor</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="app-subtitle">Interactive salary prediction app using Adult Census data</div>',
    unsafe_allow_html=True
)

tab_dataset, tab_eda, tab_eval, tab_pred = st.tabs([
    "📂 Dataset",
    "📊 EDA",
    "🤖 Evaluation",
    "🎯 Prediction"
])

with tab_dataset:
    st.header("Dataset Overview")

    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Clean Rows", df_clean.shape[0])

    inner1, inner2, inner3 = st.tabs(["Preview", "Missing Values", "Data Types"])

    with inner1:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(10), use_container_width=True, height=300)

        st.subheader("Cleaned Data Preview")
        st.dataframe(df_clean.head(10), use_container_width=True, height=300)

    with inner2:
        missing = df.replace("?", pd.NA).isnull().sum()
        st.dataframe(missing.to_frame("Missing Count"), use_container_width=True)

    with inner3:
        st.dataframe(df.dtypes.to_frame("Data Type"), use_container_width=True)

with tab_eda:
    st.header("Compact EDA")

    numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()

    inner1, inner2, inner3 = st.tabs(["Overview", "Plots", "Correlation"])

    with inner1:
        c1, c2, c3 = st.columns(3)
        c1.metric("Rows", df_clean.shape[0])
        c2.metric("Columns", df_clean.shape[1])
        c3.metric("Missing", int(df_clean.isnull().sum().sum()))

        st.subheader("Quick Preview")
        st.dataframe(df_clean.head(8), use_container_width=True, height=260)

        with st.expander("Descriptive Statistics"):
            st.dataframe(df_clean.describe(), use_container_width=True)

    with inner2:
        plot_type = st.radio(
            "Plot Type",
            ["Numeric Distribution", "Categorical Count", "Boxplot"],
            horizontal=True
        )

        if plot_type == "Numeric Distribution":
            col = st.selectbox("Numeric column", numeric_cols)

            fig, ax = plt.subplots(figsize=(6, 3))
            sns.histplot(df_clean[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig, use_container_width=False)

        elif plot_type == "Categorical Count":
            col = st.selectbox("Categorical column", categorical_cols)

            fig, ax = plt.subplots(figsize=(7, 3.5))
            order = df_clean[col].value_counts().head(10).index
            sns.countplot(data=df_clean, y=col, order=order, ax=ax)
            ax.set_title(f"Top 10 {col}")
            st.pyplot(fig, use_container_width=False)

        elif plot_type == "Boxplot":
            col = st.selectbox("Numeric column", numeric_cols)

            fig, ax = plt.subplots(figsize=(6, 2.8))
            sns.boxplot(data=df_clean, x=col, ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig, use_container_width=False)

    with inner3:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        corr = df_clean[numeric_cols].corr()
        sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", ax=ax)
        ax.set_title("Correlation Heatmap")
        st.pyplot(fig, use_container_width=False)

with tab_eval:
    st.header("Model Evaluation")

    X = df_encoded.drop("income", axis=1)
    y = df_encoded["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)

    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    c1, c2, c3 = st.columns(3)
    c1.metric("Accuracy", f"{acc:.2%}")
    c2.metric("Test Samples", len(y_test))
    c3.metric("Features", X.shape[1])

    inner1, inner2 = st.tabs(["Confusion Matrix", "Report"])

    with inner1:
        fig, ax = plt.subplots(figsize=(4.8, 3.5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig, use_container_width=False)

    with inner2:
        st.text(classification_report(y_test, y_pred))

with tab_pred:
    st.header("Salary Prediction")
    st.caption("Fill in the profile below and click Predict Salary.")

    col1, col2 = st.columns(2)

    with col1:
        age = st.slider("Age", 18, 90, 30)
        workclass = st.selectbox("Workclass", encoders["workclass"].classes_)
        education = st.selectbox("Education", encoders["education"].classes_)
        education_num = st.slider("Education Number", 1, 16, 10)
        marital_status = st.selectbox("Marital Status", encoders["marital_status"].classes_)
        occupation = st.selectbox("Occupation", encoders["occupation"].classes_)
        relationship = st.selectbox("Relationship", encoders["relationship"].classes_)

    with col2:
        race = st.selectbox("Race", encoders["race"].classes_)
        sex = st.selectbox("Sex", encoders["sex"].classes_)
        capital_gain = st.number_input("Capital Gain", min_value=0, value=0)
        capital_loss = st.number_input("Capital Loss", min_value=0, value=0)
        hours_per_week = st.slider("Hours per Week", 1, 100, 40)
        native_country = st.selectbox("Native Country", encoders["native_country"].classes_)
        fnlwgt = st.number_input("Fnlwgt", min_value=1, value=100000)

    st.divider()

    if st.button("Predict Salary", use_container_width=True):
        input_data = {
            "age": age,
            "workclass": encoders["workclass"].transform([workclass])[0],
            "fnlwgt": fnlwgt,
            "education": encoders["education"].transform([education])[0],
            "education_num": education_num,
            "marital_status": encoders["marital_status"].transform([marital_status])[0],
            "occupation": encoders["occupation"].transform([occupation])[0],
            "relationship": encoders["relationship"].transform([relationship])[0],
            "race": encoders["race"].transform([race])[0],
            "sex": encoders["sex"].transform([sex])[0],
            "capital_gain": capital_gain,
            "capital_loss": capital_loss,
            "hours_per_week": hours_per_week,
            "native_country": encoders["native_country"].transform([native_country])[0]
        }

        input_df = pd.DataFrame([input_data])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)[0]

        st.subheader("Prediction Result")

        if prediction == 1:
            st.success("💸 Predicted Income: >50K")
        else:
            st.warning("💼 Predicted Income: <=50K")

        with st.expander("View Encoded Input Data"):
            st.dataframe(input_df, use_container_width=True)