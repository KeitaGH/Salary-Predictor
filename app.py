import streamlit as st
import pandas as pd
import joblib
import seaborn as sns
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from sklearn.preprocessing import (
    StandardScaler,
    MinMaxScaler,
    RobustScaler,
    QuantileTransformer,
    Normalizer
)

from sklearn.ensemble import (
    RandomForestClassifier,
    ExtraTreesClassifier,
    GradientBoostingClassifier
)

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier


# ======================================================
# PAGE CONFIG
# ======================================================
st.set_page_config(
    page_title="SmartSalary Predictor",
    page_icon="💰",
    layout="wide"
)


# ======================================================
# CUSTOM CSS
# ======================================================
st.markdown("""
<style>
/* ================= GLOBAL ================= */
.stApp {
    background: linear-gradient(135deg, #fffaf0 0%, #ffffff 45%, #fff7d6 100%);
    color: #1f2937;
}

.block-container {
    max-width: 1180px;
    padding-top: 2.6rem;
    padding-bottom: 3rem;
}

/* reduce excessive boldness globally */
h1, h2, h3 {
    color: #1f2937 !important;
    font-weight: 600 !important;
}

p, li, div {
    font-weight: 400;
}

/* ================= SIDEBAR ================= */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #111827 0%, #1f2937 100%);
}

/* do NOT force every sidebar text to white globally */
[data-testid="stSidebar"] > div:first-child {
    padding-top: 1.7rem;
    padding-bottom: 1.5rem;
}

/* sidebar text only */
[data-testid="stSidebar"] .stRadio label,
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stCaption,
[data-testid="stSidebar"] label {
    color: #f9fafb !important;
}

.sidebar-card {
    background: linear-gradient(135deg, #facc15 0%, #f59e0b 100%);
    padding: 18px 16px;
    border-radius: 22px;
    margin-bottom: 20px;
    box-shadow: 0 12px 26px rgba(245, 158, 11, 0.22);
}

.sidebar-card-title {
    font-size: 1.25rem;
    font-weight: 650;
    color: #111827 !important;
    margin-bottom: 2px;
}

.sidebar-card-subtitle {
    font-size: 0.84rem;
    color: #374151 !important;
    font-weight: 500;
}

.sidebar-section {
    margin-top: 14px;
    margin-bottom: 8px;
    font-size: 0.78rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    color: #facc15 !important;
    text-transform: uppercase;
}

/* ================= SIDEBAR SELECTBOX FIX ================= */
/* selected box background */
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #fffaf0 !important;
    border: 1px solid rgba(250, 204, 21, 0.65) !important;
    border-radius: 12px !important;
}

/* selected value text */
[data-testid="stSidebar"] div[data-baseweb="select"] span {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* select input text */
[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* dropdown arrow */
[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    fill: #111827 !important;
    color: #111827 !important;
}

/* dropdown menu options */
div[data-baseweb="popover"] div[role="listbox"] {
    background-color: #ffffff !important;
}

div[data-baseweb="popover"] div[role="option"] {
    color: #111827 !important;
    background-color: #ffffff !important;
}

div[data-baseweb="popover"] div[role="option"]:hover {
    background-color: #fff7d6 !important;
    color: #111827 !important;
}

/* ================= HOME HERO ================= */
.home-hero {
    background: linear-gradient(135deg, #0f172a 0%, #1e293b 55%, #3b2f12 100%);
    padding: 30px 34px;
    border-radius: 28px;
    margin-top: 0.6rem;
    margin-bottom: 28px;
    box-shadow: 0 16px 34px rgba(17, 24, 39, 0.16);
    border: 1px solid rgba(250, 204, 21, 0.22);
}

.home-hero-title {
    font-size: 2.0rem;
    font-weight: 650;
    color: #ffffff !important;
    margin-bottom: 14px;
    line-height: 1.25;
}

.home-hero-subtitle {
    color: #fde68a !important;
    font-size: 0.98rem;
    line-height: 1.85;
    font-weight: 400;
    max-width: 980px;
}

/* ================= SECTION CARDS ================= */
.content-card {
    background: rgba(255, 255, 255, 0.92);
    border-radius: 24px;
    padding: 22px 24px;
    border: 1px solid rgba(245, 158, 11, 0.20);
    box-shadow: 0 12px 28px rgba(31, 41, 55, 0.07);
    margin-bottom: 18px;
}

.info-note {
    background: #fffbeb;
    border-left: 5px solid #f59e0b;
    padding: 14px 16px;
    border-radius: 14px;
    color: #374151;
    font-size: 0.94rem;
    margin-bottom: 16px;
}

.section-title {
    font-size: 1.55rem;
    font-weight: 600;
    color: #1f2937;
    margin-bottom: 14px;
}

/* ================= METRIC ================= */
[data-testid="stMetric"] {
    background: #ffffff;
    border: 1px solid rgba(245, 158, 11, 0.25);
    padding: 15px;
    border-radius: 20px;
    box-shadow: 0 8px 18px rgba(31, 41, 55, 0.05);
}

[data-testid="stMetricLabel"] {
    color: #6b7280;
    font-weight: 400 !important;
    font-size: 0.9rem !important;
}

[data-testid="stMetricValue"] {
    color: #111827;
    font-weight: 550 !important;
    font-size: 2rem !important;
}

/* make large text in metric less aggressive */
[data-testid="stMetricValue"] div {
    font-weight: 550 !important;
}

/* ================= BUTTONS ================= */
.stButton > button {
    border-radius: 14px;
    border: 1px solid #f59e0b;
    background: linear-gradient(135deg, #facc15 0%, #f59e0b 100%);
    color: #111827;
    font-weight: 550;
    padding: 0.65rem 1rem;
    transition: 0.18s ease;
}

.stButton > button:hover {
    background: linear-gradient(135deg, #fde047 0%, #facc15 100%);
    border-color: #d97706;
    color: #111827;
    transform: translateY(-1px);
}

/* ================= TABS ================= */
.stTabs [data-baseweb="tab-list"] {
    gap: 10px;
}

.stTabs [data-baseweb="tab"] {
    background: #fff7d6;
    border-radius: 14px 14px 0 0;
    padding: 10px 18px;
    font-weight: 550;
    color: #374151;
}

.stTabs [aria-selected="true"] {
    background: #facc15 !important;
    color: #111827 !important;
}

/* ================= DATAFRAME ================= */
[data-testid="stDataFrame"] {
    border-radius: 18px;
    overflow: hidden;
}

/* ================= FLOW CARD ================= */
.flow-card {
    background: #ffffff;
    border: 1px solid rgba(245, 158, 11, 0.22);
    border-radius: 18px;
    padding: 16px;
    height: 100%;
    box-shadow: 0 8px 18px rgba(31, 41, 55, 0.05);
}

.flow-step {
    width: 34px;
    height: 34px;
    border-radius: 50%;
    background: linear-gradient(135deg, #facc15 0%, #f59e0b 100%);
    color: #111827;
    display: flex;
    align-items: center;
    justify-content: center;
    font-weight: 600;
    margin-bottom: 10px;
}

.flow-title {
    font-size: 1.02rem;
    font-weight: 550;
    color: #1f2937;
    margin-bottom: 8px;
}

/* ================= PREDICTION RESULT ================= */
.prediction-card-success {
    background: linear-gradient(135deg, #fff7d6 0%, #fffbeb 100%);
    border: 1px solid rgba(245, 158, 11, 0.35);
    border-left: 8px solid #f59e0b;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 12px 24px rgba(245, 158, 11, 0.10);
    margin-top: 8px;
    margin-bottom: 16px;
}

.prediction-card-warning {
    background: linear-gradient(135deg, #fffaf0 0%, #ffffff 100%);
    border: 1px solid rgba(245, 158, 11, 0.24);
    border-left: 8px solid #facc15;
    border-radius: 22px;
    padding: 22px;
    box-shadow: 0 12px 24px rgba(245, 158, 11, 0.08);
    margin-top: 8px;
    margin-bottom: 16px;
}

.prediction-title {
    font-size: 1.25rem;
    font-weight: 550;
    color: #111827;
    margin-bottom: 8px;
}

.prediction-main {
    font-size: 1.65rem;
    font-weight: 600;
    color: #111827;
    margin-bottom: 4px;
}

.prediction-sub {
    color: #4b5563;
    font-size: 0.95rem;
    line-height: 1.65;
}

/* ================= FORM ================= */
[data-testid="stForm"] {
    background: rgba(255,255,255,0.92);
    border-radius: 24px;
    border: 1px solid rgba(245, 158, 11, 0.20);
    padding: 20px;
    box-shadow: 0 12px 28px rgba(31, 41, 55, 0.07);
}
            
/* ======================================================
   FINAL SIDEBAR FIX: NAVIGATION + SELECTBOX THEME SAFE
   ====================================================== */

/* Force sidebar navigation text to stay readable */
[data-testid="stSidebar"] [role="radiogroup"] label,
[data-testid="stSidebar"] [role="radiogroup"] label span,
[data-testid="stSidebar"] [role="radiogroup"] label div,
[data-testid="stSidebar"] [role="radiogroup"] p {
    color: #f9fafb !important;
    opacity: 1 !important;
    -webkit-text-fill-color: #f9fafb !important;
}

/* Make selected radio label slightly brighter */
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) span,
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) div,
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) p {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    font-weight: 600 !important;
}

/* Radio circle color */
[data-testid="stSidebar"] [role="radiogroup"] input[type="radio"] {
    accent-color: #facc15 !important;
}

/* Sidebar selectbox: stable in both light and dark mode */
[data-testid="stSidebar"] div[data-baseweb="select"] > div {
    background-color: #fffaf0 !important;
    border: 1px solid rgba(250, 204, 21, 0.75) !important;
    border-radius: 12px !important;
}

/* Selected value inside selectbox */
[data-testid="stSidebar"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] div[data-baseweb="select"] div,
[data-testid="stSidebar"] div[data-baseweb="select"] input {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
}

/* Selectbox arrow */
[data-testid="stSidebar"] div[data-baseweb="select"] svg {
    color: #111827 !important;
    fill: #111827 !important;
}

/* Dropdown menu when opened */
div[data-baseweb="popover"] {
    z-index: 9999 !important;
}

div[data-baseweb="popover"] div[role="listbox"] {
    background-color: #fffaf0 !important;
    border: 1px solid rgba(250, 204, 21, 0.75) !important;
}

div[data-baseweb="popover"] div[role="option"],
div[data-baseweb="popover"] div[role="option"] span,
div[data-baseweb="popover"] div[role="option"] div {
    color: #111827 !important;
    -webkit-text-fill-color: #111827 !important;
    background-color: #fffaf0 !important;
}

div[data-baseweb="popover"] div[role="option"]:hover,
div[data-baseweb="popover"] div[role="option"][aria-selected="true"] {
    background-color: #fde68a !important;
    color: #111827 !important;
}
            
</style>
""", unsafe_allow_html=True)

# ======================================================
# PATH
# ======================================================
BASE_DIR = Path(__file__).parent


# ======================================================
# LOAD DATA AND ENCODERS
# ======================================================
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


@st.cache_resource
def load_encoders():
    return joblib.load(BASE_DIR / "encoders.pkl")


df = load_data()
encoders = load_encoders()


# ======================================================
# DATA PROCESSING
# ======================================================
def clean_data(data):
    data = data.copy()

    for col in data.select_dtypes(include="object").columns:
        data[col] = data[col].astype(str).str.strip()

    data = data.replace("?", pd.NA)
    data = data.dropna()

    if "income" in data.columns:
        data["income"] = data["income"].str.replace(".", "", regex=False)

    return data


def encode_data(data):
    data = data.copy()

    for col in data.select_dtypes(include="object").columns:
        if col in encoders:
            data[col] = encoders[col].transform(data[col])

    return data


df_clean = clean_data(df)
df_encoded = encode_data(df_clean)


# ======================================================
# SCALER + MODEL
# ======================================================
def get_scaler(name):
    if name == "StandardScaler":
        return StandardScaler()
    if name == "MinMaxScaler":
        return MinMaxScaler()
    if name == "RobustScaler":
        return RobustScaler()
    if name == "QuantileTransformer":
        return QuantileTransformer(output_distribution="normal", random_state=42)
    if name == "Normalizer":
        return Normalizer()
    return None


def get_model(name):
    if name == "Random Forest":
        return RandomForestClassifier(
            n_estimators=180,
            random_state=42,
            n_jobs=-1
        )

    if name == "Extra Trees":
        return ExtraTreesClassifier(
            n_estimators=180,
            random_state=42,
            n_jobs=-1
        )

    if name == "Gradient Boosting":
        return GradientBoostingClassifier(random_state=42)

    if name == "Logistic Regression":
        return LogisticRegression(max_iter=1200, random_state=42)

    if name == "Decision Tree":
        return DecisionTreeClassifier(max_depth=12, random_state=42)

    if name == "KNN":
        return KNeighborsClassifier(n_neighbors=7)

    return RandomForestClassifier(random_state=42)


@st.cache_resource
def train_selected_model(scaler_name, model_name):
    X = df_encoded.drop("income", axis=1)
    y = df_encoded["income"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    scaler = get_scaler(scaler_name)

    if scaler is not None:
        X_train_processed = scaler.fit_transform(X_train)
        X_test_processed = scaler.transform(X_test)
    else:
        X_train_processed = X_train
        X_test_processed = X_test

    model = get_model(model_name)
    model.fit(X_train_processed, y_train)
    y_pred = model.predict(X_test_processed)

    return model, scaler, X_train, X_test, y_train, y_test, y_pred


def get_income_label(encoded_value):
    if "income" in encoders:
        return encoders["income"].inverse_transform([encoded_value])[0]
    return str(encoded_value)


def get_prediction_confidence(model, processed_input):
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(processed_input)[0]
        return float(max(probs))
    return None


# ======================================================
# SIDEBAR
# ======================================================
with st.sidebar:
    st.markdown("""
    <div class="sidebar-card">
        <div class="sidebar-card-title">💰 SmartSalary</div>
        <div class="sidebar-card-subtitle">Salary prediction dashboard</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section">Navigation</div>', unsafe_allow_html=True)

    menu = st.radio(
        "Navigation",
        [
            "🏠 Home",
            "📂 Dataset",
            "📊 EDA",
            "🧹 Preprocessing",
            "⚙️ Model Setup",
            "🤖 Evaluation",
            "🎯 Prediction"
        ],
        label_visibility="collapsed"
    )

    st.markdown('<div class="sidebar-section">Experiment Settings</div>', unsafe_allow_html=True)

    scaler_name = st.selectbox(
        "Scaler",
        [
            "No Scaling",
            "StandardScaler",
            "MinMaxScaler",
            "RobustScaler",
            "QuantileTransformer",
            "Normalizer"
        ]
    )

    model_name = st.selectbox(
        "AI Model",
        [
            "Random Forest",
            "Extra Trees",
            "Gradient Boosting",
            "Logistic Regression",
            "Decision Tree",
            "KNN"
        ]
    )

    st.caption("Pages are flexible. You can open any page anytime.")


model, scaler, X_train, X_test, y_train, y_test, y_pred = train_selected_model(
    scaler_name,
    model_name
)


# ======================================================
# HOME
# ======================================================
if menu == "🏠 Home":
    st.markdown("""
    <div class="home-hero">
        <div class="home-hero-title">💰 SmartSalary Predictor</div>
        <div class="home-hero-subtitle">
            A machine learning dashboard for predicting whether a person's income is above or below 50K.
            This application allows users to explore the dataset, understand preprocessing steps,
            configure the experiment, evaluate model performance, and generate custom predictions.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Project Background</div>', unsafe_allow_html=True)
    st.write("""
    Income classification is a common machine learning problem that helps demonstrate how raw demographic
    and socio-economic data can be transformed into useful predictions. In this project, we use the Adult
    Census dataset to classify whether a person's income is less than or equal to 50K, or greater than 50K.
    """)
    st.write("""
    The dashboard is designed not only to provide prediction results, but also to help users understand the
    full machine learning workflow, from data exploration to preprocessing, model selection, evaluation,
    and final prediction.
    """)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Project Objectives</div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        - Provide an interactive interface for exploring the dataset  
        - Show how preprocessing affects model readiness  
        - Allow users to compare several scalers  
        """)
    with col2:
        st.markdown("""
        - Allow users to test several machine learning models  
        - Evaluate performance clearly and interactively  
        - Predict salary category from custom user input  
        """)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">System Workflow</div>', unsafe_allow_html=True)

    c1, c2, c3, c4, c5 = st.columns(5)

    with c1:
        st.markdown("""
        <div class="flow-card">
            <div class="flow-step">1</div>
            <div class="flow-title">Load Data</div>
            <div>Read the Adult Census dataset and inspect the raw records.</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="flow-card">
            <div class="flow-step">2</div>
            <div class="flow-title">Clean & Encode</div>
            <div>Handle missing values, clean labels, and transform categorical data.</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        st.markdown("""
        <div class="flow-card">
            <div class="flow-step">3</div>
            <div class="flow-title">Apply Scaler</div>
            <div>Use the selected scaler to transform the feature space if needed.</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        st.markdown("""
        <div class="flow-card">
            <div class="flow-step">4</div>
            <div class="flow-title">Train Model</div>
            <div>Fit the selected machine learning model on the processed data.</div>
        </div>
        """, unsafe_allow_html=True)

    with c5:
        st.markdown("""
        <div class="flow-card">
            <div class="flow-step">5</div>
            <div class="flow-title">Evaluate & Predict</div>
            <div>Evaluate performance and predict salary class from user input.</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="content-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-title">Current Experiment Setup</div>', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Selected Scaler", scaler_name)
    col2.metric("Selected Model", model_name)
    col3.metric("Usable Rows", df_clean.shape[0])

    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# DATASET
# ======================================================
elif menu == "📂 Dataset":
    st.header("Dataset Overview")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    col1.metric("Raw Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Clean Rows", df_clean.shape[0])

    st.markdown("""
    <div class="info-note">
        Click a button to reveal the dataset information. Nothing is shown automatically.
    </div>
    """, unsafe_allow_html=True)

    btn1, btn2, btn3 = st.columns(3)

    with btn1:
        show_raw = st.button("Show Raw Data", use_container_width=True)

    with btn2:
        show_clean = st.button("Show Cleaned Data", use_container_width=True)

    with btn3:
        show_missing = st.button("Show Missing Values", use_container_width=True)

    if show_raw:
        st.subheader("Raw Data Preview")
        st.dataframe(df.head(20), use_container_width=True, height=360)

    if show_clean:
        st.subheader("Cleaned Data Preview")
        st.dataframe(df_clean.head(20), use_container_width=True, height=360)

    if show_missing:
        st.subheader("Missing Values")
        missing = df.replace("?", pd.NA).isnull().sum()
        st.dataframe(missing.to_frame("Missing Count"), use_container_width=True)

    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# EDA
# ======================================================
elif menu == "📊 EDA":
    st.header("Exploratory Data Analysis")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    numeric_cols = df_clean.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = df_clean.select_dtypes(include=["object"]).columns.tolist()

    eda_choice = st.selectbox(
        "Choose EDA output",
        [
            "Descriptive Statistics",
            "Numeric Distribution",
            "Categorical Count",
            "Boxplot",
            "Correlation Heatmap"
        ]
    )

    if eda_choice == "Descriptive Statistics":
        if st.button("Generate Descriptive Statistics", use_container_width=True):
            st.dataframe(df_clean.describe(), use_container_width=True)

    elif eda_choice == "Numeric Distribution":
        col = st.selectbox("Choose numeric column", numeric_cols)

        if st.button("Generate Distribution Plot", use_container_width=True):
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.histplot(df_clean[col], kde=True, ax=ax)
            ax.set_title(f"Distribution of {col}")
            st.pyplot(fig)

    elif eda_choice == "Categorical Count":
        col = st.selectbox("Choose categorical column", categorical_cols)

        if st.button("Generate Countplot", use_container_width=True):
            fig, ax = plt.subplots(figsize=(8, 4.8))
            order = df_clean[col].value_counts().head(12).index
            sns.countplot(data=df_clean, y=col, order=order, ax=ax)
            ax.set_title(f"Top Categories in {col}")
            st.pyplot(fig)

    elif eda_choice == "Boxplot":
        col = st.selectbox("Choose numeric column", numeric_cols)

        if st.button("Generate Boxplot", use_container_width=True):
            fig, ax = plt.subplots(figsize=(8, 3.8))
            sns.boxplot(data=df_clean, x=col, ax=ax)
            ax.set_title(f"Boxplot of {col}")
            st.pyplot(fig)

    elif eda_choice == "Correlation Heatmap":
        if st.button("Generate Correlation Heatmap", use_container_width=True):
            fig, ax = plt.subplots(figsize=(9, 5.5))
            corr = df_clean[numeric_cols].corr()
            sns.heatmap(corr, annot=True, cmap="YlOrBr", fmt=".2f", ax=ax)
            ax.set_title("Correlation Heatmap")
            st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# PREPROCESSING
# ======================================================
elif menu == "🧹 Preprocessing":
    st.header("Preprocessing")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-note">
        This page shows the transformation flow from raw data to model-ready data.
        Click each button to inspect a preprocessing result.
    </div>
    """, unsafe_allow_html=True)

    if st.button("Show Cleaning Summary", use_container_width=True):
        c1, c2, c3 = st.columns(3)
        c1.metric("Before Cleaning", df.shape[0])
        c2.metric("After Cleaning", df_clean.shape[0])
        c3.metric("Rows Removed", df.shape[0] - df_clean.shape[0])

        st.write("""
        Cleaning steps applied:
        - remove leading and trailing spaces in categorical data  
        - replace `?` with missing values  
        - drop rows containing missing values  
        - clean income labels  
        """)

    if st.button("Show Encoded Dataset", use_container_width=True):
        st.subheader("Encoded Data Preview")
        st.dataframe(df_encoded.head(15), use_container_width=True, height=340)

    selected_feature = st.selectbox(
        "Choose feature to preview scaling",
        df_encoded.drop("income", axis=1).columns.tolist()
    )

    if st.button("Preview Selected Scaler", use_container_width=True):
        preview = df_encoded[[selected_feature]].copy()
        selected_scaler = get_scaler(scaler_name)

        if selected_scaler is not None:
            preview["scaled_value"] = selected_scaler.fit_transform(preview)
        else:
            preview["scaled_value"] = preview[selected_feature]

        st.write(f"Selected scaler: **{scaler_name}**")
        st.dataframe(preview.head(20), use_container_width=True)

        fig, ax = plt.subplots(figsize=(8, 4))
        sns.histplot(preview["scaled_value"], kde=True, ax=ax)
        ax.set_title(f"Scaled Distribution of {selected_feature}")
        st.pyplot(fig)

    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# MODEL SETUP
# ======================================================
elif menu == "⚙️ Model Setup":
    st.header("Model Setup")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    st.markdown("""
    <div class="info-note">
        The scaler and model are selected from the sidebar. This page explains the current setup.
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    c1.metric("Selected Scaler", scaler_name)
    c2.metric("Selected Model", model_name)

    if st.button("Show Training Configuration", use_container_width=True):
        st.json({
            "test_size": 0.2,
            "random_state": 42,
            "stratify": True,
            "selected_scaler": scaler_name,
            "selected_model": model_name,
            "train_samples": len(X_train),
            "test_samples": len(X_test),
            "features": X_train.shape[1]
        })

    if st.button("Show Model Object", use_container_width=True):
        st.code(str(model), language="python")

    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# EVALUATION
# ======================================================
elif menu == "🤖 Evaluation":
    st.header("Model Evaluation")

    st.markdown('<div class="content-card">', unsafe_allow_html=True)

    st.write(f"Current experiment: **{model_name}** with **{scaler_name}**.")

    if st.button("Run Evaluation", use_container_width=True):
        acc = accuracy_score(y_test, y_pred)
        class_names = list(encoders["income"].classes_) if "income" in encoders else ["0", "1"]
        cm = confusion_matrix(y_test, y_pred)

        c1, c2, c3 = st.columns(3)
        c1.metric("Accuracy", f"{acc:.2%}")
        c2.metric("Test Samples", len(y_test))
        c3.metric("Features", X_test.shape[1])

        st.subheader("Confusion Matrix")
        fig, ax = plt.subplots(figsize=(5.5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="YlOrBr",
            xticklabels=class_names,
            yticklabels=class_names,
            ax=ax
        )
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        st.pyplot(fig)

        st.subheader("Classification Report")
        st.text(classification_report(y_test, y_pred, target_names=class_names))

    st.markdown('</div>', unsafe_allow_html=True)


# ======================================================
# PREDICTION
# ======================================================
elif menu == "🎯 Prediction":
    st.header("Salary Prediction")

    st.markdown("""
    <div class="info-note">
        Fill in the form below, then click <b>Predict Salary</b>.
        The prediction uses the scaler and model chosen in the sidebar.
    </div>
    """, unsafe_allow_html=True)

    with st.form("prediction_form"):
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

        submitted = st.form_submit_button("Predict Salary", use_container_width=True)

    if submitted:
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

        if scaler is not None:
            input_processed = scaler.transform(input_df)
        else:
            input_processed = input_df

        pred_encoded = model.predict(input_processed)[0]
        pred_label = get_income_label(pred_encoded)
        confidence = get_prediction_confidence(model, input_processed)

        is_high_income = pred_label == ">50K"

        if is_high_income:
            st.markdown(f"""
            <div class="prediction-card-success">
                <div class="prediction-title">Prediction Result</div>
                <div class="prediction-main">Predicted Income: {pred_label}</div>
                <div class="prediction-sub">
                    The selected model predicts that this profile belongs to the <b>higher income</b> category.
                    {"Model confidence: <b>" + str(round(confidence * 100, 2)) + "%</b>." if confidence is not None else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="prediction-card-warning">
                <div class="prediction-title">Prediction Result</div>
                <div class="prediction-main">Predicted Income: {pred_label}</div>
                <div class="prediction-sub">
                    The selected model predicts that this profile belongs to the <b>lower income</b> category.
                    {"Model confidence: <b>" + str(round(confidence * 100, 2)) + "%</b>." if confidence is not None else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.subheader("Prediction Summary")
            st.write(f"**Model used:** {model_name}")
            st.write(f"**Scaler used:** {scaler_name}")
            st.write(f"**Predicted class:** {pred_label}")
            if confidence is not None:
                st.write(f"**Confidence:** {round(confidence * 100, 2)}%")
            st.markdown('</div>', unsafe_allow_html=True)

        with col_b:
            st.markdown('<div class="content-card">', unsafe_allow_html=True)
            st.subheader("Interpretation")
            if is_high_income:
                st.write("""
                This means the profile is estimated to have a higher likelihood of earning
                more than 50K per year based on the current model and selected preprocessing setup.
                """)
            else:
                st.write("""
                This means the profile is estimated to have a higher likelihood of earning
                50K or less per year based on the current model and selected preprocessing setup.
                """)
            st.markdown('</div>', unsafe_allow_html=True)

        with st.expander("View Encoded Input Data"):
            st.dataframe(input_df, use_container_width=True)