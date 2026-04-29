import streamlit as st
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

model = joblib.load('model.pkl')
scaler = joblib.load('scaler.pkl')
encoder = joblib.load('encoders.pkl')
st.markdown("""
<style>

/* Background utama */
.stApp {
    background: linear-gradient(to right, #141e30, #243b55);
    color: white;
}

/* Judul */
h1, h2, h3 {
    color: white;
    text-align: center;
    margin-top: 10px;
}


/* Tabs */
.stTabs [data-baseweb="tab"] {
    background-color: rgba(255,255,255,0.1);
    border-radius: 10px;
    padding: 10px;
    color: white;
}

/* DataFrame container */
[data-testid="stDataFrame"] {
    background-color: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 8px;
}

/* Header tabel */
[data-testid="stDataFrame"] th {
    background-color: rgba(0, 201, 167, 0.8);
    color: white;
}

/* Isi tabel */
[data-testid="stDataFrame"] td {
    color: white;
}

/* Hover effect */
[data-testid="stDataFrame"] tr:hover td {
    background-color: rgba(255,255,255,0.08);
}

/* Button */
.stButton>button {
    background-color: #00c9a7;
    color: white;
    border-radius: 12px;
    padding: 10px;
}

/* Input */
.stSelectbox>div, .stTextInput>div>div>input {
    background-color: #1e293b;
    color: white;
    border-radius: 8px;
}

</style>
""", unsafe_allow_html=True)

columns = [
    "age", "workclass", "fnlwgt", "education", "education_num",
    "marital_status", "occupation", "relationship", "race",
    "sex", "capital_gain", "capital_loss", "hours_per_week",
    "native_country", "income"
]

@st.cache_data
def ambil_data():
    data = pd.read_csv('adult.csv', names=columns, skipinitialspace=True)
    return data

df = ambil_data()
st.markdown("<h1 style='font-size: 55px; text-align: center;'>💰SmartSalary</h1>", unsafe_allow_html=True)
st.markdown("<h3 style='text-align: center; font-size: 28px;'>PREDICT YOUR INCOME INSTANTLY</h3>", unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs(["📂 Dataset",
    "🧹 Preprocessing",
    "🤖 Model",
    "🎯 Prediksi"])

with tab1:
    st.header("Dataset")
    st.dataframe(df.head())
    st.write("Shape : ", df.shape)
    st.subheader("📊 Ringkasan Dataset")

    summary = pd.DataFrame({
        "Kolom": df.columns,
        "Tipe Data": df.dtypes.values,
        "Non-Null": df.notnull().sum().values,
        "Missing": df.isnull().sum().values
    })

    st.dataframe(summary)

    st.subheader("📈 Statistik Data Numerik")
    st.dataframe(df.describe())

    st.subheader("🔍 Sample Data")
    st.dataframe(df.head(20))
with tab2:
    st.header("Preprocessing Time ⏳")
    st.write("Melakukan preprocessing pada data...")
    df_clean = df.copy()
    st.subheader("Membersihkan Data")
    df_clean = df_clean.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    st.write("Data setelah dibersihkan:")
    st.dataframe(df_clean.head())
    st.subheader("Mengisi Nilai yang Hilang")
    df_clean = df_clean.replace('?', pd.NA)
    df_clean = df_clean.dropna()
    st.dataframe(df_clean.head())
    st.subheader("Encoding Data")
    for col in df_clean.select_dtypes(include="object").columns:
        df_clean[col] = encoder[col].transform(df_clean[col])
    st.dataframe(df_clean.head())
with tab3:
    st.title("Model dan Evaluasi")

    df_model = df.copy()
    df_model = df_model.apply(lambda x: x.str.strip() if x.dtype == "object" else x)
    df_model = df_model.replace('?', pd.NA)
    df_model = df_model.dropna()
    df_model["income"] = df_model["income"].str.replace(".", "", regex=False)
    for col in df_model.select_dtypes(include="object").columns:
        df_model[col] = encoder[col].transform(df_model[col])

    X = df_model.drop("income", axis = 1)
    y = df_model["income"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

    X_test = scaler.transform(X_test)
    y_pred = model.predict(X_test)

    st.subheader("Accuracy Score")
    st.success(f"{accuracy_score(y_test, y_pred):.2f}")

    st.subheader("Classification Report")
    st.text(f"{classification_report(y_test, y_pred)}")

    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    st.write(cm)

with tab4:
    st.header("Input Data")

    age = st.slider("Age", 18, 65, 30)
    workclass = st.selectbox("Workclass", encoder["workclass"].classes_)
    education = st.selectbox("Education", encoder["education"].classes_)
    marital = st.selectbox("Marital Status", encoder["marital_status"].classes_)
    occupation = st.selectbox("Occupation", encoder["occupation"].classes_)
    relationship = st.selectbox("Relationship", encoder["relationship"].classes_)
    race = st.selectbox("Race", encoder["race"].classes_)
    sex = st.selectbox("Sex", encoder["sex"].classes_)
    country = st.selectbox("Country", encoder["native_country"].classes_)
    education_num = st.slider("Education Num", 1, 16, 10)
    hours_per_week = st.slider("Hours Per Week Work", 1, 80, 40)
    default_values = [0] * 11
    if st.button("Prediksi"):
        input_dict = {
            "age": age,
            "workclass": encoder["workclass"].transform([workclass])[0],
            "fnlwgt": 100000,  # default realistis
            "education": encoder["education"].transform([education])[0],
            "education_num": education_num,
            "marital_status": encoder["marital_status"].transform([marital])[0],
            "occupation": encoder["occupation"].transform([occupation])[0],
            "relationship": encoder["relationship"].transform([relationship])[0],
            "race": encoder["race"].transform([race])[0],
            "sex": encoder["sex"].transform([sex])[0],
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": hours_per_week,
            "native_country": encoder["native_country"].transform([country])[0]
        }
        input_df = pd.DataFrame([input_dict])
        input_scaled = scaler.transform(input_df)
        prediction = model.predict(input_scaled)

        if prediction[0] == 1:
            st.success("💸 Gaji > 50K")
        else:
            st.warning("💼 Gaji <= 50K")
