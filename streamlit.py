import streamlit as st
import pandas as pd
import numpy as np

# Streamlit App Title
st.title("Weather Classification Analysis")

# Load Dataset
@st.cache
def load_data():
    return pd.read_csv("klasifikasi_cuaca.csv")

data = load_data()

# Sidebar Navigation
st.sidebar.header("Navigation")
options = st.sidebar.radio(
    "Go to:",
    ["Dataset Overview", "Distribution Analysis", "Correlation Matrix", "Model Training & Evaluation"]
)

# Dataset Overview
if options == "Dataset Overview":
    st.subheader("Dataset Overview")
    st.write("### Dataset Information")
    buffer = st.text_area("Dataset Info", data.info(buf=None))  # Dataset info
    st.write("### Descriptive Statistics")
    st.write(data.describe())
    st.write("### Missing Values")
    st.write(data.isnull().sum())

# Distribution Analysis
elif options == "Distribution Analysis":
    st.subheader("Distribution Analysis")
    numerical_features = ['Suhu (°C)', 'Kelembapan (%)', 'Kecepatan Angin (km/jam)']
    for col in numerical_features:
        st.write(f"### Distribution of {col}")
        fig, ax = plt.subplots()
        sns.histplot(data[col], kde=True, bins=10, color='blue', ax=ax)
        ax.set_title(f"Distribution of {col}")
        st.pyplot(fig)

# Correlation Matrix
elif options == "Correlation Matrix":
    st.subheader("Correlation Matrix")
    numerical_features = ['Suhu (°C)', 'Kelembapan (%)', 'Kecepatan Angin (km/jam)']
    correlation_matrix = data[numerical_features].corr()
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f', ax=ax)
    ax.set_title('Correlation Matrix')
    st.pyplot(fig)

# Model Training & Evaluation
elif options == "Model Training & Evaluation":
    st.subheader("Model Training & Evaluation")

    # Preprocessing
    label_encoder = LabelEncoder()
    data['Jenis Cuaca Encoded'] = label_encoder.fit_transform(data['Jenis Cuaca'])
    scaler = StandardScaler()
    numerical_features = ['Suhu (°C)', 'Kelembapan (%)', 'Kecepatan Angin (km/jam)']
    data[numerical_features] = scaler.fit_transform(data[numerical_features])

    # Splitting data
    X = data[numerical_features]
    y = data['Jenis Cuaca Encoded']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Model Training
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train)

    # Model Evaluation
    predictions = model.predict(X_test)
    st.write("### Classification Report")
    report = classification_report(y_test, predictions, target_names=label_encoder.classes_, output_dict=True)
    st.dataframe(pd.DataFrame(report).transpose())

    # Confusion Matrix
    st.write("### Confusion Matrix")
    conf_matrix = confusion_matrix(y_test, predictions)
    fig, ax = plt.subplots()
    sns.heatmap(conf_matrix, annot=True, cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_, ax=ax)
    ax.set_title('Confusion Matrix')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')
    st.pyplot(fig)
