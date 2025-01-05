import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# Streamlit App Title
st.title("Weather Classification Analysis")

# Load the dataset
data = pd.read_csv("klasifikasi_cuaca.csv")

# Sidebar
st.sidebar.header("Settings")

def main():
    # Exploratory Data Analysis (EDA)
    st.subheader("Dataset Information")
    st.write(data.head())
    st.write("--- Dataset Info ---")
    buffer = st.text_area("Dataset Info", data.info(buf=None))
    st.write("--- Descriptive Statistics ---")
    st.write(data.describe())

    # Distribution Analysis
    st.subheader("Distribution Analysis")
    numerical_features = ['Suhu (°C)', 'Kelembapan (%)', 'Kecepatan Angin (km/jam)']
    for col in numerical_features:
        st.write(f"Distribution of {col}")
        sns.histplot(data[col], kde=True, bins=10)

