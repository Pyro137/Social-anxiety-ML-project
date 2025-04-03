import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle

# Modeli yükle
def load_model():
    model_path = "src/models/xgb_clf.pkl"  # Model dosyanızın yolu
    with open(model_path, "rb") as f:
        model_info = pickle.load(f)
    return model_info

# Encoder ve Scaler'ı yükle
def load_encoder_scaler():
    with open('src/models/encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('src/models/scaler.pkl', 'rb') as f:
        scaler = pickle.load(f)
    return encoder, scaler

# Kullanıcıdan veri girişi al
def user_input_features():
    st.sidebar.header("Kullanıcı Girişi")
    
    # Kullanıcıdan değerler al
    age = st.sidebar.slider("Yaş", 18, 100, 29)
    gender = st.sidebar.selectbox("Cinsiyet", ("Female", "Male", "Other"))
    occupation = st.sidebar.selectbox("Meslek", ("Artist", "Athlete", "Chef", "Doctor", "Engineer", "Freelancer", 
                                                 "Lawyer", "Musician", "Nurse", "Other", "Scientist", "Student", "Teacher"))
    sleep_hours = st.sidebar.slider("Uyku Saatleri", 0.0, 24.0, 6.0)
    physical_activity = st.sidebar.slider("Fiziksel Aktivite (saat/hafta)", 0.0, 20.0, 2.7)
    caffeine_intake = st.sidebar.slider("Kafein Tüketimi (mg/gün)", 0, 1000, 181)
    alcohol_consumption = st.sidebar.slider("Alkol Tüketimi (içecek/hafta)", 0, 10, 10)
    smoking = st.sidebar.selectbox("Sigara İçiyor musunuz?", ("Yes", "No"))
    family_history = st.sidebar.selectbox("Ailede Anksiyete Hikayesi?", ("Yes", "No"))
    stress_level = st.sidebar.slider("Stres Seviyesi (1-10)", 1, 10, 10)
    heart_rate = st.sidebar.slider("Kalp Atış Hızı (bpm)", 40, 180, 114)
    breathing_rate = st.sidebar.slider("Solunum Hızı (nefes/dakika)", 10, 50, 14)
    sweating_level = st.sidebar.slider("Terleme Seviyesi (1-5)", 1, 5, 4)
    dizziness = st.sidebar.selectbox("Baş Dönmesi", ("Yes", "No"))
    medication = st.sidebar.selectbox("İlaç Kullanıyor musunuz?", ("Yes", "No"))
    therapy_sessions = st.sidebar.slider("Terapi Seansları (ayda)", 0, 10, 3)
    recent_life_event = st.sidebar.selectbox("Son Zamanlarda Önemli Bir Hayat Olayı?", ("Yes", "No"))
    diet_quality = st.sidebar.slider("Diyet Kalitesi (1-10)", 1, 10, 7)
    
    data = {
        "Age": age,
        "Gender": gender,
        "Occupation": occupation,
        "Sleep Hours": sleep_hours,
        "Physical Activity (hrs/week)": physical_activity,
        "Caffeine Intake (mg/day)": caffeine_intake,
        "Alcohol Consumption (drinks/week)": alcohol_consumption,
        "Smoking": smoking,
        "Family History of Anxiety": family_history,
        "Stress Level (1-10)": stress_level,
        "Heart Rate (bpm)": heart_rate,
        "Breathing Rate (breaths/min)": breathing_rate,
        "Sweating Level (1-5)": sweating_level,
        "Dizziness": dizziness,
        "Medication": medication,
        "Therapy Sessions (per month)": therapy_sessions,
        "Recent Major Life Event": recent_life_event,
        "Diet Quality (1-10)": diet_quality
    }
    
    features = pd.DataFrame(data, index=[0])
    return features

# Verileri ön işleme
def preprocess_data(input_data, encoder, scaler):
    # Özelliklerin ön işlenmesi
    numerical_cols = input_data.select_dtypes(include=['int64', 'float64']).columns
    scaled_data = scaler.transform(input_data[numerical_cols])  # Use the loaded scaler
    numerical_preprocessed_data = pd.DataFrame(data=scaled_data, columns=numerical_cols)
    
    categorical_cols = [col for col in input_data.columns if input_data[col].dtype == "object"]
    encoded_categorical_data = encoder.transform(input_data[categorical_cols])  # Use the loaded encoder
    
    preprocessed_data = pd.concat([numerical_preprocessed_data, encoded_categorical_data], axis=1)
    return preprocessed_data

# Ana fonksiyon
def main():
    st.title("Anksiyete Tahmini Uygulaması")
    
    # Modeli yükle
    model = load_model()
    
    # Encoder ve Scaler'ı yükle
    encoder, scaler = load_encoder_scaler()
    
    # Kullanıcıdan verileri al
    user_data = user_input_features()
    
    # Verileri işleme
    preprocessed_data = preprocess_data(user_data, encoder, scaler)
    
    # Tahmin yap
    prediction = model.predict(preprocessed_data)
    
    # The prediction is already on a scale from 1-10, so directly display it
    st.write("Tahmin Sonucu:")
    st.write(f"Anksiyete Seviyesi: {int(prediction[0])}")  # Displaying the predicted level directly (1-10 scale)

# Streamlit uygulamasını başlat
if __name__ == "__main__":
    main()
