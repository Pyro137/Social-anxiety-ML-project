import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import pickle
import plotly.graph_objects as go

# Set page configuration
st.set_page_config(
    page_title="Sosyal Anksiyete Değerlendirme",
    page_icon="🧠",
    layout="wide"
)

# Custom CSS to improve the appearance
st.markdown("""
    <style>
    .main {
        padding: 2rem;
    }
    .stButton>button {
        width: 100%;
        background-color: #FF4B4B;
        color: white;
    }
    .stProgress .st-bo {
        background-color: #FF4B4B;
    }
    .sidebar .sidebar-content {
        background-color: #f5f5f5;
    }
    h1 {
        color: #FF4B4B;
        text-align: center;
    }
    .result-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
        background-color: #f8f9fa;
        border: 2px solid #FF4B4B;
    }
    </style>
""", unsafe_allow_html=True)

# Modeli yükle
def load_model():
    model_path = "src/models/xgb_clf.pkl"
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

def create_gauge_chart(value, title):
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [0, 10]},
            'bar': {'color': "#FF4B4B"},
            'steps': [
                {'range': [0, 3], 'color': "lightgray"},
                {'range': [3, 7], 'color': "gray"},
                {'range': [7, 10], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': value
            }
        }
    ))
    fig.update_layout(height=300)
    return fig

def user_input_features():
    st.sidebar.header("📝 Kullanıcı Bilgileri")
    
    # Organize inputs into sections
    with st.sidebar.expander("👤 Kişisel Bilgiler", expanded=True):
        age = st.slider("Yaş", 18, 100, 29)
        gender = st.selectbox("Cinsiyet", ("Female", "Male", "Other"))
        occupation = st.selectbox("Meslek", ("Artist", "Athlete", "Chef", "Doctor", "Engineer", "Freelancer", 
                                           "Lawyer", "Musician", "Nurse", "Other", "Scientist", "Student", "Teacher"))

    with st.sidebar.expander("🌙 Yaşam Tarzı", expanded=True):
        sleep_hours = st.slider("Uyku Saatleri", 0.0, 24.0, 6.0)
        physical_activity = st.slider("Fiziksel Aktivite (saat/hafta)", 0.0, 20.0, 2.7)
        diet_quality = st.slider("Diyet Kalitesi (1-10)", 1, 10, 7)

    with st.sidebar.expander("🍷 Alışkanlıklar", expanded=True):
        caffeine_intake = st.slider("Kafein Tüketimi (mg/gün)", 0, 1000, 181)
        alcohol_consumption = st.slider("Alkol Tüketimi (içecek/hafta)", 0, 10, 10)
        smoking = st.selectbox("Sigara İçiyor musunuz?", ("Yes", "No"))

    with st.sidebar.expander("❤️ Sağlık Göstergeleri", expanded=True):
        heart_rate = st.slider("Kalp Atış Hızı (bpm)", 40, 180, 114)
        breathing_rate = st.slider("Solunum Hızı (nefes/dakika)", 10, 50, 14)
        sweating_level = st.slider("Terleme Seviyesi (1-5)", 1, 5, 4)
        dizziness = st.selectbox("Baş Dönmesi", ("Yes", "No"))

    with st.sidebar.expander("🏥 Tıbbi Geçmiş", expanded=True):
        family_history = st.selectbox("Ailede Anksiyete Hikayesi?", ("Yes", "No"))
        medication = st.selectbox("İlaç Kullanıyor musunuz?", ("Yes", "No"))
        therapy_sessions = st.slider("Terapi Seansları (ayda)", 0, 10, 3)

    with st.sidebar.expander("😰 Stres Faktörleri", expanded=True):
        stress_level = st.slider("Stres Seviyesi (1-10)", 1, 10, 10)
        recent_life_event = st.selectbox("Son Zamanlarda Önemli Bir Hayat Olayı?", ("Yes", "No"))

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
    
    return pd.DataFrame(data, index=[0])

# Verileri ön işleme
def preprocess_data(input_data, encoder, scaler):
    numerical_cols = input_data.select_dtypes(include=['int64', 'float64']).columns
    scaled_data = scaler.transform(input_data[numerical_cols])
    numerical_preprocessed_data = pd.DataFrame(data=scaled_data, columns=numerical_cols)
    
    categorical_cols = [col for col in input_data.columns if input_data[col].dtype == "object"]
    encoded_categorical_data = encoder.transform(input_data[categorical_cols])
    
    preprocessed_data = pd.concat([numerical_preprocessed_data, encoded_categorical_data], axis=1)
    return preprocessed_data

def get_anxiety_description(level):
    if level <= 3:
        return "Düşük Seviye Anksiyete", "Anksiyete seviyeniz düşük görünüyor. Mevcut durumunuzu korumaya devam edin."
    elif level <= 7:
        return "Orta Seviye Anksiyete", "Orta seviyede anksiyete belirtileri gösteriyorsunuz. Bir uzmana danışmayı düşünebilirsiniz."
    else:
        return "Yüksek Seviye Anksiyete", "Yüksek seviyede anksiyete belirtileri gösteriyorsunuz. Bir ruh sağlığı uzmanına başvurmanız önerilir."

def main():
    st.title("🧠 Sosyal Anksiyete Değerlendirme Sistemi")
    
    st.markdown("""
    <div style='text-align: center; padding: 1rem;'>
        Bu uygulama, verdiğiniz bilgiler doğrultusunda sosyal anksiyete seviyenizi değerlendirir. 
        Lütfen sol menüdeki bilgileri doldurarak değerlendirmeyi başlatın.
    </div>
    """, unsafe_allow_html=True)

    # Load model and preprocessors
    model = load_model()
    encoder, scaler = load_encoder_scaler()
    
    # Get user input
    user_data = user_input_features()
    
    # Add a "Değerlendir" button
    if st.sidebar.button("📊 Değerlendir"):
        # Process data and make prediction
        preprocessed_data = preprocess_data(user_data, encoder, scaler)
        prediction = model.predict(preprocessed_data)
        anxiety_level = int(prediction[0])
        
        # Create columns for results
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Display gauge chart
            fig = create_gauge_chart(anxiety_level, "Anksiyete Seviyesi")
            st.plotly_chart(fig, use_container_width=True)
            
        with col2:
            # Display results and recommendations
            level_text, description = get_anxiety_description(anxiety_level)
            st.markdown(f"""
            <div class='result-box'>
                <h3 style='color: #FF4B4B;'>{level_text}</h3>
                <p>{description}</p>
            </div>
            """, unsafe_allow_html=True)
            
        # Display risk factors
        st.subheader("🔍 Risk Faktörleri")
        risk_factors = []
        if user_data["Stress Level (1-10)"].iloc[0] > 7:
            risk_factors.append("Yüksek stres seviyesi")
        if user_data["Sleep Hours"].iloc[0] < 6:
            risk_factors.append("Yetersiz uyku")
        if user_data["Physical Activity (hrs/week)"].iloc[0] < 2:
            risk_factors.append("Düşük fiziksel aktivite")
        
        if risk_factors:
            st.warning("Dikkat edilmesi gereken faktörler: " + ", ".join(risk_factors))
        
        # Add disclaimer
        st.markdown("""
        <div style='font-size: 0.8em; color: gray; text-align: center; margin-top: 2rem;'>
            ⚠️ Bu değerlendirme sadece bilgilendirme amaçlıdır ve profesyonel tıbbi tavsiye yerine geçmez. 
            Endişeleriniz varsa lütfen bir sağlık uzmanına başvurun.
        </div>
        """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
