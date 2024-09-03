import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import requests

def load_lottieurl(url):
    r = requests.get(url)
    if r.status_code != 200:
        return None
    return r.json()

def get_clean_data():
    data = pd.read_csv("data/data.csv")
    return data

def add_sidebar():
    st.sidebar.markdown("# 🩺 Input Measurements")
    data = get_clean_data()
    slider_labels = [
        ("Pregnancies", "Pregnancies"),
        ("Glucose (mg/dL)", "Glucose"),
        ("Blood Pressure (mm Hg)", "BloodPressure"),
        ("Skin Thickness (mm)", "SkinThickness"),
        ("Insulin (μU/mL)", "Insulin"),
        ("BMI", "BMI"),
        ("Diabetes Pedigree Function", "DiabetesPedigreeFunction"),
        ("Age (years)", "Age"),
    ]
    input_dict = {}
    for label, key in slider_labels:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def get_scaled_values(input_dict):
    data = get_clean_data()
    X = data.drop(['Outcome'], axis=1)
    scaled_dict = {}
    for key, value in input_dict.items():
        max_val = X[key].max()
        min_val = X[key].min()
        scaled_value = (value - min_val) / (max_val - min_val)
        scaled_dict[key] = scaled_value
    return scaled_dict

def get_radar_chart(input_data):
    input_data = get_scaled_values(input_data)
    categories = ['Pregnancies', 'Glucose', 'Blood Pressure', 'Skin Thickness',
                  'Insulin', 'BMI', 'Diabetes Pedigree', 'Age']
    
    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=[input_data[key] for key in input_data.keys()],
        theta=categories,
        fill='toself',
        name='Patient Data',
        line_color='rgb(67, 147, 195)'
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 1],
                tickfont=dict(
                color='black',  
                size=14       
            )
            )
        ),
        showlegend=False,
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white')
    )
    return fig

def add_predictions(input_data):
    model = pickle.load(open("model/model.pkl", "rb"))
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    
    input_array = np.array(list(input_data.values())).reshape(1, -1)
    input_array_scaled = scaler.transform(input_array)
    prediction = model.predict(input_array_scaled)
    
    with open("assets/style.css") as f:
        st.markdown("<style>{}</style>".format(f.read()), unsafe_allow_html=True)
  
    st.markdown("## 🔬 Diagnosis")
    
    if prediction[0] == 0:
        st.markdown("<h5 style='color: #2ecc71;'>Patient is likely <span style='background-color: #2ecc71; color: white; padding: 0.2rem 0.5rem; border-radius: 0.5rem;'>NOT DIABETIC</span></h3>", unsafe_allow_html=True)
    else:
        st.markdown("<h5 style='color: #e74c3c;'>Patient is likely <span style='background-color: #e74c3c; color: white; padding: 0.2rem 0.5rem; border-radius: 0.5rem;'>DIABETIC</span></h3>", unsafe_allow_html=True)
    
    probabilities = model.predict_proba(input_array_scaled)[0]
    st.write(f"Probability of being non-diabetic: {probabilities[0]:.2%}")
    st.write(f"Probability of being diabetic: {probabilities[1]:.2%}")
    
    st.markdown("---")
    st.markdown("⚠️ **Disclaimer:** This app is for educational purposes only and should not replace professional medical advice.")

def main():
    st.set_page_config(
        page_title="AI-Powered Diabetes Predictor",
        page_icon="🩺",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Custom CSS
    st.markdown("""
    <style>
    .main {
        background-color: #1e1e1e;
        color: white;
    }
    .stApp {
        background: linear-gradient(135deg, #1e1e1e 25%, #2c2c2c 100%);
    }
    /* Sidebar styling */
    [data-testid=stSidebar] {
        background-color: #111;#2c3e50;
        color: #ecf0f1;
    }
    [data-testid=stSidebar] h1 {
        color: #3498db;
    }
    [data-testid=stSidebar] .stMarkdown {
        color: #100;#bdc3c7;
    }
    h1, h2, h3 {
        color: #3498db;
    }
    /* Slider label styling */
    [data-testid=stSidebar] .stSlider label {
        font-size: 1.05em;
        font-weight: bold;
        color: #999;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        margin-bottom: 5px;
    }
    /* Optional: Add a subtle hover effect */
    [data-testid=stSidebar] .stSlider label:hover {
        color: #3498db;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.2);
    }

    /* Footer styling */
    .footer {
        position: relative;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: rgba(44, 62, 80, 0.7);
        color: #ecf0f1;
        text-align: center;
        font-size: 0.8em;
    }
    </style>
    """, unsafe_allow_html=True)


    st.title("🩺 AI-Powered Diabetes Predictor")
    st.markdown("---")
    st.markdown("## How It Works")
    st.write("""
    1. Input patient data using the sliders in the sidebar.
    2. The radar chart visualizes the scaled input values.
    3. Our AI model analyzes the data to predict diabetes risk.
    4. The diagnosis is displayed with probability percentages.
    
    Remember: This tool is meant to assist healthcare professionals, not replace them.
    """)


    input_data = add_sidebar()
    
    col1, col2 = st.columns([3, 2])
    
    with col1:
    
        st.header("Polar Chart for the Measurements")
        st.write("Harness the power of machine learning to assess diabetes risk based on patient data.")
        
        radar_chart = get_radar_chart(input_data)
        st.plotly_chart(radar_chart, use_container_width=True)
        
    
    with col2:
        add_predictions(input_data)
    
    

    st.markdown(
        """
        <div class="footer">
            © 2024 Anes MOULAI-KHATIR. All Rights Reserved.
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == '__main__':
    main()