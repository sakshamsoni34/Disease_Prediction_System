import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ---------------- PAGE CONFIG ----------------

st.set_page_config(
    page_title="AI Disease Prediction",
    page_icon="🏥",
    layout="wide"
)

# ---------------- LOAD MODEL ----------------

BASE_DIR = Path(__file__).resolve().parent


def load_artifact(filename):
    candidate_paths = [BASE_DIR / filename, BASE_DIR / "notebooks" / filename]
    for candidate_path in candidate_paths:
        if candidate_path.exists():
            return joblib.load(candidate_path)

    st.error(
        f"Missing {filename}. Save the trained artifact in the app folder "
        "or the notebooks folder before running the app."
    )
    st.stop()


model = load_artifact("disease_model.pkl")
label_encoder = load_artifact("label_encoder.pkl")

# ---------------- CUSTOM CSS ----------------

st.markdown("""
<style>

.main {
    background-color: #f7fafc;
}

.card {
    background:white;
    padding:20px;
    border-radius:15px;
    box-shadow:0px 4px 15px rgba(0,0,0,0.08);
}

.prediction-box {
    background:#ecfdf5;
    padding:20px;
    border-radius:12px;
    border-left:6px solid #10b981;
    color:#0f172a;
}

.prediction-box h3,
.prediction-box h2,
.prediction-box p,
.prediction-box span {
    color:#0f172a !important;
    opacity:1 !important;
    font-weight:700;
}

.footer {
    text-align:center;
    color:gray;
    margin-top:30px;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>

/* Full width black button */
div.stButton > button {
    width: 100%;
    background: #000000 !important;
    color: white !important;
    border: 1px solid #000000 !important;
    border-radius: 12px !important;
    height: 55px !important;
    font-size: 18px !important;
    font-weight: 600 !important;
    transition: all 0.3s ease;
}

/* Hover effect */
div.stButton > button:hover {
    background: #111111 !important;
    border: 1px solid #111111 !important;
    color: white !important;
}

/* Click effect */
div.stButton > button:active {
    background: #222222 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# ---------------- HERO SECTION ----------------

st.markdown("""
<div style='padding:25px;border-radius:15px;
background:linear-gradient(135deg,#0f766e,#14b8a6);
text-align:center;color:white;margin-bottom:20px'>

<h1>🏥 AI Disease Prediction System</h1>

<p>
Helping healthcare professionals identify possible diseases
based on patient symptoms.
</p>

</div>
""", unsafe_allow_html=True)

# ---------------- SIDEBAR ----------------

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/2966/2966486.png",
    width=100
)

st.sidebar.title("Patient Information")

patient_name = st.sidebar.text_input("Patient Name")
patient_age = st.sidebar.number_input("Age", 1, 120, 25)
gender = st.sidebar.selectbox(
    "Gender",
    ["Male", "Female", "Other"]
)

# ---------------- SYMPTOMS ----------------

st.markdown("### 🩺 Select Patient Symptoms")

col1, col2, col3, col4 = st.columns(4)

with col1:
    fever = st.checkbox("Fever")
    cough = st.checkbox("Cough")
    headache = st.checkbox("Headache")
    fatigue = st.checkbox("Fatigue")

with col2:
    nausea = st.checkbox("Nausea")
    stomach_pain = st.checkbox("Stomach Pain")
    vomiting = st.checkbox("Vomiting")
    diarrhea = st.checkbox("Diarrhea")

with col3:
    shortness_breath = st.checkbox("Shortness of Breath")
    chest_pain = st.checkbox("Chest Pain")
    dizziness = st.checkbox("Dizziness")
    high_bp = st.checkbox("High Blood Pressure")

with col4:
    sneezing = st.checkbox("Sneezing")
    runny_nose = st.checkbox("Runny Nose")
    joint_pain = st.checkbox("Joint Pain")
    itching = st.checkbox("Itching")

# ---------------- PREDICT BUTTON ----------------

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🔍 Predict Disease", use_container_width=True):

    symptoms = np.array([[
        int(fever),
        int(cough),
        int(headache),
        int(fatigue),
        int(nausea),
        int(stomach_pain),
        int(shortness_breath),
        int(chest_pain),
        int(sneezing),
        int(runny_nose),
        int(high_bp),
        int(dizziness),
        int(joint_pain),
        int(itching),
        int(vomiting),
        int(diarrhea)
    ]])

    prediction = model.predict(symptoms)

    disease = label_encoder.inverse_transform(prediction)[0]

    st.markdown("## 📋 Diagnosis Report")

    st.markdown(f"""
    <div class='prediction-box'>
        <h3>Predicted Disease</h3>
        <h2>{disease}</h2>
    </div>
    """, unsafe_allow_html=True)

    st.success("Prediction completed successfully.")

    st.markdown("### 👤 Patient Summary")

    st.write(f"**Name:** {patient_name}")
    st.write(f"**Age:** {patient_age}")
    st.write(f"**Gender:** {gender}")

# ---------------- FOOTER ----------------

st.markdown("""
<div class='footer'>
<hr>
AI Disease Prediction System | Built using Machine Learning & Streamlit
</div>
""", unsafe_allow_html=True)