import streamlit as st
import google.generativeai as genai
import pandas as pd

# Page setup
st.set_page_config(page_title="SensitiveCancerGPT", layout="centered")

# Title and description
st.title("SensitiveCancerGPT")
st.write("🧬 A biomedical AI app using Google Gemini for:")
st.markdown("- **Anti-cancer drug sensitivity prediction** (BRCA only)")
st.markdown("- **Breast cancer detection** based on clinical inputs")

# Step 1: Configure Gemini API key
try:
    gemini_key = 'AIzaSyBSBPcZEvpeMCcrHX6kIB1E0rweJW3DHJs'
    genai.configure(api_key=gemini_key)
    st.success("Gemini API key loaded successfully.")
except Exception as e:
    st.error(f"Failed to load Gemini API key: {e}")
    gemini_key = None

# Session state storage
if "sensitivity_results" not in st.session_state:
    st.session_state.sensitivity_results = []

if "detection_results" not in st.session_state:
    st.session_state.detection_results = []

# Function for sensitivity prediction
def predict_sensitivity(input_prompt, drug, cell):
    system_instructions = """
You are a biomedical expert AI assistant specialized in predicting anti-cancer drug sensitivity.

Your task is to predict whether a given drug is likely to be effective ("Sensitive") or not effective ("Resistant") for a specific cell line and cancer tissue type, based on known mechanisms, targets, and patterns.

Always end with: Prediction: Sensitive or Prediction: Resistant

Examples:

Tissue: LUAD
Drug: Gefitinib
Cell line: PC9
Prediction: Sensitive

Tissue: BRCA
Drug: Paclitaxel
Cell line: MDA-MB-231
Prediction: Resistant

Tissue: COREAD
Drug: Cetuximab
Cell line: HCT116
Prediction: Resistant

Now respond to:
"""
    full_prompt = f"{system_instructions}\nTissue: BRCA\n{input_prompt}"
    st.write("📤 Sending prompt to Gemini for drug sensitivity...")
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        response = model.generate_content(full_prompt)
        output = response.text.strip()
        st.write("✅ Gemini response:")
        st.write(output)
        result = output.split('\n')[-1]
        st.session_state.sensitivity_results.append({
            "Drug": drug,
            "Cell Line": cell,
            "Prediction": result
        })
        return result
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return f"Error: {e}"

# Function for cancer detection
def detect_cancer(clinical_info):
    detection_prompt = f"""
You are a medical AI assistant that detects whether a patient is likely to have breast cancer based on clinical indicators.

The format of the input includes: Age, Tumor Size (in mm), Lymph Node Status (positive/negative), Menopause Status (pre/post), and Tumor Grade (1-3). Provide a short explanation and end with: "Diagnosis: Positive" or "Diagnosis: Negative".

Now respond to:
{clinical_info}
"""
    st.write("📤 Sending prompt to Gemini for cancer detection...")
    try:
        model = genai.GenerativeModel(model_name='gemini-1.5-flash')
        response = model.generate_content(detection_prompt)
        output = response.text.strip()
        st.write("✅ Gemini response:")
        st.write(output)
        diagnosis = output.split('\n')[-1]
        st.session_state.detection_results.append({
            "Clinical Info": clinical_info,
            "Diagnosis": diagnosis
        })
        return diagnosis
    except Exception as e:
        st.error(f"Error calling Gemini API: {e}")
        return f"Error: {e}"

# Tabs for two features
tab1, tab2, tab3 = st.tabs(["🔬 Sensitivity Prediction", "🩺 Cancer Detection", "📊 Visualizations"])

# --- Tab 1: Sensitivity Prediction ---
with tab1:
    st.header("🔍 Anti-Cancer Drug Sensitivity Prediction (BRCA only)")

    user_input_drug = st.text_input("💊 Enter drug name:", "ML323")
    user_input_cell = st.text_input("🧬 Enter cell line name:", "USP1")
    st.info("Tissue type is fixed: `BRCA` (Breast invasive carcinoma)")

    if st.button("🔍 Predict Sensitivity"):
        if not user_input_drug or not user_input_cell:
            st.warning("⚠️ Please fill out both drug and cell line fields.")
        else:
            input_prompt = f"The drug is {user_input_drug}. The cell line is {user_input_cell}."
            pred = predict_sensitivity(input_prompt, user_input_drug, user_input_cell)
            st.subheader("🧠 Prediction Result:")
            st.success(pred)

# --- Tab 2: Cancer Detection ---
with tab2:
    st.header("🩺 Breast Cancer Detection")

    age = st.number_input("👩 Age (years):", min_value=20, max_value=100, value=45)
    tumor_size = st.number_input("📏 Tumor Size (mm):", min_value=1, max_value=100, value=20)
    lymph_nodes = st.selectbox("🧪 Lymph Node Status:", ["positive", "negative"])
    menopause = st.selectbox("🌙 Menopause Status:", ["pre", "post"])
    grade = st.selectbox("🎓 Tumor Grade:", [1, 2, 3])

    if st.button("🧾 Detect Cancer"):
        clinical_input = (
            f"Age: {age}, Tumor Size: {tumor_size}mm, Lymph Node Status: {lymph_nodes}, "
            f"Menopause Status: {menopause}, Tumor Grade: {grade}"
        )
        result = detect_cancer(clinical_input)
        st.subheader("🧠 Detection Result:")
        st.success(result)

# --- Tab 3: Visualizations ---
with tab3:
    st.header("📊 Stored Predictions Overview")

    # Sensitivity Results Table & Chart
    if st.session_state.sensitivity_results:
        st.subheader("🔬 Drug Sensitivity Predictions")
        df_sens = pd.DataFrame(st.session_state.sensitivity_results)
        st.dataframe(df_sens, use_container_width=True)

        chart_data = df_sens["Prediction"].value_counts()
        st.bar_chart(chart_data)
    else:
        st.info("No drug sensitivity predictions yet.")

    st.markdown("---")

    # Cancer Detection Results Table & Chart
    if st.session_state.detection_results:
        st.subheader("🩺 Cancer Detection Results")
        df_detect = pd.DataFrame(st.session_state.detection_results)
        st.dataframe(df_detect, use_container_width=True)

        pie_data = df_detect["Diagnosis"].value_counts()
        st.write("📈 Diagnosis Summary:")
        st.plotly_chart({
            "data": [{
                "type": "pie",
                "labels": pie_data.index.tolist(),
                "values": pie_data.tolist(),
                "hole": 0.4
            }],
            "layout": {"margin": {"l": 0, "r": 0, "b": 0, "t": 30}}
        }, use_container_width=True)
    else:
        st.info("No cancer detection results yet.")
