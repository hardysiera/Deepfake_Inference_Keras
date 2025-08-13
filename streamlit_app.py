import streamlit as st
import numpy as np
import tensorflow as tf
from PIL import Image
import cv2
import pandas as pd
import altair as alt

# =========================
# Page Configuration
# =========================
st.set_page_config(
    page_title="Deepfake Image Detector",
    page_icon="🖼️",
    layout="wide"
)

# =========================
# CSS Styling
# =========================
st.markdown(
    """
    <style>
    html, body, .stApp {
        background-color: white !important;
        color: black !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stMarkdown, .stText, .stSpinner div, .stProgress div, .stDownloadButton {
        color: black !important;
    }
    [data-testid="stSidebar"] {
        background-color: white !important;
        color: black !important;
        box-shadow: 2px 0 5px -2px rgba(0,0,0,0.1);
        border-radius: 0.75rem !important;
    }
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }
    .stImage > div,
    .prediction-box,
    .stDataFrame,
    .stFileUpload,
    .stTextInput,
    .stSelectbox,
    .stNumberInput,
    .stDateInput,
    .stTimeInput,
    .stCheckbox,
    .stRadio,
    .stSlider {
        border-radius: 0.75rem !important;
        border: 2px solid black !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    .stFileUpload > div > button {
        background-color: #f0f2f6 !important;
        color: black !important;
        border: 1px solid black !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.2s ease-in-out !important;
    }
    .stFileUpload > div > button:hover {
        background-color: #e2e8f0 !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button {
        background-color: #f0f2f6 !important;
        color: black !important;
        border: 1px solid black !important;
        border-radius: 0.75rem !important;
        padding: 0.5rem 1rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
        transition: all 0.2s ease-in-out !important;
    }
    .stButton > button:hover {
        background-color: #e2e8f0 !important;
        transform: translateY(-2px) !important;
    }
    .stImage > div {
        border: 2px solid black !important;
        padding: 5px !important;
        background-color: white !important;
    }
    .prediction-box {
        border: 2px solid black !important;
        padding: 15px !important;
        background-color: white !important;
        margin-bottom: 1rem;
    }
    .stDataFrame {
        border: 2px solid black !important;
        overflow: hidden !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Constants
# =========================
IMAGE_SIZE = (240, 240)
MODEL_PATH = 'deepfake_inference_model.keras'
THRESHOLD = 0.5

# =========================
# Load Keras Model
# =========================
@st.cache_resource
def load_keras_model():
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"🚨 **Error:** Could not load the Keras model from `{MODEL_PATH}`. **Details:** `{e}`")
        return None

# =========================
# Preprocess Image
# =========================
def preprocess_image(image):
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, IMAGE_SIZE)
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_resized)
    img_batch = np.expand_dims(img_preprocessed, axis=0).astype(np.float32)
    return img_batch

# =========================
# Predict
# =========================
def keras_predict(model, img_batch):
    probability = float(model.predict(img_batch)[0][0])
    is_fake = probability > THRESHOLD
    return is_fake, probability

# =========================
# Confidence Bar
# =========================
def display_confidence_bar(filename, probability):
    data = pd.DataFrame({
        'Label': ['Real', 'Fake'],
        'Confidence': [1 - probability, probability]
    })
    chart = alt.Chart(data).mark_bar().encode(
        x=alt.X('Confidence:Q', axis=alt.Axis(format='.1%', title='Confidence')),
        y=alt.Y('Label:N', sort=None, title=''),
        color=alt.Color('Label:N', scale=alt.Scale(range=['#28a745', '#dc3545']), legend=None)
    ).properties(
        width=400,
        height=70,
        title={
            "text": f"Confidence for {filename}",
            "anchor": "middle",
            "fontSize": 16,
            "color": "black"
        }
    ).configure_axis(
        labelColor='black',
        titleColor='black'
    ).configure_view(
        stroke='transparent'
    )
    st.altair_chart(chart, use_container_width=True)

# =========================
# Main App
# =========================
def main():
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0.5rem;'>🖼️ Deepfake Image Detector</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: black;'>Unmasking synthetic imagery with AI</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    This application uses a **fine-tuned EfficientNetB0 Keras model** to determine whether an image
    is real or AI-generated. Upload one or more images below for analysis.
    """)

    st.sidebar.title("✨ About This Detector")
    st.sidebar.info("""
    - **Model Architecture:** EfficientNetB0
    - **Model Type:** Keras
    - **Prediction Threshold:** >0.5 = Fake
    - **Supported Uploads:** JPG, JPEG, PNG
    """)
    st.sidebar.markdown("---")

    model = load_keras_model()
    if model is None:
        st.stop()

    st.subheader("📤 Upload Your Images for Analysis")
    uploaded_files = st.file_uploader(
        "Choose image files (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True
    )

    if uploaded_files:
        results = []
        st.subheader("🔍 Analysis Results")

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file).convert("RGB")
            st.write(f"### Analyzing: **{uploaded_file.name}**")
            with st.spinner("Processing..."):
                img_batch = preprocess_image(image)
                is_fake, probability = keras_predict(model, img_batch)

            col1, col2 = st.columns([1, 2])
            with col1:
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)
            with col2:
                st.markdown(f"<div class='prediction-box'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin-top: 0; color: black;'>Prediction Outcome</h3>", unsafe_allow_html=True)
                if is_fake:
                    st.error("🚨 **FAKE IMAGE DETECTED!** Likely AI-generated.")
                else:
                    st.success("✅ **REAL IMAGE.** Appears authentic.")
                st.markdown(f"**Confidence:** `{probability*100:.2f}%`")
                display_confidence_bar(uploaded_file.name, probability)
                st.markdown("</div>", unsafe_allow_html=True)

            results.append({
                "Filename": uploaded_file.name,
                "Prediction": "FAKE" if is_fake else "REAL",
                "Confidence": f"{probability*100:.2f}%"
            })
            st.markdown("---")

        st.markdown("### 📊 Overall Summary")
        df_results = pd.DataFrame(results)
        st.dataframe(df_results, use_container_width=True)
    else:
        st.info("👆 Please upload one or more images above to start detection.")

if __name__ == "__main__":
    main()
