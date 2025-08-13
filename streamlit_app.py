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
    page_title="Deepfake Image Detector (Keras)",
    page_icon="🕵️‍♂️",
    layout="wide"
)

# =========================
# CSS Styling
# =========================
st.markdown(
    """
    <style>
    /* Global styles for white background and black text */
    html, body, .stApp {
        background-color: white !important;
        color: black !important;
        font-family: 'Inter', sans-serif !important;
    }

    /* Ensure specific Streamlit text components are black on light backgrounds */
    .stMarkdown, .stText, .stSpinner div, .stProgress div, .stDownloadButton {
        color: black !important;
    }

    /* Sidebar background and styling */
    [data-testid="stSidebar"] {
        background-color: white !important;
        color: black !important;
        box-shadow: 2px 0 5px -2px rgba(0,0,0,0.1);
        border-radius: 0.75rem !important;
    }

    /* Headers color */
    h1, h2, h3, h4, h5, h6 {
        color: black !important;
    }

    /* Apply rounded corners and shadows to various Streamlit elements */
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
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }

    /* File uploader button styling */
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
    .stFileUpload > div > button:active {
        transform: translateY(0) !important;
        box-shadow: none !important;
    }

    /* General button styling */
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
    .stButton > button:active {
        transform: translateY(0) !important;
        box-shadow: none !important;
    }

    /* Image display container */
    .stImage > div {
        border: 2px solid black !important;
        padding: 5px !important;
        background-color: white !important;
    }

    /* Prediction result box */
    .prediction-box {
        border: 2px solid black !important;
        padding: 15px !important;
        background-color: white !important;
        margin-bottom: 1rem;
    }

    /* Dataframe container styling */
    .stDataFrame {
        border: 2px solid black !important;
        overflow: hidden !important;
    }

    /* Styling for Streamlit's alert boxes (info, success, error) */
    .stAlert {
        border-radius: 0.75rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -1px rgba(0, 0, 0, 0.06);
    }
    </style>
    """,
    unsafe_allow_html=True
)

# =========================
# Constants
# =========================
MODEL_PATH = 'deepfake_inference_model.keras' # Path to your Keras model file
THRESHOLD = 0.5 # Probability threshold to classify as fake

# =========================
# Load Keras Model
# =========================
@st.cache_resource
def load_keras_model():
    """
    Loads the Keras model from the specified path, fixing channel mismatch.
    """
    try:
        # Load without enforcing input shape
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)

        # If model expects 1 channel, rebuild to accept 3 channels
        if model.input_shape[3] == 1:
            st.warning("⚠ Model expects 1-channel input. Adjusting to accept RGB...")
            from tensorflow.keras import layers, models

            # New RGB input
            new_input = layers.Input(shape=(model.input_shape[1], model.input_shape[2], 3))
            # Convert RGB -> Grayscale to match original model expectation
            x = layers.Lambda(lambda x: tf.image.rgb_to_grayscale(x))(new_input)
            x = model(x)
            model = models.Model(inputs=new_input, outputs=x)

        height, width, channels = model.input_shape[1], model.input_shape[2], model.input_shape[3]
        st.session_state['model_input_size'] = (width, height)
        st.session_state['model_input_channels'] = channels

        print(f"✅ Model loaded. Expected input size: {width}x{height} with {channels} channels.")
        return model

    except Exception as e:
        st.error(
            f"🚨 **Error:** Could not load the Keras model from `{MODEL_PATH}`. "
            f"Ensure the file is in the same directory as this script. "
            f"**Details:** `{e}`"
        )
        return None


# =========================
# Preprocess Image (Corrected Version)
# =========================
def preprocess_image(image, target_size, expected_channels=3):
    # Force RGB
    if expected_channels == 3:
        image = image.convert("RGB")
    elif expected_channels == 1:
        image = image.convert("L")  # Grayscale if model expects 1 channel
    
    # Resize
    img_resized = image.resize(target_size, resample=Image.BICUBIC)
    
    # Convert to NumPy
    img_array = np.array(img_resized).astype(np.float32)
    
    # If still grayscale but RGB expected, stack channels
    if expected_channels == 3 and img_array.ndim == 2:
        img_array = np.stack((img_array,)*3, axis=-1)
    
    # Preprocess for EfficientNet
    img_preprocessed = tf.keras.applications.efficientnet.preprocess_input(img_array)
    img_batch = np.expand_dims(img_preprocessed, axis=0)
    return img_batch


# =========================
# Predict Deepfake
# =========================
def keras_predict(model, img_batch):
    """
    Performs inference using the loaded Keras model.
    """
    output_data = model.predict(img_batch)
    probability = float(output_data[0][0])
    is_fake = probability > THRESHOLD
    return is_fake, probability

# =========================
# Display Confidence Bar
# =========================
def display_confidence_bar(filename, probability):
    """
    Displays a horizontal bar chart visualizing the confidence of
    "Real" vs. "Fake" predictions.
    """
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
    """
    Main function to run the Streamlit Deepfake Image Detector application.
    """
    st.markdown("<h1 style='text-align: center; font-size: 3.5rem; margin-bottom: 0.5rem;'>🕵️‍♂️ Deepfake Image Detector (Keras)</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.1rem; color: #555;'>Unmasking synthetic imagery with AI</p>", unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("""
    This application utilizes a **fine-tuned EfficientNet Keras model** to discern whether an image is
    authentic or synthetically generated. Upload one or more images below for analysis.
    """)

    # Sidebar for 'About' information
    st.sidebar.title("✨ About This Detector")
    st.sidebar.info("""
    - **Model Architecture:** EfficientNet (or similar CNN)
    - **Model Type:** Keras (`.keras` file)
    - **Prediction Threshold:** Images with a probability greater than `0.5` are classified as fake.
    - **Supported Uploads:** JPG, JPEG, PNG images
    """)
    st.sidebar.markdown("---")

    # Load Keras model at the start
    model = load_keras_model()
    if model is None:
        st.stop()
    
    # Retrieve model input size and channels from session state (set during model loading)
    model_input_size = st.session_state.get('model_input_size')
    model_input_channels = st.session_state.get('model_input_channels')

    if model_input_size is None or model_input_channels is None:
        st.error("Could not determine model input dimensions. Please restart the app.")
        st.stop()

    st.markdown("---")
    st.subheader("📤 Upload Your Images for Analysis")
    uploaded_files = st.file_uploader(
        "Choose image files (JPG, JPEG, PNG):",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Upload one or more images to check authenticity. Max file size: 200MB."
    )

    if uploaded_files:
        results = []
        st.markdown("---")
        st.subheader("🔍 Analysis Results")

        for uploaded_file in uploaded_files:
            image = Image.open(uploaded_file)
            st.write(f"### Analyzing: **{uploaded_file.name}**")

            with st.spinner("Processing image and making a prediction..."):
                try:
                    # Pass dynamically determined target_size and expected_channels
                    img_batch = preprocess_image(image, model_input_size, model_input_channels)
                    is_fake, probability = keras_predict(model, img_batch)
                except ValueError as ve:
                    st.error(f"Error processing image '{uploaded_file.name}': {ve}")
                    continue # Skip to the next file if preprocessing fails

            col1, col2 = st.columns([1, 2])

            with col1:
                st.image(image, caption=f"Uploaded: {uploaded_file.name}", use_container_width=True)

            with col2:
                st.markdown(f"<div class='prediction-box'>", unsafe_allow_html=True)
                st.markdown("<h3 style='margin-top: 0; color: black;'>Prediction Outcome</h3>", unsafe_allow_html=True)

                if is_fake:
                    st.error("🚨 **FAKE IMAGE DETECTED!** This image likely originated from an AI.")
                else:
                    st.success("✅ **REAL IMAGE.** This image appears to be authentic.")

                st.markdown(f"**Confidence (Fake):** `{probability*100:.2f}%`")
                display_confidence_bar(uploaded_file.name, probability)
                st.markdown("</div>", unsafe_allow_html=True)

            results.append({
                "Filename": uploaded_file.name,
                "Prediction": "FAKE" if is_fake else "REAL",
                "Confidence (Fake)": f"{probability*100:.2f}%"
            })
            st.markdown("---")

        if results:
            st.markdown("### 📊 Overall Summary")
            df_results = pd.DataFrame(results)
            st.dataframe(df_results, use_container_width=True)
    else:
        st.info("👆 Please upload one or more images above to initiate the deepfake detection process.")
        st.markdown("---")

if __name__ == "__main__":
    # Initialize session state for model dimensions if not already present
    if 'model_input_size' not in st.session_state:
        st.session_state['model_input_size'] = None
        st.session_state['model_input_channels'] = None
    
    main()
