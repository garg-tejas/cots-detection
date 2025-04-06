import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd
import os
import torch

# Set Streamlit page settings
st.set_page_config(page_title="COTS Detection App", page_icon="🌊", layout="centered")

# Display header image
st.image("assets/header.png", use_container_width=True)

# Title and description
st.title("🌊 COTS Detection App")
st.write("Upload a reef image and detect crown-of-thorns starfish (COTS) with bounding boxes!")

# Load YOLO model once and cache it
@st.cache_resource
def load_model():
    return YOLO('runs/detect/train/weights/best.pt')

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # Save uploaded image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name

    # Predict button
    predict_button = st.button("🔍 Predict COTS")

    if predict_button:
        with st.spinner('Detecting crown-of-thorns starfish...'):
            results = model(temp_path)
            result_image = results[0].plot()

        st.success("✅ Detection Complete!")
        st.image(result_image, caption="Detection Result", use_container_width=True)

        # Show detection details
        st.subheader("📋 Detection Details:")

        detections = results[0].boxes
        if detections is not None and detections.data is not None:
            data = detections.data.cpu().numpy()
            columns = ["x_min", "y_min", "x_max", "y_max", "confidence", "class_id"]
            df = pd.DataFrame(data, columns=columns)

            # Map class IDs to class names if available
            if hasattr(results[0], "names"):
                df["class_name"] = df["class_id"].apply(lambda x: results[0].names[int(x)])

            df = df[["class_name", "confidence", "x_min", "y_min", "x_max", "y_max"]]
            st.dataframe(df.style.format({"confidence": "{:.2f}"}), use_container_width=True)
        else:
            st.info("No COTS detected.")

        os.remove(temp_path)
else:
    st.info("Please upload a reef image to start detection.")
