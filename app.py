import streamlit as st
from ultralytics import YOLO
from PIL import Image
import tempfile
import pandas as pd
import os
import torch

st.set_page_config(page_title="COTS Detection App", page_icon="🌊", layout="centered")
st.title("🌊 COTS Detection App")
st.write("Upload a reef image and detect crown-of-thorns starfish (COTS) with bounding boxes!")

uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_bytes = uploaded_file.read()
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Save image temporarily for processing
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        temp_file.write(file_bytes)
        temp_path = temp_file.name
    
# WHILE RUNNING LOCALLY
    # @st.cache_resource
    # def load_model():
    #     return YOLO('runs/detect/train/weights/best.pt')
    
    @st.cache_resource
    def load_model():
        model_path = "model/best.pt"
        
        # Download if not exists
        if not os.path.exists(model_path):
            os.makedirs("model", exist_ok=True)
            # For Hugging Face
            from huggingface_hub import hf_hub_download
            hf_hub_download(repo_id="ggtejas/cots-detector", 
                           filename="best.pt")
        
        return YOLO(model_path)

    predict_button = st.button("🔍 Predict COTS")
    
    if predict_button:
        model = load_model()
        
        with st.spinner('Detecting crown-of-thorns starfish...'):
            results = model(temp_path)
            result_image = results[0].plot()

        st.success("Detection Complete!")
        st.image(result_image, caption="Detection Result", use_container_width=True)
        
        st.subheader("📋 Detection Details:")
        
        detections = results[0].boxes
        if detections is not None and detections.data is not None:
            data = detections.data.cpu().numpy()
            columns = ["x_min", "y_min", "x_max", "y_max", "confidence", "class_id"]
            df = pd.DataFrame(data, columns=columns)
            
            if hasattr(results[0], "names"):
                df["class_name"] = df["class_id"].apply(lambda x: results[0].names[int(x)])
            
            df = df[["class_name", "confidence", "x_min", "y_min", "x_max", "y_max"]]
            st.dataframe(df.style.format({"confidence": "{:.2f}"}), use_container_width=True)
        else:
            st.info("No COTS detected.")
else:
    st.info("Please upload a reef image to start detection.")