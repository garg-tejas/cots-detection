# COTS Detection App

## 🌊 Overview

The COTS Detection App is an interactive web application that uses computer vision to identify and classify crown-of-thorns starfish (COTS) in underwater images from the Great Barrier Reef. Built with Streamlit and YOLOv8, it provides real-time detection with bounding boxes and confidence scores. This project leverages the "TensorFlow - Help Protect the Great Barrier Reef" dataset to contribute to coral reef conservation efforts.

## ✨ Features

- Upload reef images in common formats (JPG, JPEG, PNG)
- Detect crown-of-thorns starfish with visual bounding boxes
- Display detection results with confidence scores
- User-friendly interface with simple, intuitive controls

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

1. Clone the repository:

   ```bash
   git clone https://github.com/garg-tejas/cots-detection.git
   cd cots-detection
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Make sure you have the trained model in the correct location:
   ```
   runs/detect/train/weights/best.pt
   ```

## 🔍 Usage

1. Start the Streamlit application:

   ```bash
   streamlit run app.py
   ```

2. Open your web browser and go to the provided URL (typically http://localhost:8501)

3. Upload a reef image using the file uploader

4. Click the "Predict COTS" button to run the detection

5. View the results with bounding boxes and detailed information about detected starfish

## 🧠 Model Information

The application uses a YOLOv8 model trained on the "TensorFlow - Help Protect the Great Barrier Reef" dataset. This dataset was created to help detect crown-of-thorns starfish, which pose a significant threat to coral reefs by feeding on coral polyps. Early detection of these starfish can help conservation teams take action to protect the Great Barrier Reef ecosystem.

The model is capable of identifying crown-of-thorns starfish in underwater imagery with high accuracy, providing a valuable tool for marine conservationists and researchers.

## 📋 Requirements

Check requirements.txt for the full list of dependencies.
