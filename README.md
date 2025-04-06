# 🌊 COTS Detection App

### Crown-of-Thorns Starfish Detection in Underwater Images

---

## 📖 Overview

The **COTS Detection App** is an interactive web tool that leverages **YOLOv8** and **Streamlit** to automatically detect and classify **Crown-of-Thorns Starfish (COTS)** in reef images.  
Built using the _TensorFlow - Help Protect the Great Barrier Reef_ dataset, this app aims to assist marine conservation efforts by providing fast, accurate detection of a major threat to coral ecosystems.

---

## ✨ Features

- 📸 Upload reef images (JPG, JPEG, PNG)
- 🧠 Real-time COTS detection with bounding boxes
- 🎯 Displays confidence scores for each detection
- 🖥️ Simple and intuitive Streamlit-based interface

---

## 🚀 Installation

### Prerequisites

- Python 3.8+
- pip package manager

### Setup

```bash
git clone https://github.com/garg-tejas/cots-detection.git
cd cots-detection
pip install -r requirements.txt
```

> **Note:** Make sure you have your trained model file at:
>
> ```
> runs/detect/train/weights/best.pt
> ```

---

## 🔍 Usage

Start the application locally:

```bash
streamlit run app.py
```

Then open your browser at [http://localhost:8501](http://localhost:8501).

### How to Use

1. Upload an underwater reef image.
2. Click **"Predict COTS"**.
3. View the image with bounding boxes and detection confidence.

---

## 🧠 Model Details

This project uses a **YOLOv8** model trained on the _TensorFlow - Help Protect the Great Barrier Reef_ dataset.  
It can reliably identify Crown-of-Thorns Starfish in various underwater environments, helping marine biologists and conservation teams intervene before large-scale reef damage occurs.

---

## 📦 Requirements

See [`requirements.txt`](requirements.txt) for the full list of dependencies.

---

## ❤️ Acknowledgements

- [TensorFlow - Great Barrier Reef Challenge](https://www.kaggle.com/competitions/tensorflow-great-barrier-reef)
- [Ultralytics YOLOv8](https://docs.ultralytics.com/)
- [Streamlit](https://streamlit.io/)

---

# 📸 Demo

![App Demo](demo/demo.gif)
