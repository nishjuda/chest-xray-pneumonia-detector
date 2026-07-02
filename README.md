# Chest X-Ray Pneumonia Detection System

This project uses a deep learning model to classify chest X-ray images as either **NORMAL** or **PNEUMONIA**. The model was built using transfer learning with MobileNetV2 and deployed using FastAPI and Gradio.

## Features

* Pneumonia detection from chest X-ray images
* Transfer learning using MobileNetV2
* FastAPI backend for model inference
* Gradio interface for uploading images and viewing predictions
* Confidence score for predictions
* Grad-CAM visualization to understand where the model focused

## Model Architecture

* Base Model: MobileNetV2 (pretrained on ImageNet)
* Input Size: 224 × 224 × 3
* Classification Head:

  * GlobalAveragePooling2D
  * Dense(128, ReLU)
  * Dropout(0.3)
  * Dense(1, Sigmoid)

## Results

* Test Accuracy: ~87%
* Classes:

  * NORMAL
  * PNEUMONIA

## Project Structure

chest_xray/

* app.py – FastAPI backend
* frontend.py – Gradio interface
* train.py – Model training script
* grad_cam.py – Grad-CAM generation
* models/

  * pneumonia-classifier.keras
* requirements.txt
* README.md
* .gitignore

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd chest_xray
```

Install the required packages:

```bash
pip install -r requirements.txt
```

## Running the Application

Start the FastAPI server:

```bash
uvicorn app:app --reload
```

Start the Gradio interface:

```bash
python frontend.py
```

The application will open locally in your browser.

## Note

Grad-CAM highlights the regions that influenced the model's prediction. It does **not** indicate the exact location of pneumonia and should only be used as an explainability tool.

## Disclaimer

This project was developed for learning and research purposes and should not be used for clinical diagnosis.
