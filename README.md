# Chest X-Ray Pneumonia Detection System

This project uses a Convolutional Neural Network based on MobileNetV2 to classify chest X-ray images as either **Normal** or **Pneumonia**. The trained model is served through a FastAPI backend and can be run locally or inside a Docker container.

## Features

- Pneumonia detection from chest X-ray images
- Transfer learning using MobileNetV2
- FastAPI REST API
- Interactive Gradio interface
- Docker support for deployment

## Technologies Used

- Python
- TensorFlow / Keras
- FastAPI
- Uvicorn
- Gradio
- Docker
- NumPy
- Pillow

## Project Structure

```
.
├── models/
├── screenshots/
├── app.py
├── frontend.py
├── train.py
├── Dockerfile
├── requirements.txt
└── README.md
```

## Installation

Clone the repository.

```bash
git clone https://github.com/nishjuda/chest-xray-pneumonia-detector.git
cd chest-xray-pneumonia-detector
```

Install the required packages.

```bash
pip install -r requirements.txt
```

Start the FastAPI server.

```bash
python app.py
```

Open the Swagger documentation at:

```
http://localhost:8000/docs
```

## Running with Docker

Build the Docker image.

```bash
docker build -t pneumonia-app .
```

Run the container.

```bash
docker run -d -p 8000:8000 --name pneumonia-container pneumonia-app
```

The API will be available at:

```
http://localhost:8000/docs
```

## API

| Method | Endpoint | Description |
|---------|----------|-------------|
| GET | `/` | Home page |
| POST | `/predict` | Predict whether an X-ray is Normal or Pneumonia |

## Future Improvements

- Deploy the application online
- Replace the Gradio interface with a React frontend
- Add prediction history using a database
- Improve model performance with more data

## Author

Nish Juda

GitHub: https://github.com/nishjuda
