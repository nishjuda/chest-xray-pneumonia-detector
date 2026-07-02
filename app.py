from fastapi import UploadFile, File, FastAPI
import uvicorn
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
app = FastAPI()
from PIL import Image
import numpy as np
import os
DIR_PATH = os.path.dirname(__file__)
MODEL_PATH = os.path.join(DIR_PATH,"models","pneumonia-classifier.keras")


model = load_model(MODEL_PATH)

@app.get("/")
def home_page():
    return {"message" : "the server is running with the model active"}

@app.post("/predict")
def predict(file : UploadFile = File(description = "upload the file with chest x-ray u wanna predict the presence of pneumonia")):
    img = Image.open(file.file)
    img = img.convert("RGB")
    img = img.resize((224,224))

    arr = np.array(img)
    arr = preprocess_input(arr)

    arr = np.expand_dims(arr,axis=0)

    prediction = model.predict(arr)
    probability = (float(prediction[0][0]))

    if probability > 0.5:
        result = "pneumonia"
    else:
        result = "normal"
    
    return {
        "prediction" : result,
        "probablity" : f"{probability*100:.3f}%",
        "score": probability
    }

