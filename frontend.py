import frontend as gr
import requests

def predict(image):
    with open(image, "rb") as f:
        response = requests.post(
            "http://127.0.0.1:8000/predict",
            files={"file": f}
        )
    return response.json()

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="filepath"),
    outputs="json",
    title="🩻 Chest X-Ray Pneumonia Detection"
)

demo.launch()