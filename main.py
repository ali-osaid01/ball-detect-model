from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np

app = FastAPI()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dOuAxm6u8yAxmRJ4zss8"
)

@app.get("/")
async def root():
    return JSONResponse({"message":"Welcome to Toucher App -> AI Model YOLO"})
    
@app.post("/detect-balls")
async def detect_balls(file: UploadFile = File(...)):
    image_data = await file.read()
    np_img = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Extract the dimensions of the image
    height, width, _ = img.shape

    # Perform inference
    result = CLIENT.infer(img, model_id="tennis-ball-detection-owaqx/2")

    detections = result.get('predictions', [])
    balls_coordinates = []

    for detection in detections:
        x = detection['x']  
        y = detection['y']  
        width_box = detection['width']
        height_box = detection['height']

        balls_coordinates.append({
            "label": detection.get('label', 'ball'),
            "x": x,
            "y": y,
            "width": width_box,
            "height": height_box,
        })

    # Return detections along with image dimensions
    return JSONResponse(content={
        "image_width": width,
        "image_height": height,
        "balls_detected": balls_coordinates
    })
