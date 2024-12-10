from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from inference_sdk import InferenceHTTPClient
import cv2
import numpy as np
import os
from datetime import datetime

app = FastAPI()

CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dOuAxm6u8yAxmRJ4zss8"
)

@app.get("/")
async def root():
    return JSONResponse({"message":"Welcome to Toucher App -> AI Model YOLO"})
    

@app.post("/detect-ball")
async def visualize_detections(
    file: UploadFile = File(...),
    screen_width: float = 1290.0,  # AR screen width
    screen_height: float = 2796.0  # AR screen height
):
   
    image_data = await file.read()
    np_img = np.frombuffer(image_data, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    # Get image dimensions
    img_height, img_width, _ = img.shape

    # Use provided screen dimensions or default to image dimensions
    screen_width = screen_width or img_width
    screen_height = screen_height or img_height

    # Perform inference
    result = CLIENT.infer(img, model_id="tennis-ball-detection-owaqx/2")
    detections = result.get('predictions', [])

    img_with_boxes = img.copy()
    balls_data = []

    for detection in detections:
        # Original pixel coordinates
        x = int(detection['x'])
        y = int(detection['y'])
        w = int(detection['width'])
        h = int(detection['height'])

        # Calculate bounding box coordinates
        x1 = int(x - w / 2)
        y1 = int(y - h / 2)
        x2 = int(x + w / 2)
        y2 = int(y + h / 2)

        # Draw rectangle for visualization
        # cv2.rectangle(img_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Calculate normalized coordinates (0-1)
        norm_x = x / img_width
        norm_y = y / img_height
        norm_w = w / img_width
        norm_h = h / img_height

        # Calculate AR screen coordinates
        ar_x = norm_x * screen_width
        ar_y = norm_y * screen_height
        ar_w = norm_w * screen_width
        ar_h = norm_h * screen_height

        # arx1 = int(ar_x - ar_w / 2)
        # ary1 = int(ar_y - ar_h / 2)
        # arx2 = int(ar_x + ar_w / 2)
        # ary2 = int(ar_y + ar_h / 2)

        # ar marking
        # cv2.rectangle(img_with_boxes, (arx1, ary1), (arx2, ary2), (0, 255, 0), 2)
        #
        # # Save marked image
        # timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # filename = f"marked_ball_{img_width}x{img_height}_{timestamp}.jpg"
        # filepath = os.path.join(MARKED_IMAGES_DIR, filename)
        # cv2.imwrite(filepath, img_with_boxes)

        ball_data = {
            "label": detection.get('label', 'ball'),
            "confidence": detection.get('confidence', None),
            "original_pixels": {
                "center": {"x": x, "y": y},
                "width": w,
                "height": h,
                "bounds": {
                    "left": x1,
                    "top": y1,
                    "right": x2,
                    "bottom": y2
                }
            },
            "normalized": {  # 0-1 range, useful for any screen size
                "center": {"x": norm_x, "y": norm_y},
                "width": norm_w,
                "height": norm_h,
                "bounds": {
                    "left": x1 / img_width,
                    "top": y1 / img_height,
                    "right": x2 / img_width,
                    "bottom": y2 / img_height
                }
            },
            "ar_screen": {  # Coordinates mapped to AR screen
                "center": {"x": ar_x, "y": ar_y},
                "width": ar_w,
                "height": ar_h,
                "bounds": {
                    "left": (x1 / img_width) * screen_width,
                    "top": (y1 / img_height) * screen_height,
                    "right": (x2 / img_width) * screen_width,
                    "bottom": (y2 / img_height) * screen_height
                }
            }
        }

        balls_data.append(ball_data)

    return JSONResponse(content={
        "dimensions": {
            "image": {
                "width": img_width,
                "height": img_height
            },
            "ar_screen": {
                "width": screen_width,
                "height": screen_height
            }
        },
        # "marked_image_path": filepath,
        "balls_detected": balls_data,
        "total_balls": len(balls_data)
    })

