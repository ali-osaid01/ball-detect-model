import cv2
import numpy as np
from inference_sdk import InferenceHTTPClient

# Initialize the Roboflow client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="dOuAxm6u8yAxmRJ4zss8"
)

def scale_to_arkit_range(value, old_min, old_max, new_min, new_max):
    return new_min + (value - old_min) * (new_max - new_min) / (old_max - old_min)

def detect_balls(input_image_path, output_image_path, screen_width=2556, screen_height=1179):
    # Load the image
    img = cv2.imread(input_image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found at {input_image_path}")
    
    # Get original image dimensions
    img_height, img_width = img.shape[:2]
    print(f"Original Image Dimensions: Width={img_width}, Height={img_height}")
    print(f"Screen Dimensions: Width={screen_width}, Height={screen_height}")

    result = CLIENT.infer(img, model_id="tennis-ball-detection-owaqx/2")
    detections = result.get('predictions', [])

    arkit_x_range = (200, 280)
    arkit_y_range = (450, 550)

    print(f"Raw Detections: {detections}")

    for detection in detections:
        center_x = detection['x']
        center_y = detection['y']
        width = detection['width']
        height = detection['height']

        top_left_x = int(center_x - (width / 2))
        top_left_y = int(center_y - (height / 2))
        bottom_right_x = int(center_x + (width / 2))
        bottom_right_y = int(center_y + (height / 2))

        arkit_x = scale_to_arkit_range(center_x, 0, img_width, *arkit_x_range)
        arkit_y = scale_to_arkit_range(center_y, 0, img_height, *arkit_y_range)

        print(f"ARKit Space -> X: {arkit_x:.2f}, Y: {arkit_y:.2f}")

        label = detection.get('label', 'ball')
        confidence = detection.get('confidence', 0.0)
        cv2.rectangle(img, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), (0, 255, 0), 2)
        cv2.putText(img, f"{label} ({confidence:.2f})", (top_left_x, top_left_y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

   
    cv2.imwrite(output_image_path, img)
    print(f"Annotated image saved at {output_image_path}")

# Example usage
if __name__ == "__main__":
    input_image_path = "./ball2.jpeg"  # Replace with your input image path
    output_image_path = "./output_detected_balls.jpg"  # Replace with your desired output image path

    detect_balls(input_image_path, output_image_path)
