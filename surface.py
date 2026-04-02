import cv2
import numpy as np

def analyze_surface(image_bytes):
    file_bytes = np.asarray(bytearray(image_bytes), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img is None:
        raise ValueError("Could not decode image")

    return {
        "message": "image loaded successfully",
        "shape": img.shape
    }
