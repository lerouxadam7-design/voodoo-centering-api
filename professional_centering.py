import cv2
import numpy as np


class VoodooRawCardCentering:

    def __init__(self):
        pass

    def analyze_array(self, image_array):

        # Downscale
        target_width = 1000
        h, w = image_array.shape[:2]
        scale = target_width / w
        image = cv2.resize(image_array, (target_width, int(h * scale)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        h2, w2 = gray.shape

        # Small crop to remove tiny background edge
        crop_margin = int(w2 * 0.02)
        gray = gray[:, crop_margin:w2 - crop_margin]

        h2, w2 = gray.shape

        # Heavy blur to smooth reflections
        blur = cv2.GaussianBlur(gray, (31, 31), 0)

        # LEFT border detection
        left = 0
        for x in range(w2 // 3):
            if np.mean(blur[:, x]) < 200:
                left = x
                break

        # RIGHT border detection
        right = 0
        for x in range(w2 - 1, 2 * w2 // 3, -1):
            if np.mean(blur[:, x]) < 200:
                right = w2 - x
                break

        # TOP border detection
        top = 0
        for y in range(h2 // 3):
            if np.mean(blur[y, :]) < 200:
                top = y
                break

        # BOTTOM border detection
        bottom = 0
        for y in range(h2 - 1, 2 * h2 // 3, -1):
            if np.mean(blur[y, :]) < 200:
                bottom = h2 - y
                break

        if min(left, right, top, bottom) == 0:
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "confidence": 0.0
            }

        horizontal_ratio = min(left, right) / max(left, right)
        vertical_ratio = min(top, bottom) / max(top, bottom)

        return {
            "horizontal_ratio": float(horizontal_ratio),
            "vertical_ratio": float(vertical_ratio),
            "confidence": 1.0
        }
