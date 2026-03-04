import cv2
import numpy as np


class VoodooSlabCentering:

    def __init__(self):
        pass

    # --------------------------------
    # Crop slab edges deterministically
    # --------------------------------
    def detect_border_thickness(self, gray):

        h, w = gray.shape

        # Smooth heavily to remove design noise
        blur = cv2.GaussianBlur(gray, (25, 25), 0)

        # LEFT border
        left = 0
        for x in range(w // 3):
            column_mean = np.mean(blur[:, x])
            if column_mean < 200:  # leaving white region
                left = x
                break

        # RIGHT border
        right = 0
        for x in range(w - 1, 2 * w // 3, -1):
            column_mean = np.mean(blur[:, x])
            if column_mean < 200:
                right = w - x
                break

        # TOP border
        top = 0
        for y in range(h // 3):
            row_mean = np.mean(blur[y, :])
            if row_mean < 200:
                top = y
                break

        # BOTTOM border
        bottom = 0
        for y in range(h - 1, 2 * h // 3, -1):
            row_mean = np.mean(blur[y, :])
            if row_mean < 200:
                bottom = h - y
                break

        return left, right, top, bottom

    # --------------------------------
    # Main entry
    # --------------------------------
    def analyze_array(self, image_array):

        # Downscale for stability
        target_width = 1000
        h, w = image_array.shape[:2]
        scale = target_width / w
        image = cv2.resize(image_array, (target_width, int(h * scale)))

        # Remove slab border
        cropped = self.crop_slab(image)

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        left, right, top, bottom = self.detect_border_thickness(gray)

        if min(left, right, top, bottom) == 0:
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "confidence": 0.0
            }

        horizontal_ratio = min(left, right) / max(left, right)
        vertical_ratio   = min(top, bottom) / max(top, bottom)

        return {
            "horizontal_ratio": float(horizontal_ratio),
            "vertical_ratio": float(vertical_ratio),
            "confidence": 1.0
        }
