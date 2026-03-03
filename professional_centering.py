import cv2
import numpy as np


class VoodooSlabCentering:

    def __init__(self):
        pass

    def detect_border_thickness(self, gray):

        h, w = gray.shape

        # Threshold to isolate white border
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)

        # LEFT border
        left_border = 0
        for x in range(w // 4):
            column = thresh[:, x]
            if np.mean(column) < 240:  # transition from white
                left_border = x
                break

        # RIGHT border
        right_border = 0
        for x in range(w - 1, 3 * w // 4, -1):
            column = thresh[:, x]
            if np.mean(column) < 240:
                right_border = w - x
                break

        # TOP border
        top_border = 0
        for y in range(h // 4):
            row = thresh[y, :]
            if np.mean(row) < 240:
                top_border = y
                break

        # BOTTOM border
        bottom_border = 0
        for y in range(h - 1, 3 * h // 4, -1):
            row = thresh[y, :]
            if np.mean(row) < 240:
                bottom_border = h - y
                break

        return left_border, right_border, top_border, bottom_border

    def analyze_array(self, image_array):

        # Downscale for stability
        target_width = 1000
        h, w = image_array.shape[:2]
        scale = target_width / w
        image = cv2.resize(image_array, (target_width, int(h * scale)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        left, right, top, bottom = self.detect_border_thickness(gray)

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
