import cv2
import numpy as np


class VoodooSlabCentering:

    def __init__(self):
        pass

    def detect_border_thickness(self, gray):

        h, w = gray.shape

        # Threshold to isolate white border
        _, thresh = cv2.threshold(gray, 210, 255, cv2.THRESH_BINARY)

        # LEFT border
        left_border = 0
        for x in range(w // 3):
            column = thresh[:, x]
            if np.mean(column) < 245:
                left_border = x
                break

        # RIGHT border
        right_border = 0
        for x in range(w - 1, 2 * w // 3, -1):
            column = thresh[:, x]
            if np.mean(column) < 245:
                right_border = w - x
                break

        # TOP border
        top_border = 0
        for y in range(h // 3):
            row = thresh[y, :]
            if np.mean(row) < 245:
                top_border = y
                break

        # BOTTOM border
        bottom_border = 0
        for y in range(h - 1, 2 * h // 3, -1):
            row = thresh[y, :]
            if np.mean(row) < 245:
                bottom_border = h - y
                break

        return left_border, right_border, top_border, bottom_border

    def analyze_array(self, image_array):

        # Downscale
        target_width = 1000
        h, w = image_array.shape[:2]
        scale = target_width / w
        image = cv2.resize(image_array, (target_width, int(h * scale)))

        h2, w2 = image.shape[:2]

        # Remove slab plastic via deterministic crop
        crop_pct = 0.12
        x_start = int(w2 * crop_pct)
        x_end   = int(w2 * (1 - crop_pct))
        y_start = int(h2 * crop_pct)
        y_end   = int(h2 * (1 - crop_pct))

        cropped = image[y_start:y_end, x_start:x_end]

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
