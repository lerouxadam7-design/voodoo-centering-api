import cv2
import numpy as np


class VoodooSlabCentering:

    def __init__(self):
        pass

    # --------------------------------
    # Crop slab edges deterministically
    # --------------------------------
    def crop_slab(self, image):

        h, w = image.shape[:2]

        # Slab border crop (works because framing is consistent)
        crop_pct = 0.12

        x_start = int(w * crop_pct)
        x_end   = int(w * (1 - crop_pct))
        y_start = int(h * crop_pct)
        y_end   = int(h * (1 - crop_pct))

        return image[y_start:y_end, x_start:x_end]

    # --------------------------------
    # Detect white border thickness
    # --------------------------------
    def detect_white_borders(self, gray):

        h, w = gray.shape

        # Dynamic threshold based on brightest region
        bright_percentile = np.percentile(gray, 95)
        threshold_value = bright_percentile - 5

        _, thresh = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

        # LEFT
        left = 0
        for x in range(w // 3):
            col = thresh[:, x]
            if np.mean(col) < 250:
                left = x
                break

        # RIGHT
        right = 0
        for x in range(w - 1, 2 * w // 3, -1):
            col = thresh[:, x]
            if np.mean(col) < 250:
                right = w - x
                break

        # TOP
        top = 0
        for y in range(h // 3):
            row = thresh[y, :]
            if np.mean(row) < 250:
                top = y
                break

        # BOTTOM
        bottom = 0
        for y in range(h - 1, 2 * h // 3, -1):
            row = thresh[y, :]
            if np.mean(row) < 250:
                bottom = h - y
                break

        return left, right, top, bottom

    # --------------------------------
    # Main entry
    # --------------------------------
    def analyze_array(self, image_array):

        target_width = 1000
        h, w = image_array.shape[:2]
        scale = target_width / w
        image = cv2.resize(image_array, (target_width, int(h * scale)))

        cropped = self.crop_slab(image)
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # Return grayscale statistics
        return {
            "mean_gray": float(np.mean(gray)),
            "min_gray": float(np.min(gray)),
            "max_gray": float(np.max(gray)),
            "confidence": 1.0
        }

        horizontal_ratio = min(left, right) / max(left, right)
        vertical_ratio   = min(top, bottom) / max(top, bottom)

        return {
            "horizontal_ratio": float(horizontal_ratio),
            "vertical_ratio": float(vertical_ratio),
            "confidence": 1.0
        }
