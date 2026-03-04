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

        # Slab border crop (consistent framing assumption)
        crop_pct = 0.12

        x_start = int(w * crop_pct)
        x_end   = int(w * (1 - crop_pct))
        y_start = int(h * crop_pct)
        y_end   = int(h * (1 - crop_pct))

        return image[y_start:y_end, x_start:x_end]

    # --------------------------------
    # Detect border thickness via gradient scan
    # --------------------------------
    def detect_border_thickness(self, gray):

        h, w = gray.shape

        # Compute vertical gradient
        grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

        abs_grad_x = np.abs(grad_x)
        abs_grad_y = np.abs(grad_y)

        # LEFT border
        left = 0
        for x in range(w // 3):
            column_strength = np.mean(abs_grad_x[:, x])
            if column_strength > 20:
                left = x
                break

        # RIGHT border
        right = 0
        for x in range(w - 1, 2 * w // 3, -1):
            column_strength = np.mean(abs_grad_x[:, x])
            if column_strength > 20:
                right = w - x
                break

        # TOP border
        top = 0
        for y in range(h // 3):
            row_strength = np.mean(abs_grad_y[y, :])
            if row_strength > 20:
                top = y
                break

        # BOTTOM border
        bottom = 0
        for y in range(h - 1, 2 * h // 3, -1):
            row_strength = np.mean(abs_grad_y[y, :])
            if row_strength > 20:
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
