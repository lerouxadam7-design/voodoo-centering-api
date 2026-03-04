import cv2
import numpy as np


class VoodooPredictiveCentering:

    def __init__(self):
        pass

    def analyze_array(self, image_array):

        # Downscale for consistency
        target_width = 800
        h, w = image_array.shape[:2]
        scale = target_width / w
        image = cv2.resize(image_array, (target_width, int(h * scale)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # Remove outer 5% to reduce background influence
        h2, w2 = gray.shape
        margin_x = int(w2 * 0.05)
        margin_y = int(h2 * 0.05)

        gray = gray[margin_y:h2 - margin_y, margin_x:w2 - margin_x]

        # Heavy blur to remove text and noise
        blur = cv2.GaussianBlur(gray, (31, 31), 0)

        h3, w3 = blur.shape

        # Split halves
        left_half = blur[:, :w3//2]
        right_half = blur[:, w3//2:]

        top_half = blur[:h3//2, :]
        bottom_half = blur[h3//2:, :]

        # Mirror for comparison
        right_mirror = np.fliplr(right_half)
        bottom_mirror = np.flipud(bottom_half)

        # Crop to equal size
        min_w = min(left_half.shape[1], right_mirror.shape[1])
        left_half = left_half[:, :min_w]
        right_mirror = right_mirror[:, :min_w]

        min_h = min(top_half.shape[0], bottom_mirror.shape[0])
        top_half = top_half[:min_h, :]
        bottom_mirror = bottom_mirror[:min_h, :]

        # Difference metrics
        horizontal_diff = np.mean(np.abs(left_half - right_mirror))
        vertical_diff = np.mean(np.abs(top_half - bottom_mirror))

        # Normalize
        horizontal_ratio = 1 - (horizontal_diff / 255)
        vertical_ratio = 1 - (vertical_diff / 255)

        horizontal_ratio = max(0, min(1, horizontal_ratio))
        vertical_ratio = max(0, min(1, vertical_ratio))

        return {
            "horizontal_ratio": float(horizontal_ratio),
            "vertical_ratio": float(vertical_ratio),
            "confidence": 1.0
        }
