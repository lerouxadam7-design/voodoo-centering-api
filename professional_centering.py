import cv2
import numpy as np


class ProfessionalCenteringEngineV68:

    def analyze_array(self, image_array):

        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            # threshold for card region
            _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

            coords = np.column_stack(np.where(thresh > 0))

            if coords.size == 0:
                return {
                    "horizontal_ratio": 0.5,
                    "vertical_ratio": 0.5,
                    "error": "Card not detected"
                }

            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)

            h, w = gray.shape

            left_margin = x_min
            right_margin = w - x_max
            top_margin = y_min
            bottom_margin = h - y_max

            if max(left_margin, right_margin) == 0 or max(top_margin, bottom_margin) == 0:
                return {
                    "horizontal_ratio": 0.5,
                    "vertical_ratio": 0.5,
                    "error": "Invalid margins"
                }

            h_ratio = min(left_margin, right_margin) / max(left_margin, right_margin)
            v_ratio = min(top_margin, bottom_margin) / max(top_margin, bottom_margin)

            return {
                "horizontal_ratio": float(h_ratio),
                "vertical_ratio": float(v_ratio)
            }

        except Exception as e:
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "error": str(e)
            }
