import cv2
import numpy as np


class ProfessionalCenteringEngineV68:

    def analyze_array(self, image_array):

        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            # Detect main object via adaptive threshold
            thresh = cv2.adaptiveThreshold(
                gray,
                255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY_INV,
                51,
                5
            )

            contours, _ = cv2.findContours(
                thresh,
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )

            if not contours:
                return {
                    "horizontal_ratio": 0.5,
                    "vertical_ratio": 0.5,
                    "error": "No contours"
                }

            largest = max(contours, key=cv2.contourArea)

            x, y, w, h = cv2.boundingRect(largest)

            img_h, img_w = gray.shape

            left_margin = x
            right_margin = img_w - (x + w)
            top_margin = y
            bottom_margin = img_h - (y + h)

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
