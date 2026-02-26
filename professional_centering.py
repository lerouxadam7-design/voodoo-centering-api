import cv2
import numpy as np


class ProfessionalCenteringEngineV68:

    def analyze_array(self, image_array):

        try:
            gray = cv2.cvtColor(image_array, cv2.COLOR_BGR2GRAY)

            height, width = gray.shape

            left_half = gray[:, :width // 2]
            right_half = gray[:, width // 2:]
            right_half_flipped = np.fliplr(right_half)

            min_width = min(left_half.shape[1], right_half_flipped.shape[1])
            left_half = left_half[:, :min_width]
            right_half_flipped = right_half_flipped[:, :min_width]

            diff = np.mean(np.abs(left_half - right_half_flipped))
            normalized_diff = diff / 255.0

            symmetry_score = 1.0 - normalized_diff

            # Clamp safely
            symmetry_score = max(0.0, min(symmetry_score, 1.0))

            return {
                "horizontal_ratio": float(symmetry_score),
                "vertical_ratio": float(symmetry_score)
            }

        except Exception as e:
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "error": str(e)
            }
