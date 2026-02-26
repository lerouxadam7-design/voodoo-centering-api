import cv2
import numpy as np


class ProfessionalCenteringEngineV68:

    def isolate_card_region(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # threshold bright regions (card surface)
        _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY)

        coords = np.column_stack(np.where(thresh > 0))

        if coords.size == 0:
            return image  # fallback to full image

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        cropped = image[y_min:y_max, x_min:x_max]

        return cropped

    def detect_borders(self, image):
        h, w, _ = image.shape

        band_w = int(w * 0.15)
        band_h = int(h * 0.15)

        bands = [
            image[:, :band_w],            # left
            image[:, w - band_w:],        # right
            image[:band_h, :],            # top
            image[h - band_h:, :]         # bottom
        ]

        borders = []

        for idx, band in enumerate(bands):
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=60,
                minLineLength=int((h if idx < 2 else w) * 0.6),
                maxLineGap=10
            )

            if lines is None:
                borders.append(None)
                continue

            lines = lines[:, 0, :]
            best_line = max(lines, key=lambda l: abs(l[1] - l[3]))

            if idx == 0:
                borders.append(best_line[0])
            elif idx == 1:
                borders.append(w - band_w + best_line[0])
            elif idx == 2:
                borders.append(best_line[1])
            else:
                borders.append(h - band_h + best_line[1])

        return borders

    def calculate_centering(self, image, borders):
        if None in borders:
            return None

        h, w, _ = image.shape
        left, right, top, bottom = borders

        left_border = left
        right_border = w - right
        top_border = top
        bottom_border = h - bottom

        if max(left_border, right_border) == 0 or max(top_border, bottom_border) == 0:
            return None

        h_ratio = min(left_border, right_border) / max(left_border, right_border)
        v_ratio = min(top_border, bottom_border) / max(top_border, bottom_border)

        return h_ratio, v_ratio

    def analyze_array(self, image_array):
        try:
            card = self.isolate_card_region(image_array)

            borders = self.detect_borders(card)
            ratios = self.calculate_centering(card, borders)

            if ratios is None:
                return {
                    "horizontal_ratio": 0.5,
                    "vertical_ratio": 0.5,
                    "error": "Border detection failed"
                }

            h_ratio, v_ratio = ratios

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
