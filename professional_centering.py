import cv2
import numpy as np


class ProfessionalCenteringEngineV68:

    def detect_card(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 75, 200)
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            raise ValueError("No contours detected")

        largest = max(contours, key=cv2.contourArea)
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

        if len(approx) != 4:
            raise ValueError("Card detection failed")

        pts = approx.reshape(4, 2).astype("float32")
        return self.order_points(pts)

    def warp_card(self, image, pts):
        (tl, tr, br, bl) = pts

        widthA = np.linalg.norm(br - bl)
        widthB = np.linalg.norm(tr - tl)
        maxWidth = int(max(widthA, widthB))

        heightA = np.linalg.norm(tr - br)
        heightB = np.linalg.norm(tl - bl)
        maxHeight = int(max(heightA, heightB))

        dst = np.array([
            [0, 0],
            [maxWidth - 1, 0],
            [maxWidth - 1, maxHeight - 1],
            [0, maxHeight - 1]
        ], dtype="float32")

        M = cv2.getPerspectiveTransform(pts, dst)
        return cv2.warpPerspective(image, M, (maxWidth, maxHeight))

    def detect_borders(self, warped):
        h, w, _ = warped.shape
        band_w = int(w * 0.15)
        band_h = int(h * 0.15)

        bands = [
            warped[:, :band_w],
            warped[:, w - band_w:],
            warped[:band_h, :],
            warped[h - band_h:, :]
        ]

        borders = []

        for idx, band in enumerate(bands):
            gray = cv2.cvtColor(band, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 50, 150)

            lines = cv2.HoughLinesP(
                edges,
                1,
                np.pi / 180,
                threshold=80,
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

    def calculate_centering(self, warped, borders):
        if None in borders:
            return None

        h, w, _ = warped.shape
        left, right, top, bottom = borders

        left_border = left
        right_border = w - right
        top_border = top
        bottom_border = h - bottom

        h_ratio = min(left_border, right_border) / max(left_border, right_border)
        v_ratio = min(top_border, bottom_border) / max(top_border, bottom_border)

        return h_ratio, v_ratio

    def analyze_array(self, image_array):

        pts = self.detect_card(image_array)
        warped = self.warp_card(image_array, pts)

        borders = self.detect_borders(warped)
        ratios = self.calculate_centering(warped, borders)

        if ratios is None:
            return {"error": "Border detection failed"}

        h_ratio, v_ratio = ratios

        return {
            "horizontal_ratio": float(h_ratio),
            "vertical_ratio": float(v_ratio)
        }

    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
