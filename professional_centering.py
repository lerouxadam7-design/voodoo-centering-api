import cv2
import numpy as np


class VoodooSlabCentering:

    def __init__(self):
        pass

    # -----------------------------
    # Order 4 points consistently
    # -----------------------------
    def order_points(self, pts):
        rect = np.zeros((4, 2), dtype="float32")

        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]

        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]

        return rect

    # -----------------------------
    # Perspective warp
    # -----------------------------
    def perspective_warp(self, image, pts):

        rect = self.order_points(pts)
        (tl, tr, br, bl) = rect

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

        M = cv2.getPerspectiveTransform(rect, dst)
        warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))

        return warped

    # -----------------------------
    # Find inner card contour (slab-first)
    # -----------------------------
    def find_card_contour(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        thresh = cv2.adaptiveThreshold(
            blur,
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

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        h, w = image.shape[:2]
        image_area = h * w

        for cnt in contours:

            area = cv2.contourArea(cnt)

            # Skip slab frame (too large)
            if area > image_area * 0.90:
                continue

            if area < image_area * 0.10:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:

                x, y, cw, ch = cv2.boundingRect(approx)
                aspect_ratio = ch / float(cw)

                # Sports card vertical ratio range
                if 1.2 < aspect_ratio < 1.7:
                    return approx.reshape(4, 2)

        return None

    # -----------------------------
    # Calculate centering ratios
    # -----------------------------
    def calculate_centering(self, warped):

        h, w = warped.shape[:2]

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        coords = np.column_stack(np.where(edges > 0))

        if coords.size == 0:
            return 0.5, 0.5, 0.0

        y_min, x_min = coords.min(axis=0)
        y_max, x_max = coords.max(axis=0)

        left_margin = x_min
        right_margin = w - x_max
        top_margin = y_min
        bottom_margin = h - y_max

        if max(left_margin, right_margin) == 0 or max(top_margin, bottom_margin) == 0:
            return 0.5, 0.5, 0.0

        h_ratio = min(left_margin, right_margin) / max(left_margin, right_margin)
        v_ratio = min(top_margin, bottom_margin) / max(top_margin, bottom_margin)

        return float(h_ratio), float(v_ratio), 1.0

    # -----------------------------
    # Main entry
    # -----------------------------
    def analyze_array(self, image_array):

        # Downscale for performance & stability
        target_width = 1200
        h, w = image_array.shape[:2]
        scale = target_width / w
        image = cv2.resize(image_array, (target_width, int(h * scale)))

        pts = self.find_card_contour(image)

        if pts is None:
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "confidence": 0.0
            }

        warped = self.perspective_warp(image, pts)

        h_ratio, v_ratio, confidence = self.calculate_centering(warped)

        return {
            "horizontal_ratio": h_ratio,
            "vertical_ratio": v_ratio,
            "confidence": confidence
        }
     
