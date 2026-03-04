import cv2
import numpy as np


class VoodooRawCardCentering:

    def __init__(self):
        pass

    # -----------------------------
    # Order points
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
    # Warp card to rectangle
    # -----------------------------
    def warp(self, image, pts):

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
    # Detect card contour
    # -----------------------------
    def detect_card(self, image):

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        thresh = cv2.adaptiveThreshold(
            blur,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            51,
            5
        )

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        contours = sorted(contours, key=cv2.contourArea, reverse=True)

        image_area = image.shape[0] * image.shape[1]

        for cnt in contours:

            area = cv2.contourArea(cnt)

            if area < image_area * 0.40:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                return approx.reshape(4, 2)

        return None

    # -----------------------------
    # Detect white border thickness
    # -----------------------------
    def detect_white_border(self, gray):

        h, w = gray.shape

        blur = cv2.GaussianBlur(gray, (25, 25), 0)

        left = 0
        for x in range(w // 3):
            if np.mean(blur[:, x]) < 200:
                left = x
                break

        right = 0
        for x in range(w - 1, 2 * w // 3, -1):
            if np.mean(blur[:, x]) < 200:
                right = w - x
                break

        top = 0
        for y in range(h // 3):
            if np.mean(blur[y, :]) < 200:
                top = y
                break

        bottom = 0
        for y in range(h - 1, 2 * h // 3, -1):
            if np.mean(blur[y, :]) < 200:
                bottom = h - y
                break

        return left, right, top, bottom

    # -----------------------------
    # Main entry
    # -----------------------------
    def analyze_array(self, image_array):

        target_width = 1200
        h, w = image_array.shape[:2]
        scale = target_width / w
        image = cv2.resize(image_array, (target_width, int(h * scale)))

        pts = self.detect_card(image)

        if pts is None:
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "confidence": 0.0
            }

        warped = self.warp(image, pts)

        gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)

        left, right, top, bottom = self.detect_white_border(gray)

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
