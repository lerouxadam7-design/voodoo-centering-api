import cv2
import numpy as np


class VoodooCornerEngine:

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
    # Warp card
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
    # Detect card contour (solid background assumed)
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
            if cv2.contourArea(cnt) < image_area * 0.40:
                continue

            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt, 0.02 * peri, True)

            if len(approx) == 4:
                return approx.reshape(4, 2)

        return None

    # -----------------------------
    # Corner integrity analysis
    # -----------------------------
    def analyze_corners(self, warped):

        h, w = warped.shape[:2]

        # Define 4 corner patches
        patch_size = int(min(h, w) * 0.12)

        patches = [
            warped[0:patch_size, 0:patch_size],                  # TL
            warped[0:patch_size, w-patch_size:w],                # TR
            warped[h-patch_size:h, w-patch_size:w],              # BR
            warped[h-patch_size:h, 0:patch_size]                 # BL
        ]

        scores = []

        for patch in patches:

            gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

            # Edge strength
            edges = cv2.Canny(gray, 50, 150)
            edge_strength = np.mean(edges)

            # Whitening detection (high brightness variance)
            brightness_var = np.var(gray)

            # Combine metrics
            score = edge_strength - (brightness_var * 0.02)

            scores.append(score)

        # Normalize
        scores = np.array(scores)
        norm_score = np.clip(np.mean(scores) / 255, 0, 1)

        return float(norm_score)

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
            return {"corner_score": 0.5, "confidence": 0.0}

        warped = self.warp(image, pts)

        corner_score = self.analyze_corners(warped)

        return {
            "corner_score": corner_score,
            "confidence": 1.0
        }
