import cv2
import numpy as np


# ============================================================
# RAW FULL-CARD FEATURE ENGINE
# ============================================================

class VoodooRawEngine:

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # Detect dominant card bounding box (solid background assumed)
    # ---------------------------------------------------------
    def detect_card_bbox(self, image):

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

        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(
            thresh,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest)

        image_area = image.shape[0] * image.shape[1]
        if w * h < image_area * 0.30:
            return None

        return x, y, w, h

    # ---------------------------------------------------------
    # Predictive centering (symmetry-based)
    # ---------------------------------------------------------
    def compute_centering(self, card_img):

        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)

        h, w = gray.shape
        mx = int(w * 0.05)
        my = int(h * 0.05)

        gray = gray[my:h-my, mx:w-mx]
        blur = cv2.GaussianBlur(gray, (31, 31), 0)

        h2, w2 = blur.shape

        left = blur[:, :w2//2]
        right = np.fliplr(blur[:, w2//2:])

        top = blur[:h2//2, :]
        bottom = np.flipud(blur[h2//2:, :])

        min_w = min(left.shape[1], right.shape[1])
        left = left[:, :min_w]
        right = right[:, :min_w]

        min_h = min(top.shape[0], bottom.shape[0])
        top = top[:min_h, :]
        bottom = bottom[:min_h, :]

        h_diff = np.mean(np.abs(left - right))
        v_diff = np.mean(np.abs(top - bottom))

        h_ratio = 1 - (h_diff / 255)
        v_ratio = 1 - (v_diff / 255)

        return float(np.clip(h_ratio, 0, 1)), float(np.clip(v_ratio, 0, 1))

    # ---------------------------------------------------------
    # Edge integrity feature
    # ---------------------------------------------------------
    def compute_edge_score(self, card_img):

        h, w = card_img.shape[:2]
        strip = int(min(h, w) * 0.05)

        strips = [
            card_img[0:strip, :],
            card_img[h-strip:h, :],
            card_img[:, 0:strip],
            card_img[:, w-strip:w]
        ]

        scores = []

        for s in strips:
            gray = cv2.cvtColor(s, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (9, 9), 0)

            variance = np.var(blur)
            edges = cv2.Canny(blur, 50, 150)
            edge_density = np.mean(edges)

            edge_norm = edge_density / 255
            var_norm = variance / (255**2)

            score = (edge_norm * 0.7) - (var_norm * 0.3)
            scores.append(score)

        normalized = np.clip(np.mean(scores), 0, 1)
        return float(normalized)

    # ---------------------------------------------------------
    # Main RAW card analysis
    # ---------------------------------------------------------
    def analyze_array(self, image_array):

        target_width = 1000
        h, w = image_array.shape[:2]
        scale = target_width / w
        image = cv2.resize(image_array, (target_width, int(h * scale)))

        bbox = self.detect_card_bbox(image)

        if bbox is None:
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "edge_score": 0.5,
                "confidence": 0.0
            }

        x, y, w2, h2 = bbox
        card = image[y:y+h2, x:x+w2]

        h_ratio, v_ratio = self.compute_centering(card)
        edge_score = self.compute_edge_score(card)

        return {
            "horizontal_ratio": h_ratio,
            "vertical_ratio": v_ratio,
            "edge_score": edge_score,
            "confidence": 1.0
        }


# ============================================================
# CLOSE-UP CORNER ENGINE
# ============================================================

class VoodooCornerCloseupEngine:

    def __init__(self):
        pass

    def analyze_array(self, image_array):

        target_width = 600
        h, w = image_array.shape[:2]
        scale = target_width / w
        image = cv2.resize(image_array, (target_width, int(h * scale)))

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (7, 7), 0)

        edges = cv2.Canny(blur, 50, 150)

        contours, _ = cv2.findContours(
            edges,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        if not contours:
            return {"corner_score": 0.5, "confidence": 0.0}

        largest = max(contours, key=cv2.contourArea)

        # Fit polygon to contour
        peri = cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, 0.01 * peri, True)

        if len(approx) < 2:
            return {"corner_score": 0.5, "confidence": 0.0}

        # Estimate curvature by analyzing contour smoothness
        curvature_values = []

        for i in range(1, len(largest) - 1):
            p_prev = largest[i - 1][0]
            p_curr = largest[i][0]
            p_next = largest[i + 1][0]

            v1 = p_curr - p_prev
            v2 = p_next - p_curr

            angle = np.arccos(
                np.clip(
                    np.dot(v1, v2) /
                    (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-6),
                    -1,
                    1
                )
            )

            curvature_values.append(angle)

        avg_curvature = np.mean(curvature_values)

        # Rounded corners have lower curvature concentration
        # Sharp corners have high curvature spike

        corner_score = np.clip(avg_curvature / np.pi, 0, 1)

        return {
            "corner_score": float(corner_score),
            "confidence": 1.0
        }
            "corner_score": float(final_score),
            "confidence": 1.0
        }
