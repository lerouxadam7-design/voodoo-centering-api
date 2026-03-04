import cv2
import numpy as np


class VoodooRawEngine:

    def __init__(self):
        pass

    # -------------------------------------------------
    # Detect dominant card region (solid background)
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Predictive symmetry centering
    # -------------------------------------------------
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

    # -------------------------------------------------
    # Corner sharpness metric
    # -------------------------------------------------
    def compute_corner_score(self, card_img):

        h, w = card_img.shape[:2]
        patch = int(min(h, w) * 0.12)

        corners = [
            card_img[0:patch, 0:patch],
            card_img[0:patch, w-patch:w],
            card_img[h-patch:h, w-patch:w],
            card_img[h-patch:h, 0:patch]
        ]

        scores = []

        for c in corners:
            gray = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (15, 15), 0)

            grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)

            magnitude = np.sqrt(grad_x**2 + grad_y**2)
            scores.append(np.mean(magnitude))

        score = np.mean(scores)
        return float(np.clip(score / 100, 0, 1))

    # -------------------------------------------------
    # Edge integrity metric
    # -------------------------------------------------
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

            score = edge_density - (variance * 0.01)
            scores.append(score)

        normalized = np.clip(np.mean(scores) / 255, 0, 1)
        return float(normalized)

    # -------------------------------------------------
    # Main entry
    # -------------------------------------------------
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
                "corner_score": 0.5,
                "edge_score": 0.5,
                "confidence": 0.0
            }

        x, y, w2, h2 = bbox
        card = image[y:y+h2, x:x+w2]

        h_ratio, v_ratio = self.compute_centering(card)
        corner_score = self.compute_corner_score(card)
        edge_score = self.compute_edge_score(card)

        return {
            "horizontal_ratio": h_ratio,
            "vertical_ratio": v_ratio,
            "corner_score": corner_score,
            "edge_score": edge_score,
            "confidence": 1.0
        }
