import cv2
import numpy as np


# ============================================================
# RAW FULL CARD FEATURE ENGINE
# ============================================================

class VoodooRawEngine:

    def __init__(self):
        pass

    # ---------------------------------------------------------
    # Detect dominant card bounding box
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
    # BORDER-BASED CENTERING
    # ---------------------------------------------------------
    def compute_centering(self, card_img):

        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)

        h, w = edges.shape

        left_distances = []
        right_distances = []

        for y in range(int(h * 0.2), int(h * 0.8)):
            row = edges[y, :]
            indices = np.where(row > 0)[0]
            if len(indices) > 0:
                left_distances.append(indices[0])
                right_distances.append(w - indices[-1])

        top_distances = []
        bottom_distances = []

        for x in range(int(w * 0.2), int(w * 0.8)):
            col = edges[:, x]
            indices = np.where(col > 0)[0]
            if len(indices) > 0:
                top_distances.append(indices[0])
                bottom_distances.append(h - indices[-1])

        if not left_distances or not top_distances:
            return 0.5, 0.5

        left_mean = np.mean(left_distances)
        right_mean = np.mean(right_distances)
        top_mean = np.mean(top_distances)
        bottom_mean = np.mean(bottom_distances)

        horizontal_ratio = min(left_mean, right_mean) / max(left_mean, right_mean)
        vertical_ratio = min(top_mean, bottom_mean) / max(top_mean, bottom_mean)

        return float(horizontal_ratio), float(vertical_ratio)

    # ---------------------------------------------------------
    # EDGE INTEGRITY FEATURE
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
            edges = cv2.Canny(blur, 75, 200)
            edge_density = np.mean(edges)

            edge_norm = edge_density / 255
            var_norm = variance / (255**2)

            score = (edge_norm * 0.7) - (var_norm * 0.3)
            scores.append(score)

        normalized = np.clip(np.mean(scores), 0, 1)
        return float(normalized)

    # ---------------------------------------------------------
    # SURFACE SMOOTHNESS FEATURE (NEW)
    # ---------------------------------------------------------
    def compute_surface_score(self, card_img):

        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        grad_x = cv2.Sobel(blur, cv2.CV_64F, 1, 0, ksize=3)
        grad_y = cv2.Sobel(blur, cv2.CV_64F, 0, 1, ksize=3)
        magnitude = np.sqrt(grad_x**2 + grad_y**2)

        noise = np.var(magnitude)

        # Empirical normalization
        surface_score = 1 - min(noise / 5000, 1)

        return float(np.clip(surface_score, 0, 1))

    # ---------------------------------------------------------
    # MAIN RAW ANALYSIS
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
                "surface_score": 0.5,
                "confidence": 0.0
            }

        x, y, w2, h2 = bbox
        card = image[y:y+h2, x:x+w2]

        h_ratio, v_ratio = self.compute_centering(card)
        edge_score = self.compute_edge_score(card)
        surface_score = self.compute_surface_score(card)

        return {
            "horizontal_ratio": h_ratio,
            "vertical_ratio": v_ratio,
            "edge_score": edge_score,
            "surface_score": surface_score,
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
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        edges = cv2.Canny(blur, 75, 200)

        h2, w2 = edges.shape
        radius = int(min(h2, w2) * 0.25)

        y_idx, x_idx = np.ogrid[:radius, :radius]
        mask = (x_idx**2 + y_idx**2) <= radius**2

        region = edges[0:radius, 0:radius]
        masked_edges = region[mask]

        if masked_edges.size == 0:
            return {"corner_score": 0.5, "confidence": 0.0}

        edge_density = np.mean(masked_edges) / 255

        ys, xs = np.where(region > 0)
        if len(xs) == 0:
            return {"corner_score": 0.5, "confidence": 0.0}

        distances = np.sqrt(xs**2 + ys**2)
        avg_distance = np.mean(distances) / radius

        score = (edge_density ** 0.5) * (1 - avg_distance)
        score = score * 2

        return {
            "corner_score": float(np.clip(score, 0, 1)),
            "confidence": 1.0
        }
