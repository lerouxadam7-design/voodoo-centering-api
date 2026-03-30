import cv2
import numpy as np


# ============================================================
# CORNER ENGINE
# ============================================================

class VoodooCornerCloseupEngine:
    """
    Corner scoring engine for close-up corner images.

    Goals:
    - Normalize all uploaded corners to the same orientation
    - Reject clearly unusable images
    - Blend shape, whitening, and roughness signals
    - Avoid collapsing usable corners to 0.0 too easily
    """

    def __init__(self):
        self.min_patch_size = 24
        self.blur_floor = 20.0
        self.dark_mean_floor = 25.0
        self.bright_mean_ceiling = 240.0

    # --------------------------------------------------------
    # Quality checks
    # --------------------------------------------------------

    def image_quality_ok(self, patch):
        if patch is None:
            return False

        h, w = patch.shape[:2]
        if h < self.min_patch_size or w < self.min_patch_size:
            return False

        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        mean_val = float(np.mean(gray))
        if mean_val < self.dark_mean_floor or mean_val > self.bright_mean_ceiling:
            return False

        blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        if blur_score < self.blur_floor:
            return False

        return True

    # --------------------------------------------------------
    # Orientation normalization
    # --------------------------------------------------------

    def normalize_to_top_left(self, patch, orientation="top_left"):
        """
        Rotate/flip the patch so every corner is scored as if it were
        a top-left corner.
        """
        if orientation == "top_left":
            return patch
        if orientation == "top_right":
            return cv2.rotate(patch, cv2.ROTATE_90_COUNTERCLOCKWISE)
        if orientation == "bottom_left":
            return cv2.rotate(patch, cv2.ROTATE_90_CLOCKWISE)
        if orientation == "bottom_right":
            return cv2.rotate(patch, cv2.ROTATE_180)

        return patch

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    def extract_features(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 75, 200)

        h, w = gray.shape[:2]
        radius = int(min(h, w) * 0.42)

        if radius <= 6:
            return {
                "shape_score": 0.0,
                "whitening_penalty": 0.3,
                "roughness_penalty": 0.2,
                "confidence": 0.0,
            }

        gray_region = gray[:radius, :radius]
        edge_region = edges[:radius, :radius]

        # --------------------------
        # 1. Shape / concentration
        # --------------------------
        ys, xs = np.where(edge_region > 0)

        if len(xs) == 0:
            shape_score = 0.0
        else:
            edge_density = float(np.mean(edge_region) / 255.0)
            distances = np.sqrt(xs**2 + ys**2)
            avg_distance = float(np.mean(distances) / max(radius, 1))

            shape_score = (edge_density ** 0.5) * (1.0 - avg_distance)
            shape_score = float(np.clip(shape_score * 2.5, 0, 1))

        # --------------------------
        # 2. Whitening / chipping
        # --------------------------
        border_band = max(3, int(radius * 0.18))

        top_band = gray_region[:border_band, :]
        left_band = gray_region[:, :border_band]

        border_pixels = np.concatenate([
            top_band.flatten(),
            left_band.flatten()
        ])

        bright_mask = border_pixels > 220
        whitening_density = float(np.mean(bright_mask.astype(np.float32)))
        whitening_penalty = float(np.clip(whitening_density * 3.5, 0, 1))

        # --------------------------
        # 3. Roughness / fray
        # --------------------------
        roughness = float(np.var(edge_region.astype(np.float32) / 255.0))
        roughness_penalty = float(np.clip(roughness * 7.0, 0, 1))

        # --------------------------
        # 4. Confidence
        # --------------------------
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        mean_val = float(np.mean(gray))

        conf_blur = np.clip(lap_var / 120.0, 0, 1)
        conf_light = np.clip((mean_val - 25.0) / 140.0, 0, 1)
        confidence = float(np.clip((conf_blur * 0.6) + (conf_light * 0.4), 0, 1))

        return {
            "shape_score": shape_score,
            "whitening_penalty": whitening_penalty,
            "roughness_penalty": roughness_penalty,
            "confidence": confidence,
        }

    # --------------------------------------------------------
    # Main close-up corner scoring
    # --------------------------------------------------------

    def analyze_patch(self, patch, orientation="top_left"):
        if patch is None:
            return 0.5

        norm_patch = self.normalize_to_top_left(patch, orientation=orientation)

        h, w = norm_patch.shape[:2]
        longest = max(h, w)
        if longest > 300:
            scale = 300.0 / float(longest)
            norm_patch = cv2.resize(
                norm_patch,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA,
            )

        if not self.image_quality_ok(norm_patch):
            return 0.5

        feats = self.extract_features(norm_patch)

        score = (
            feats["shape_score"] * 0.82
            - feats["whitening_penalty"] * 0.12
            - feats["roughness_penalty"] * 0.04
        )

        # lighter confidence damping
        score = score * (0.90 + 0.10 * feats["confidence"])

        # keep usable corners from collapsing to zero
        if feats["confidence"] > 0.35:
            score = max(score, 0.03)

        return float(np.clip(score, 0, 1))


# ============================================================
# RAW ENGINE
# ============================================================

class VoodooRawEngine:
    def __init__(self):
        self.corner_engine = VoodooCornerCloseupEngine()

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
    # Centering
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
    # Edge feature
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

            edge_norm = edge_density / 255.0
            var_norm = variance / (255.0 ** 2)

            score = (edge_norm * 0.7) - (var_norm * 0.3)
            scores.append(score)

        return float(np.clip(np.mean(scores), 0, 1))

    # ---------------------------------------------------------
    # Auto 4-corner extraction
    # ---------------------------------------------------------

    def compute_corner_score(self, card_img):
        h, w = card_img.shape[:2]
        patch_size = int(min(h, w) * 0.25)

        patches = [
            ("top_left", card_img[0:patch_size, 0:patch_size]),
            ("top_right", card_img[0:patch_size, w-patch_size:w]),
            ("bottom_left", card_img[h-patch_size:h, 0:patch_size]),
            ("bottom_right", card_img[h-patch_size:h, w-patch_size:w]),
        ]

        scores = []
        for orientation, patch in patches:
            score = self.corner_engine.analyze_patch(patch, orientation=orientation)
            scores.append(score)

        scores = sorted(scores)

        if len(scores) == 0:
            return 0.5
        if len(scores) == 1:
            return float(scores[0])

        # Worst + second-worst blend for stability
        final_score = (scores[0] * 0.70) + (scores[1] * 0.30)

        return float(np.clip(final_score, 0, 1))

    # ---------------------------------------------------------
    # Main analysis
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
                "corner_score": 0.5,
                "confidence": 0.0
            }

        x, y, w2, h2 = bbox
        card = image[y:y+h2, x:x+w2]

        h_ratio, v_ratio = self.compute_centering(card)
        edge_score = self.compute_edge_score(card)
        corner_score = self.compute_corner_score(card)

        return {
            "horizontal_ratio": h_ratio,
            "vertical_ratio": v_ratio,
            "edge_score": edge_score,
            "corner_score": corner_score,
            "confidence": 1.0
        }
