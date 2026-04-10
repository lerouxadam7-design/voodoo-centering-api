import cv2
import numpy as np


# ============================================================
# CORNER ENGINE
# ============================================================

class VoodooCornerCloseupEngine:
    """
    Geometry-first corner scoring.

    Designed to be more stable on chrome/refractor cards by focusing on:
    - border line continuity
    - corner tip sharpness
    - mild whitening penalty

    and focusing less on raw texture/noise.
    """

    def __init__(self):
        self.min_patch_size = 28
        self.blur_floor = 18.0
        self.dark_mean_floor = 18.0
        self.bright_mean_ceiling = 245.0

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
    # Border finding helpers
    # --------------------------------------------------------

    def detect_top_border(self, gray):
        """
        Find the top border profile across x positions.
        Returns y positions for detected border points.
        """
        h, w = gray.shape
        search_h = max(12, int(h * 0.45))
        rows = []

        for x in range(int(w * 0.12), int(w * 0.88)):
            col = gray[:search_h, x].astype(np.float32)
            grad = np.abs(np.diff(col))
            if len(grad) == 0:
                continue

            y = int(np.argmax(grad))
            rows.append((x, y))

        return rows

    def detect_left_border(self, gray):
        """
        Find the left border profile across y positions.
        Returns x positions for detected border points.
        """
        h, w = gray.shape
        search_w = max(12, int(w * 0.45))
        cols = []

        for y in range(int(h * 0.12), int(h * 0.88)):
            row = gray[y, :search_w].astype(np.float32)
            grad = np.abs(np.diff(row))
            if len(grad) == 0:
                continue

            x = int(np.argmax(grad))
            cols.append((y, x))

        return cols

    # --------------------------------------------------------
    # Feature extraction
    # --------------------------------------------------------

    def extract_features(self, patch):
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        h, w = gray.shape[:2]
        radius = int(min(h, w) * 0.55)

        if radius < 12:
            return {
                "shape_score": 0.25,
                "continuity_score": 0.25,
                "whitening_penalty": 0.08,
                "confidence": 0.2,
            }

        gray_region = gray[:radius, :radius]

        # --------------------------
        # 1. Border detection
        # --------------------------
        top_pts = self.detect_top_border(gray_region)
        left_pts = self.detect_left_border(gray_region)

        if len(top_pts) < 8 or len(left_pts) < 8:
            return {
                "shape_score": 0.28,
                "continuity_score": 0.25,
                "whitening_penalty": 0.05,
                "confidence": 0.25,
            }

        top_y = np.array([p[1] for p in top_pts], dtype=np.float32)
        left_x = np.array([p[1] for p in left_pts], dtype=np.float32)

        # --------------------------
        # 2. Border continuity
        # --------------------------
        top_std = float(np.std(top_y))
        left_std = float(np.std(left_x))
        continuity_score = 1.0 - np.clip(((top_std + left_std) / 2.0) / 10.0, 0, 1)

        # --------------------------
        # 3. Tip sharpness
        # --------------------------
        top_y_med = float(np.median(top_y))
        left_x_med = float(np.median(left_x))

        tip_distance = np.sqrt((left_x_med ** 2) + (top_y_med ** 2))
        shape_score = 1.0 - np.clip(tip_distance / (radius * 0.65), 0, 1)
        shape_score = float(np.clip((shape_score * 0.75) + (continuity_score * 0.25), 0, 1))

        # --------------------------
        # 4. Whitening penalty
        # --------------------------
        border_band = max(3, int(radius * 0.12))
        top_band = gray_region[:border_band, :]
        left_band = gray_region[:, :border_band]
        border_pixels = np.concatenate([top_band.flatten(), left_band.flatten()])

        bright_mask = border_pixels > 228
        whitening_density = float(np.mean(bright_mask.astype(np.float32)))
        whitening_penalty = float(np.clip(whitening_density * 1.8, 0, 0.25))

        # --------------------------
        # 5. Confidence
        # --------------------------
        lap_var = float(cv2.Laplacian(gray_region, cv2.CV_64F).var())
        mean_val = float(np.mean(gray_region))

        conf_blur = np.clip(lap_var / 110.0, 0, 1)
        conf_light = np.clip((mean_val - 20.0) / 120.0, 0, 1)
        confidence = float(np.clip((conf_blur * 0.55) + (conf_light * 0.45), 0, 1))

        return {
            "shape_score": float(shape_score),
            "continuity_score": float(continuity_score),
            "whitening_penalty": float(whitening_penalty),
            "confidence": float(confidence),
        }

    # --------------------------------------------------------
    # Main patch analysis
    # --------------------------------------------------------

    def analyze_patch(self, patch, orientation="top_left"):
        if patch is None:
            return 0.5

        norm_patch = self.normalize_to_top_left(patch, orientation=orientation)

        h, w = norm_patch.shape[:2]
        longest = max(h, w)
        if longest > 320:
            scale = 320.0 / float(longest)
            norm_patch = cv2.resize(
                norm_patch,
                (max(1, int(w * scale)), max(1, int(h * scale))),
                interpolation=cv2.INTER_AREA
            )

        if not self.image_quality_ok(norm_patch):
            return 0.5

        feats = self.extract_features(norm_patch)

        score = (
            feats["shape_score"] * 0.58
            + feats["continuity_score"] * 0.34
            - feats["whitening_penalty"] * 0.12
        )

        score = score * (0.92 + 0.08 * feats["confidence"])

        if feats["confidence"] > 0.25:
            score = max(score, 0.18)

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
    # Helpers
    # ---------------------------------------------------------

    def _trimmed_mean(self, values, trim_frac=0.18):
        if not values:
            return None

        arr = np.array(values, dtype=np.float32)
        arr.sort()

        n = len(arr)
        trim_n = int(n * trim_frac)

        if trim_n > 0 and n > (2 * trim_n):
            arr = arr[trim_n:n - trim_n]

        if len(arr) == 0:
            return None

        return float(np.mean(arr))

    def _detect_left_border_points(self, gray):
        h, w = gray.shape
        x_max = max(14, int(w * 0.22))
        points = []

        for y in range(int(h * 0.18), int(h * 0.82)):
            row = gray[y, :x_max].astype(np.float32)
            if len(row) < 6:
                continue

            grad = np.abs(np.diff(row))
            x = int(np.argmax(grad))

            if grad[x] > 8:
                points.append(float(x))

        return points

    def _detect_right_border_points(self, gray):
        h, w = gray.shape
        x_min = min(w - 2, int(w * 0.78))
        points = []

        for y in range(int(h * 0.18), int(h * 0.82)):
            row = gray[y, x_min:w].astype(np.float32)
            if len(row) < 6:
                continue

            grad = np.abs(np.diff(row))
            rel_x = int(np.argmax(grad))
            x = x_min + rel_x

            if grad[rel_x] > 8:
                points.append(float(w - x))

        return points

    def _detect_top_border_points(self, gray):
        h, w = gray.shape
        y_max = max(14, int(h * 0.22))
        points = []

        for x in range(int(w * 0.18), int(w * 0.82)):
            col = gray[:y_max, x].astype(np.float32)
            if len(col) < 6:
                continue

            grad = np.abs(np.diff(col))
            y = int(np.argmax(grad))

            if grad[y] > 8:
                points.append(float(y))

        return points

    def _detect_bottom_border_points(self, gray):
        h, w = gray.shape
        y_min = min(h - 2, int(h * 0.78))
        points = []

        for x in range(int(w * 0.18), int(w * 0.82)):
            col = gray[y_min:h, x].astype(np.float32)
            if len(col) < 6:
                continue

            grad = np.abs(np.diff(col))
            rel_y = int(np.argmax(grad))
            y = y_min + rel_y

            if grad[rel_y] > 8:
                points.append(float(h - y))

        return points

    # ---------------------------------------------------------
    # Centering
    # ---------------------------------------------------------

    def compute_centering(self, card_img):
        gray = cv2.cvtColor(card_img, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        h, w = gray.shape

        left_distances = self._detect_left_border_points(gray)
        right_distances = self._detect_right_border_points(gray)
        top_distances = self._detect_top_border_points(gray)
        bottom_distances = self._detect_bottom_border_points(gray)

        if (
            len(left_distances) < 12 or
            len(right_distances) < 12 or
            len(top_distances) < 12 or
            len(bottom_distances) < 12
        ):
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "left_mean": None,
                "right_mean": None,
                "top_mean": None,
                "bottom_mean": None,
                "inner_left_x": None,
                "inner_right_x": None,
                "inner_top_y": None,
                "inner_bottom_y": None,
                "card_width": int(w),
                "card_height": int(h),
            }

        left_mean = self._trimmed_mean(left_distances)
        right_mean = self._trimmed_mean(right_distances)
        top_mean = self._trimmed_mean(top_distances)
        bottom_mean = self._trimmed_mean(bottom_distances)

        if (
            left_mean is None or right_mean is None or
            top_mean is None or bottom_mean is None
        ):
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "left_mean": None,
                "right_mean": None,
                "top_mean": None,
                "bottom_mean": None,
                "inner_left_x": None,
                "inner_right_x": None,
                "inner_top_y": None,
                "inner_bottom_y": None,
                "card_width": int(w),
                "card_height": int(h),
            }

        left_mean = max(1.0, left_mean)
        right_mean = max(1.0, right_mean)
        top_mean = max(1.0, top_mean)
        bottom_mean = max(1.0, bottom_mean)

        horizontal_ratio = min(left_mean, right_mean) / max(left_mean, right_mean)
        vertical_ratio = min(top_mean, bottom_mean) / max(top_mean, bottom_mean)

        inner_left_x = left_mean
        inner_right_x = float(w - right_mean)
        inner_top_y = top_mean
        inner_bottom_y = float(h - bottom_mean)

        return {
            "horizontal_ratio": float(horizontal_ratio),
            "vertical_ratio": float(vertical_ratio),
            "left_mean": round(float(left_mean), 2),
            "right_mean": round(float(right_mean), 2),
            "top_mean": round(float(top_mean), 2),
            "bottom_mean": round(float(bottom_mean), 2),
            "inner_left_x": round(float(inner_left_x), 2),
            "inner_right_x": round(float(inner_right_x), 2),
            "inner_top_y": round(float(inner_top_y), 2),
            "inner_bottom_y": round(float(inner_bottom_y), 2),
            "card_width": int(w),
            "card_height": int(h),
        }

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

        # worst + second-worst blend for stability
        final_score = (scores[0] * 0.65) + (scores[1] * 0.35)

        return float(np.clip(final_score, 0, 1))

    # ---------------------------------------------------------
    # Main analysis
    # ---------------------------------------------------------

    def analyze_array(self, image_array):
        target_width = 1000
        h, w = image_array.shape[:2]

        if w <= 0 or h <= 0:
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "edge_score": 0.5,
                "corner_score": 0.5,
                "confidence": 0.0,
                "inner_left_x": None,
                "inner_right_x": None,
                "inner_top_y": None,
                "inner_bottom_y": None,
                "image_width": None,
                "image_height": None,
            }

        scale = target_width / float(w)
        resized_h = int(h * scale)
        image = cv2.resize(image_array, (target_width, resized_h))

        bbox = self.detect_card_bbox(image)

        if bbox is None:
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "edge_score": 0.5,
                "corner_score": 0.5,
                "confidence": 0.0,
                "inner_left_x": None,
                "inner_right_x": None,
                "inner_top_y": None,
                "inner_bottom_y": None,
                "image_width": int(image.shape[1]),
                "image_height": int(image.shape[0]),
            }

        x, y, w2, h2 = bbox
        card = image[y:y+h2, x:x+w2]

        centering = self.compute_centering(card)
        edge_score = self.compute_edge_score(card)
        corner_score = self.compute_corner_score(card)

        inner_left_x = None
        inner_right_x = None
        inner_top_y = None
        inner_bottom_y = None

        if centering["inner_left_x"] is not None:
            inner_left_x = float(x) + float(centering["inner_left_x"])
        if centering["inner_right_x"] is not None:
            inner_right_x = float(x) + float(centering["inner_right_x"])
        if centering["inner_top_y"] is not None:
            inner_top_y = float(y) + float(centering["inner_top_y"])
        if centering["inner_bottom_y"] is not None:
            inner_bottom_y = float(y) + float(centering["inner_bottom_y"])

        return {
            "horizontal_ratio": round(float(centering["horizontal_ratio"]), 4),
            "vertical_ratio": round(float(centering["vertical_ratio"]), 4),
            "edge_score": round(float(edge_score), 4),
            "corner_score": round(float(corner_score), 4),
            "confidence": 1.0,

            "card_bbox_x": int(x),
            "card_bbox_y": int(y),
            "card_bbox_w": int(w2),
            "card_bbox_h": int(h2),

            "left_mean": centering["left_mean"],
            "right_mean": centering["right_mean"],
            "top_mean": centering["top_mean"],
            "bottom_mean": centering["bottom_mean"],

            "inner_left_x": None if inner_left_x is None else round(float(inner_left_x), 2),
            "inner_right_x": None if inner_right_x is None else round(float(inner_right_x), 2),
            "inner_top_y": None if inner_top_y is None else round(float(inner_top_y), 2),
            "inner_bottom_y": None if inner_bottom_y is None else round(float(inner_bottom_y), 2),

            "image_width": int(image.shape[1]),
            "image_height": int(image.shape[0]),
        }
