import cv2
import numpy as np


class VoodooSlabCentering:

    def __init__(self):
        pass

    # -----------------------------
    # Order 4 points
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
    # Detect slab outer contour
    # -----------------------------
    def find_slab_contour(self, image):

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Strong blur to remove internal card noise
    blur = cv2.GaussianBlur(gray, (15, 15), 0)

    # OTSU threshold
    _, thresh = cv2.threshold(
        blur,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    contours, _ = cv2.findContours(
        thresh,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)

    peri = cv2.arcLength(largest, True)
    approx = cv2.approxPolyDP(largest, 0.02 * peri, True)

    if len(approx) == 4:
        return approx.reshape(4, 2)

    # Fallback: bounding rectangle
    x, y, w, h = cv2.boundingRect(largest)
    rect = np.array([
        [x, y],
        [x + w, y],
        [x + w, y + h],
        [x, y + h]
    ])

    return rect

    # -----------------------------
    # Compute centering after crop
    # -----------------------------
    def calculate_centering(self, cropped):

        h, w = cropped.shape[:2]

        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
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

        # Downscale for consistency
        target_width = 1200
        h, w = image_array.shape[:2]
        scale = target_width / w
        image = cv2.resize(image_array, (target_width, int(h * scale)))

        slab_pts = self.find_slab_contour(image)

        if slab_pts is None:
            return {
                "horizontal_ratio": 0.5,
                "vertical_ratio": 0.5,
                "confidence": 0.0
            }

        slab_warped = self.perspective_warp(image, slab_pts)

        # Crop inward to remove slab border (8% on each side)
        crop_pct = 0.08

        h2, w2 = slab_warped.shape[:2]

        x_start = int(w2 * crop_pct)
        x_end = int(w2 * (1 - crop_pct))
        y_start = int(h2 * crop_pct)
        y_end = int(h2 * (1 - crop_pct))

        cropped = slab_warped[y_start:y_end, x_start:x_end]

        h_ratio, v_ratio, confidence = self.calculate_centering(cropped)

        return {
            "horizontal_ratio": h_ratio,
            "vertical_ratio": v_ratio,
            "confidence": confidence
        }
