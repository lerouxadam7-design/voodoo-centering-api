from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

from professional_centering import VoodooRawEngine, VoodooCornerCloseupEngine

app = FastAPI()

raw_engine = VoodooRawEngine()
corner_engine = VoodooCornerCloseupEngine()


def decode_image(contents: bytes, max_dim: int = 1200):
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return None

    h, w = image.shape[:2]
    longest = max(h, w)

    if longest > max_dim:
        scale = max_dim / float(longest)
        new_w = max(1, int(w * scale))
        new_h = max(1, int(h * scale))
        image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    return image


def to_float_or_none(value):
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def find_first_key(d: dict, keys):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def enrich_centering_coordinates(result: dict, image) -> dict:
    """
    Best-effort coordinate extraction from the raw centering engine result.

    This does NOT assume one exact output schema from VoodooRawEngine.
    It checks several likely key names and returns standardized keys:
      - inner_left_x
      - inner_right_x
      - inner_top_y
      - inner_bottom_y

    If the engine does not expose coordinates yet, these remain None.
    """
    if not isinstance(result, dict):
        return result

    h, w = image.shape[:2]

    left_x = find_first_key(result, [
        "inner_left_x", "left_x", "card_left_x", "left_border_x", "detected_left_x"
    ])
    right_x = find_first_key(result, [
        "inner_right_x", "right_x", "card_right_x", "right_border_x", "detected_right_x"
    ])
    top_y = find_first_key(result, [
        "inner_top_y", "top_y", "card_top_y", "top_border_y", "detected_top_y"
    ])
    bottom_y = find_first_key(result, [
        "inner_bottom_y", "bottom_y", "card_bottom_y", "bottom_border_y", "detected_bottom_y"
    ])

    left_margin = find_first_key(result, [
        "left_margin_px", "left_margin", "left_border_px", "left_border"
    ])
    right_margin = find_first_key(result, [
        "right_margin_px", "right_margin", "right_border_px", "right_border"
    ])
    top_margin = find_first_key(result, [
        "top_margin_px", "top_margin", "top_border_px", "top_border"
    ])
    bottom_margin = find_first_key(result, [
        "bottom_margin_px", "bottom_margin", "bottom_border_px", "bottom_border"
    ])

    left_x = to_float_or_none(left_x)
    right_x = to_float_or_none(right_x)
    top_y = to_float_or_none(top_y)
    bottom_y = to_float_or_none(bottom_y)

    left_margin = to_float_or_none(left_margin)
    right_margin = to_float_or_none(right_margin)
    top_margin = to_float_or_none(top_margin)
    bottom_margin = to_float_or_none(bottom_margin)

    # If engine gave margins instead of x/y coordinates, convert them.
    if left_x is None and left_margin is not None:
        left_x = left_margin
    if right_x is None and right_margin is not None:
        right_x = w - right_margin
    if top_y is None and top_margin is not None:
        top_y = top_margin
    if bottom_y is None and bottom_margin is not None:
        bottom_y = h - bottom_margin

    # Basic validation.
    if left_x is not None and not (0 <= left_x <= w):
        left_x = None
    if right_x is not None and not (0 <= right_x <= w):
        right_x = None
    if top_y is not None and not (0 <= top_y <= h):
        top_y = None
    if bottom_y is not None and not (0 <= bottom_y <= h):
        bottom_y = None

    if left_x is not None and right_x is not None and right_x <= left_x:
        left_x, right_x = None, None
    if top_y is not None and bottom_y is not None and bottom_y <= top_y:
        top_y, bottom_y = None, None

    result["inner_left_x"] = None if left_x is None else round(float(left_x), 2)
    result["inner_right_x"] = None if right_x is None else round(float(right_x), 2)
    result["inner_top_y"] = None if top_y is None else round(float(top_y), 2)
    result["inner_bottom_y"] = None if bottom_y is None else round(float(bottom_y), 2)

    result["image_width"] = int(w)
    result["image_height"] = int(h)

    return result


def crop_surface_roi(image, border_frac: float = 0.06):
    h, w = image.shape[:2]
    x1 = int(w * border_frac)
    x2 = int(w * (1.0 - border_frac))
    y1 = int(h * border_frac)
    y2 = int(h * (1.0 - border_frac))

    if x2 <= x1 or y2 <= y1:
        return image

    return image[y1:y2, x1:x2]


def normalize_surface_image(image):
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)

    lab = cv2.merge([l, a, b])
    image = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return image


def local_variance(gray, ksize: int = 11):
    gray_f = gray.astype(np.float32)
    mean = cv2.blur(gray_f, (ksize, ksize))
    sq_mean = cv2.blur(gray_f * gray_f, (ksize, ksize))
    var = sq_mean - (mean * mean)
    return np.maximum(var, 0.0)


def detect_glare_mask(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]
    var = local_variance(gray, ksize=11)

    bright_mask = val >= 235
    low_sat_mask = sat <= 40
    low_texture_mask = var <= 40.0

    glare_mask = (bright_mask & low_sat_mask & low_texture_mask).astype(np.uint8) * 255

    kernel5 = np.ones((5, 5), np.uint8)
    kernel7 = np.ones((7, 7), np.uint8)

    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_CLOSE, kernel7)
    glare_mask = cv2.morphologyEx(glare_mask, cv2.MORPH_OPEN, kernel5)

    return glare_mask


def build_valid_surface_mask(image, glare_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    non_shadow = (gray > 28).astype(np.uint8) * 255
    non_glare = cv2.bitwise_not(glare_mask)

    valid_mask = cv2.bitwise_and(non_shadow, non_glare)

    kernel3 = np.ones((3, 3), np.uint8)
    valid_mask = cv2.morphologyEx(valid_mask, cv2.MORPH_OPEN, kernel3)

    return valid_mask


def compute_scratch_score(image, valid_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 70, 160)

    edges_valid = cv2.bitwise_and(edges, valid_mask)

    horiz_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 1))
    vert_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 9))

    horiz_lines = cv2.morphologyEx(edges_valid, cv2.MORPH_OPEN, horiz_kernel)
    vert_lines = cv2.morphologyEx(edges_valid, cv2.MORPH_OPEN, vert_kernel)

    line_map = cv2.max(horiz_lines, vert_lines)

    valid_pixels = max(1, np.count_nonzero(valid_mask))
    line_pixels = np.count_nonzero(line_map)

    density = line_pixels / float(valid_pixels)
    mean_strength = np.mean(line_map) / 255.0

    raw = (0.65 * density) + (0.35 * mean_strength)

    return float(np.clip(raw * 3.5, 0.0, 1.0))


def compute_speckle_score(image, valid_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, blur)

    _, blob_map = cv2.threshold(diff, 21, 255, cv2.THRESH_BINARY)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(blob_map, connectivity=8)
    filtered = np.zeros_like(blob_map)

    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if 1 <= area <= 16:
            filtered[labels == i] = 255

    filtered = cv2.bitwise_and(filtered, valid_mask)

    valid_pixels = max(1, np.count_nonzero(valid_mask))
    blob_pixels = np.count_nonzero(filtered)

    density = blob_pixels / float(valid_pixels)
    raw = density * 6.1

    return float(np.clip(raw, 0.0, 1.0))


def compute_gloss_score(image, glare_mask):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    glare_fraction = np.count_nonzero(glare_mask) / float(glare_mask.size)

    glare_strength = 0.0
    if np.count_nonzero(glare_mask) > 0:
        glare_strength = np.mean(gray[glare_mask > 0]) / 255.0

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(glare_mask, connectivity=8)
    largest_area = 0
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        largest_area = max(largest_area, area)

    spread_score = largest_area / float(glare_mask.size)

    raw = (
        0.45 * glare_fraction +
        0.35 * glare_strength +
        0.20 * spread_score
    )

    raw *= 0.79

    return float(np.clip(raw, 0.0, 1.0))


def analyze_surface(image):
    norm = normalize_surface_image(image)
    roi = crop_surface_roi(norm, border_frac=0.06)

    glare_mask = detect_glare_mask(roi)
    valid_mask = build_valid_surface_mask(roi, glare_mask)

    glare_fraction = np.count_nonzero(glare_mask) / float(glare_mask.size)
    valid_surface_fraction = np.count_nonzero(valid_mask) / float(valid_mask.size)

    scratch_score = compute_scratch_score(roi, valid_mask)
    speckle_score = compute_speckle_score(roi, valid_mask)
    gloss_score = compute_gloss_score(roi, glare_mask)

    if valid_surface_fraction < 0.55:
        scratch_score *= 0.90
        speckle_score *= 0.92

    surface_score = (
        0.56 * scratch_score +
        0.21 * speckle_score +
        0.14 * gloss_score
    )

    if valid_surface_fraction < 0.50:
        surface_score += 0.01
    if valid_surface_fraction < 0.35:
        surface_score += 0.02

    surface_score = float(np.clip(surface_score, 0.0, 1.0))

    confidence = float(
        np.clip(valid_surface_fraction * (1.0 - 0.35 * glare_fraction), 0.0, 1.0)
    )

    return {
        "surface_score": round(surface_score, 4),
        "scratch_score": round(float(np.clip(scratch_score, 0.0, 1.0)), 4),
        "speckle_score": round(float(np.clip(speckle_score, 0.0, 1.0)), 4),
        "gloss_score": round(float(np.clip(gloss_score, 0.0, 1.0)), 4),
        "confidence": round(confidence, 4),
        "glare_fraction": round(float(glare_fraction), 4),
        "valid_surface_fraction": round(float(valid_surface_fraction), 4),
    }


@app.get("/")
def root():
    return {"status": "ok"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            return {"error": "Empty file"}

        image = decode_image(contents, max_dim=1200)

        if image is None:
            return {"error": "Invalid image"}

        result = raw_engine.analyze_array(image)

        if not isinstance(result, dict):
            return {"error": "Centering engine returned invalid result"}

        result = enrich_centering_coordinates(result, image)
        return result

    except Exception as e:
        return {"error": str(e)}


@app.post("/analyze_corner")
async def analyze_corner(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            return {"error": "Empty file"}

        image = decode_image(contents, max_dim=400)

        if image is None:
            return {"error": "Invalid image"}

        score = corner_engine.analyze_patch(image, orientation="top_left")

        return {
            "corner_score": score,
            "confidence": 1.0
        }

    except Exception as e:
        return {"error": str(e)}


@app.post("/analyze_surface")
async def analyze_surface_route(file: UploadFile = File(...)):
    try:
        contents = await file.read()

        if not contents:
            return {"error": "Empty file"}

        image = decode_image(contents, max_dim=1200)

        if image is None:
            return {"error": "Invalid image"}

        result = analyze_surface(image)
        return result

    except Exception as e:
        return {"error": str(e)}
