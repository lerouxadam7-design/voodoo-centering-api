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


def analyze_surface(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # =====================================================
    # 1. Scratch detection
    # =====================================================
    edges = cv2.Canny(gray, 50, 150)
    scratch_score = np.mean(edges) / 255.0

    # =====================================================
    # 2. Speckle detection
    # =====================================================
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    diff = cv2.absdiff(gray, blur)
    speckle_score = np.mean(diff) / 255.0

    # =====================================================
    # 3. Gloss / texture variation
    # =====================================================
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    gloss_score = np.var(lap) / 1000.0

    # =====================================================
    # Combine
    # =====================================================
    surface_score = (
        0.4 * scratch_score +
        0.3 * speckle_score +
        0.3 * gloss_score
    )

    return {
        "surface_score": float(surface_score),
        "scratch_score": float(scratch_score),
        "speckle_score": float(speckle_score),
        "gloss_score": float(gloss_score),
        "confidence": 1.0
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
