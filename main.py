from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2

from professional_centering import VoodooRawEngine, VoodooCornerCloseupEngine

app = FastAPI()

raw_engine = VoodooRawEngine()
corner_engine = VoodooCornerCloseupEngine()


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

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

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

        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            return {"error": "Invalid image"}

        score = corner_engine.analyze_patch(image, orientation="top_left")

        return {
            "corner_score": score,
            "confidence": 1.0
        }

    except Exception as e:
        return {"error": str(e)}
