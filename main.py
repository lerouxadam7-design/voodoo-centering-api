from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from professional_centering import VoodooRawEngine

app = FastAPI()

engine = VoodooRawEngine()

@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    return engine.analyze_array(image)
