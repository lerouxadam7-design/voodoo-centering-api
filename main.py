from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from professional_centering import VoodooCornerEngine

# Initialize FastAPI app FIRST
app = FastAPI()

# Initialize engine
corner_engine = VoodooCornerEngine()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    result = corner_engine.analyze_array(image)

    return result
