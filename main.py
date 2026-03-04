from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from professional_centering import VoodooPredictiveCentering

app = FastAPI()

centering_engine = VoodooPredictiveCentering()


@app.post("/analyze")
async def analyze(file: UploadFile = File(...)):

    # Read uploaded file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if image is None:
        return {"error": "Invalid image"}

    # Run centering analysis
    result = centering_engine.analyze_array(image)

    return result
