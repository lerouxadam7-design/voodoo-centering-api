from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from professional_centering import VoodooSlabCentering

app = FastAPI()

centering_engine = VoodooSlabCentering()

@app.post("/analyze")

    if image is None:
        return {"error": "Invalid image"}

    result = centering_engine.analyze_array(image)

    return result
