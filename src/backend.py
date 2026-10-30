# Phase 2, Step 5: FastAPI backend. Run: uvicorn backend:app --host 0.0.0.0 --port 8000
import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile

from vision import Detector, analyze
from agent import decide_both

app = FastAPI(title="VisionPath AI")
detector = Detector()


def decode(data: bytes):
    frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    if w > 1280:
        frame = cv2.resize(frame, (1280, int(h * 1280 / w)))
    return frame


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/analyze")
async def analyze_image(file: UploadFile = File(...), ocr: bool = Form(True)):
    frame = decode(await file.read())
    return analyze(detector, frame, do_ocr=ocr)


@app.post("/guide")
async def guide(file: UploadFile = File(...), language: str = Form("English"),
                question: str = Form("")):
    frame = decode(await file.read())
    result = analyze(detector, frame, do_ocr=True)
    result["instructions"] = decide_both(result, question or None)
    return result